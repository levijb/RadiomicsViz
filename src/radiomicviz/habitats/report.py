"""Cohort index, QC, error, provenance and log reports for habitats."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping
from uuid import uuid4

from radiomicviz import __version__
from radiomicviz.habitats.models import HabitatRunResult


@dataclass(frozen=True)
class HabitatReportTargets:
    table: Path
    lesion_qc: Path
    checks: Path
    errors: Path
    metadata: Path
    log: Path
    backups: tuple[Path, ...]


def prepare_habitat_report_targets(output_root: Path, variant: str) -> HabitatReportTargets:
    table_dir = output_root / "tables" / "habitats" / variant
    qc_dir = output_root / "qc"
    log_dir = qc_dir / "logs"
    table = table_dir / "habitat_index.csv"
    lesion_qc = table_dir / "lesion_qc.csv"
    checks = qc_dir / f"habitats_{variant}_qc.csv"
    errors = qc_dir / f"habitats_{variant}_errors.csv"
    metadata = qc_dir / f"habitats_{variant}_run_metadata.json"
    log = log_dir / f"habitats_{variant}.log"
    table_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    backups: list[Path] = []
    for path in (table, lesion_qc, checks, errors, metadata, log):
        if path.exists():
            backup = _backup_path(path)
            path.replace(backup)
            backups.append(backup)
    return HabitatReportTargets(
        table, lesion_qc, checks, errors, metadata, log, tuple(backups)
    )


def write_habitat_report(
    result: HabitatRunResult, targets: HabitatReportTargets
) -> HabitatRunResult:
    rows = [dict(row) for unit in result.units for row in unit.rows]
    qc_rows = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
            for key, value in unit.qc.items()
        }
        for unit in result.units
        if unit.qc is not None
    ]
    table_fields = (
        "subject_id",
        "session_id",
        "region_id",
        "region_type",
        "variant",
        "implementation_id",
        "reference_status",
        "method",
        "mask_path",
        "source_grid_mask_path",
        "zone_id",
        "zone_name",
        "n_voxels",
        "missing",
        "eligible_for_split",
        "chosen_k",
        "silhouette_score",
        "habitat_valid",
        "mask_label_value",
    )
    qc_fields = _ordered_fields(qc_rows, ("subject_id", "session_id", "region_id"))
    _write_csv(targets.table, rows, table_fields)
    _write_csv(targets.lesion_qc, qc_rows, qc_fields)
    _write_csv(
        targets.checks,
        [row.to_dict() for row in result.checks],
        (
            "check_id",
            "status",
            "unit_key",
            "subject_id",
            "session_id",
            "region_id",
            "measured_value",
            "threshold",
            "message",
            "path",
        ),
    )
    _write_csv(
        targets.errors,
        [row.to_dict() for row in result.failures],
        (
            "error_code",
            "message",
            "unit_key",
            "subject_id",
            "session_id",
            "region_id",
            "path",
        ),
    )
    metadata = {
        "stage": "habitats",
        "status": "complete" if result.valid else "failed",
        "started_utc": result.started_utc,
        "finished_utc": result.finished_utc,
        "package_version": __version__,
        "software": dict(result.software),
        "reference_compatible": result.reference_compatible,
        "reference_deviations": list(result.reference_deviations),
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "protocol_id": result.protocol_id,
        "protocol_version": result.protocol_version,
        "protocol_sha256": result.protocol_sha256,
        "variant": result.variant,
        "implementation_id": result.implementation_id,
        "scientific_config_sha256": result.scientific_config_sha256,
        "jobs": result.jobs,
        "units": len(result.units),
        "completed": result.succeeded,
        "resumed": result.resumed,
        "skipped": result.skipped,
        "failed": result.failed,
        "warnings": result.warnings,
        "archived_reports": [str(path) for path in targets.backups],
        "outputs": {
            "habitat_index": str(targets.table),
            "lesion_qc": str(targets.lesion_qc),
            "qc": str(targets.checks),
            "errors": str(targets.errors),
            "log": str(targets.log),
        },
    }
    _atomic_text(targets.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    lines = [
        "RadiomicsViz habitat stage",
        f"status: {metadata['status']}",
        f"manifest: {result.manifest_path}",
        f"protocol: {result.protocol_id} {result.protocol_version}",
        f"variant: {result.variant}",
        f"implementation: {result.implementation_id}",
        f"scientific_config_sha256: {result.scientific_config_sha256}",
        f"reference_deviations: {'; '.join(result.reference_deviations) or 'none'}",
        f"jobs: {result.jobs}",
        f"units: {len(result.units)}",
        f"completed: {result.succeeded}",
        f"resumed: {result.resumed}",
        f"skipped: {result.skipped}",
        f"failed: {result.failed}",
        f"warnings: {result.warnings}",
        *(f"archived previous report: {path}" for path in targets.backups),
        "",
        *result.log_lines,
    ]
    _atomic_text(targets.log, "\n".join(lines).rstrip() + "\n")
    return replace(
        result,
        backups=targets.backups,
        table_path=targets.table,
        qc_table_path=targets.lesion_qc,
        checks_path=targets.checks,
        errors_path=targets.errors,
        metadata_path=targets.metadata,
        log_path=targets.log,
    )


def _ordered_fields(
    rows: list[Mapping[str, object]], first: tuple[str, ...]
) -> tuple[str, ...]:
    names = {name for row in rows for name in row}
    return (*first, *sorted(names - set(first)))


def _write_csv(
    path: Path, rows: list[Mapping[str, object]], fields: Iterable[str]
) -> None:
    temporary = _temporary_path(path)
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            restval="",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate
