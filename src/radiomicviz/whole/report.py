"""Cohort tables and concise reports for whole-region extraction."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from radiomicviz import __version__
from radiomicviz.whole.models import WholeRunResult

_IDENTITY_FIELDS = (
    "subject_id",
    "session_id",
    "region_id",
    "region_type",
    "modality",
    "variant",
    "reference_status",
    "image_path",
    "mask_path",
    "label",
    "mask_voxels",
)
_REFERENCE_FIELDS = (
    "ID",
    "Exam",
    "Label",
    "Rim",
    "Image",
    "Mask",
    "NumofVoxels",
    "Patient",
    "Reader",
)


@dataclass(frozen=True)
class WholeReportTargets:
    table: Path
    reference_table: Path
    checks: Path
    errors: Path
    metadata: Path
    log: Path
    backups: tuple[Path, ...]


def prepare_whole_report_targets(output_root: Path, variant: str) -> WholeReportTargets:
    """Create report folders and archive the preceding invocation's reports."""
    table_dir = output_root / "tables" / "whole" / variant
    qc_dir = output_root / "qc"
    log_dir = qc_dir / "logs"
    table = table_dir / "whole_region.csv"
    reference = table_dir / "whole_region_reference_format.csv"
    checks = qc_dir / f"whole_{variant}_qc.csv"
    errors = qc_dir / f"whole_{variant}_errors.csv"
    metadata = qc_dir / f"whole_{variant}_run_metadata.json"
    log = log_dir / f"whole_{variant}.log"
    table_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    targets = (table, reference, checks, errors, metadata, log)
    backups: list[Path] = []
    for path in targets:
        if path.exists():
            backup = _backup_path(path)
            path.replace(backup)
            backups.append(backup)
    return WholeReportTargets(
        table, reference, checks, errors, metadata, log, tuple(backups)
    )


def write_whole_report(
    result: WholeRunResult, targets: WholeReportTargets
) -> WholeRunResult:
    """Write cohort tables, QC, errors, metadata and a readable log."""
    rows = [dict(unit.row) for unit in result.units]
    diagnostic_names, feature_names = _consistent_schema(result)
    meta_fields = _metadata_fields(rows)
    tidy_fields = (
        *_IDENTITY_FIELDS,
        *meta_fields,
        "error",
        *diagnostic_names,
        *feature_names,
    )
    _write_csv(targets.table, rows, tidy_fields)
    reference_rows = [
        _reference_row(row, diagnostic_names, feature_names) for row in rows
    ]
    _write_csv(
        targets.reference_table,
        reference_rows,
        (*_REFERENCE_FIELDS, *diagnostic_names, *feature_names),
    )
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
            "modality",
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
            "modality",
            "path",
        ),
    )
    metadata = {
        "stage": "whole",
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
        "preset_id": result.preset_id,
        "preset_path": str(result.preset_path),
        "preset_sha256": result.preset_sha256,
        "jobs": result.jobs,
        "units": len(result.units),
        "completed": result.succeeded,
        "resumed": result.resumed,
        "skipped": result.skipped,
        "failed": result.failed,
        "warnings": result.warnings,
        "diagnostics_count": len(diagnostic_names),
        "feature_count": len(feature_names),
        "feature_schema_sha256": _schema_hash(feature_names) if feature_names else None,
        "archived_reports": [str(path) for path in targets.backups],
        "unit_results": [
            {
                "unit_key": unit.unit_key,
                "status": unit.status,
                "output_dir": str(unit.output_dir),
                "completion_path": (
                    str(unit.completion_path) if unit.completion_path else None
                ),
            }
            for unit in result.units
        ],
        "outputs": {
            "whole_region": str(targets.table),
            "reference_format": str(targets.reference_table),
            "qc": str(targets.checks),
            "errors": str(targets.errors),
            "log": str(targets.log),
        },
    }
    _atomic_text(targets.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log_lines = [
        "RadiomicsViz whole-region stage",
        f"status: {metadata['status']}",
        f"manifest: {result.manifest_path}",
        f"protocol: {result.protocol_id} {result.protocol_version}",
        f"variant: {result.variant}",
        f"preset_sha256: {result.preset_sha256}",
        f"reference_deviations: {'; '.join(result.reference_deviations) or 'none'}",
        f"jobs: {result.jobs}",
        f"units: {len(result.units)}",
        f"completed: {result.succeeded}",
        f"resumed: {result.resumed}",
        f"skipped: {result.skipped}",
        f"failed: {result.failed}",
        f"warnings: {result.warnings}",
        f"features: {len(feature_names)}",
        *(f"archived previous report: {path}" for path in targets.backups),
        "",
        *result.log_lines,
    ]
    _atomic_text(targets.log, "\n".join(log_lines).rstrip() + "\n")
    return replace(
        result,
        backups=targets.backups,
        table_path=targets.table,
        reference_table_path=targets.reference_table,
        checks_path=targets.checks,
        errors_path=targets.errors,
        metadata_path=targets.metadata,
        log_path=targets.log,
    )


def _consistent_schema(result: WholeRunResult) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schemas = {
        (unit.diagnostic_names, unit.feature_names)
        for unit in result.units
        if unit.feature_names
    }
    if len(schemas) > 1:
        raise ValueError("completed whole-region units do not share one output schema")
    return next(iter(schemas), ((), ()))


def _metadata_fields(rows: list[Mapping[str, Any]]) -> tuple[str, ...]:
    found: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("meta_") and key not in found:
                found.append(key)
    return tuple(found)


def _reference_row(
    row: Mapping[str, Any],
    diagnostics: tuple[str, ...],
    features: tuple[str, ...],
) -> dict[str, Any]:
    region = row.get("region_id", "")
    converted: dict[str, Any] = {
        "ID": row.get("subject_id", ""),
        "Exam": row.get("session_id", ""),
        "Label": region,
        "Rim": row.get("meta_rim", ""),
        "Image": row.get("image_path", ""),
        "Mask": row.get("mask_path", ""),
        "NumofVoxels": row.get("mask_voxels", ""),
        "Patient": region,
        "Reader": "N-A",
    }
    converted.update((name, row.get(name, "")) for name in (*diagnostics, *features))
    return converted


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fields: Iterable[str],
) -> None:
    fieldnames = list(fields)
    temporary = _temporary_path(path)
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            restval="",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _schema_hash(names: tuple[str, ...]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


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
