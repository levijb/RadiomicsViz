"""Concise QC, error, provenance and log reports for voxelwise extraction."""

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
from radiomicviz.voxelwise.models import VoxelwiseRunResult


@dataclass(frozen=True)
class VoxelwiseReportTargets:
    checks: Path
    errors: Path
    metadata: Path
    log: Path
    backups: tuple[Path, ...]


def prepare_voxelwise_report_targets(
    output_root: Path, variant: str
) -> VoxelwiseReportTargets:
    """Create report folders and archive the preceding invocation's reports."""
    qc_dir = output_root / "qc"
    log_dir = qc_dir / "logs"
    checks = qc_dir / f"voxelwise_{variant}_qc.csv"
    errors = qc_dir / f"voxelwise_{variant}_errors.csv"
    metadata = qc_dir / f"voxelwise_{variant}_run_metadata.json"
    log = log_dir / f"voxelwise_{variant}.log"
    qc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    backups: list[Path] = []
    for path in (checks, errors, metadata, log):
        if path.exists():
            backup = _backup_path(path)
            path.replace(backup)
            backups.append(backup)
    return VoxelwiseReportTargets(checks, errors, metadata, log, tuple(backups))


def write_voxelwise_report(
    result: VoxelwiseRunResult, targets: VoxelwiseReportTargets
) -> VoxelwiseRunResult:
    """Write invocation-level QC, errors, provenance and a readable log."""
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
        "stage": "voxelwise",
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
        "archived_reports": [str(path) for path in targets.backups],
        "unit_results": [
            {
                "unit_key": unit.unit_key,
                "status": unit.status,
                "output_dir": str(unit.output_dir),
                "completion_path": (
                    str(unit.completion_path) if unit.completion_path else None
                ),
                "mask_voxels": unit.mask_voxels,
                "map_count": len(unit.map_names),
                "message": unit.message,
            }
            for unit in result.units
        ],
        "outputs": {
            "qc": str(targets.checks),
            "errors": str(targets.errors),
            "log": str(targets.log),
        },
    }
    _atomic_text(targets.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log_lines = [
        "RadiomicsViz voxelwise stage",
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
        *(f"archived previous report: {path}" for path in targets.backups),
        "",
        *result.log_lines,
    ]
    _atomic_text(targets.log, "\n".join(log_lines).rstrip() + "\n")
    return replace(
        result,
        backups=targets.backups,
        checks_path=targets.checks,
        errors_path=targets.errors,
        metadata_path=targets.metadata,
        log_path=targets.log,
    )


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
