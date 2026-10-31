"""Cohort, compatibility, QC, and provenance reports for habitat features."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from radiomicviz import __version__
from radiomicviz.habitat_features.models import HabitatFeatureRunResult
from radiomicviz.habitat_features.plan import METHODS, MODALITIES

_IDENTITY_FIELDS = (
    "subject_id", "session_id", "region_id", "region_type", "method",
    "zone_id", "zone_name", "mask_label_value", "source_zone_voxels",
    "habitat_grid_voxels",
    "zone_flagged_small", "habitat_missing_flag", "eligible_for_split",
    "chosen_k", "silhouette_score", "habitat_valid", "modality", "variant",
    "reference_status", "image_path", "mask_path",
    "zone_status", "diagnostics_original_voxels", "diagnostics_interpolated_voxels",
)
_REFERENCE_FIELDS = (
    "ID", "Exam", "Label", "Rim", "Image", "Mask", "Cluster",
    "NumofVoxels", "Patient", "Reader",
)


@dataclass(frozen=True)
class HabitatFeatureReportTargets:
    table: Path
    method_tables: Mapping[str, Path]
    reference_tables: Mapping[str, Path]
    checks: Path
    errors: Path
    metadata: Path
    log: Path
    backups: tuple[Path, ...]


def prepare_habitat_feature_report_targets(
    output_root: Path, variant: str
) -> HabitatFeatureReportTargets:
    table_dir = output_root / "tables" / "habitat_features" / variant
    qc_dir = output_root / "qc"
    log_dir = qc_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    table = table_dir / "habitat_features.csv"
    method_tables = {
        method: table_dir / f"habitat_features_{method}.csv" for method in METHODS
    }
    reference_tables = {
        f"{method}:{modality}": table_dir / (
            f"habitat_features_{method}_{modality}_reference_format.csv"
        )
        for method in METHODS for modality in MODALITIES
    }
    checks = qc_dir / f"habitat_features_{variant}_qc.csv"
    errors = qc_dir / f"habitat_features_{variant}_errors.csv"
    metadata = qc_dir / f"habitat_features_{variant}_run_metadata.json"
    log = log_dir / f"habitat_features_{variant}.log"
    all_paths = (
        table, *method_tables.values(), *reference_tables.values(),
        checks, errors, metadata, log,
    )
    backups: list[Path] = []
    for path in all_paths:
        if path.exists():
            backup = _backup_path(path)
            path.replace(backup)
            backups.append(backup)
    return HabitatFeatureReportTargets(
        table, method_tables, reference_tables, checks, errors, metadata, log,
        tuple(backups),
    )


def write_habitat_feature_report(
    result: HabitatFeatureRunResult,
    targets: HabitatFeatureReportTargets,
) -> HabitatFeatureRunResult:
    rows = [dict(row) for unit in result.units for row in unit.rows]
    diagnostics, features = _consistent_schema(result)
    meta_fields = _metadata_fields(rows)
    fields = (
        *_IDENTITY_FIELDS, *meta_fields, "error", *diagnostics, *features,
    )
    _write_csv(targets.table, rows, fields)
    for method, path in targets.method_tables.items():
        _write_csv(path, [row for row in rows if row.get("method") == method], fields)
    for key, path in targets.reference_tables.items():
        method, modality = key.split(":", 1)
        selected = [
            row for row in rows
            if row.get("method") == method and row.get("modality") == modality
        ]
        reference_rows = [
            _reference_row(row, index, diagnostics, features)
            for index, row in enumerate(selected, start=1)
        ]
        _write_csv(
            path, reference_rows, (*_REFERENCE_FIELDS, *diagnostics, *features)
        )
    _write_csv(
        targets.checks,
        [check.to_dict() for check in result.checks],
        (
            "check_id", "status", "unit_key", "subject_id", "session_id",
            "region_id", "method", "zone_id", "zone_name", "modality",
            "measured_value", "threshold", "message", "path",
        ),
    )
    _write_csv(
        targets.errors,
        [failure.to_dict() for failure in result.failures],
        (
            "error_code", "message", "unit_key", "subject_id", "session_id",
            "region_id", "method", "zone_id", "zone_name", "modality", "path",
        ),
    )
    metadata = {
        "stage": "habitat-features",
        "status": "complete" if result.valid else "failed",
        "started_utc": result.started_utc,
        "finished_utc": result.finished_utc,
        "package_version": __version__,
        "software": dict(result.software),
        "reference_compatible": result.reference_compatible,
        "reference_deviations": list(result.reference_deviations),
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "habitat_index_path": str(result.habitat_index_path),
        "habitat_index_sha256": result.habitat_index_sha256,
        "protocol_id": result.protocol_id,
        "protocol_version": result.protocol_version,
        "protocol_sha256": result.protocol_sha256,
        "variant": result.variant,
        "preset_id": result.preset_id,
        "preset_path": str(result.preset_path),
        "preset_sha256": result.preset_sha256,
        "jobs": result.jobs,
        "units": len(result.units),
        "completed": result.completed,
        "resumed": result.resumed,
        "blocked": result.blocked,
        "zone_rows": result.zone_rows,
        "successful_zones": result.successful_zones,
        "skipped_zones": result.skipped_zones,
        "failed_zones": result.failed,
        "warnings": result.warnings,
        "diagnostics_count": len(diagnostics),
        "feature_count": len(features),
        "feature_schema_sha256": _schema_hash(features) if features else None,
        "archived_reports": [str(path) for path in targets.backups],
        "unit_results": [
            {
                "unit_key": unit.unit_key,
                "status": unit.status,
                "zone_rows": len(unit.rows),
                "output_dir": str(unit.output_dir),
                "completion_path": str(unit.completion_path) if unit.completion_path else None,
            }
            for unit in result.units
        ],
        "outputs": {
            "habitat_features": str(targets.table),
            "method_tables": {k: str(v) for k, v in targets.method_tables.items()},
            "reference_format_tables": {k: str(v) for k, v in targets.reference_tables.items()},
            "qc": str(targets.checks),
            "errors": str(targets.errors),
            "log": str(targets.log),
        },
    }
    _atomic_text(targets.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log_lines = [
        "RadiomicsViz habitat-feature stage",
        f"status: {metadata['status']}",
        f"manifest: {result.manifest_path}",
        f"habitat_index: {result.habitat_index_path}",
        f"protocol: {result.protocol_id} {result.protocol_version}",
        f"variant: {result.variant}",
        f"preset_sha256: {result.preset_sha256}",
        f"reference_deviations: {'; '.join(result.reference_deviations) or 'none'}",
        f"jobs: {result.jobs}",
        f"method-modality units: {len(result.units)}",
        f"completed: {result.completed}",
        f"resumed: {result.resumed}",
        f"zone rows: {result.zone_rows}",
        f"successful zones: {result.successful_zones}",
        f"expected skipped zones: {result.skipped_zones}",
        f"failed zones: {result.failed}",
        f"warnings: {result.warnings}",
        f"features: {len(features)}",
        *(f"archived previous report: {path}" for path in targets.backups),
        "",
        *result.log_lines,
    ]
    _atomic_text(targets.log, "\n".join(log_lines).rstrip() + "\n")
    return replace(
        result,
        backups=targets.backups,
        table_path=targets.table,
        method_table_paths=targets.method_tables,
        reference_table_paths=targets.reference_tables,
        checks_path=targets.checks,
        errors_path=targets.errors,
        metadata_path=targets.metadata,
        log_path=targets.log,
    )


def _consistent_schema(
    result: HabitatFeatureRunResult,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schemas = {
        (unit.diagnostic_names, unit.feature_names)
        for unit in result.units if unit.feature_names
    }
    if len(schemas) > 1:
        raise ValueError("completed habitat-feature units do not share one output schema")
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
    patient_number: int,
    diagnostics: tuple[str, ...],
    features: tuple[str, ...],
) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "ID": row.get("subject_id", ""),
        "Exam": row.get("session_id", ""),
        "Label": row.get("region_id", ""),
        "Rim": row.get("meta_rim", ""),
        "Image": row.get("image_path", ""),
        "Mask": row.get("mask_path", ""),
        "Cluster": row.get("mask_label_value", ""),
        "NumofVoxels": row.get("source_zone_voxels", ""),
        "Patient": patient_number,
        "Reader": "N-A",
    }
    converted.update((name, row.get(name, "")) for name in (*diagnostics, *features))
    return converted


def _write_csv(
    path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str]
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), restval="", extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _schema_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate
