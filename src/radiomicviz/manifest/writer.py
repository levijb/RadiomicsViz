"""Atomic output writing for a reviewed manifest preview."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from radiomicviz import __version__
from radiomicviz.manifest.models import ManifestPreview


@dataclass(frozen=True)
class ManifestWriteResult:
    manifest_path: Path | None
    checks_path: Path
    errors_path: Path
    metadata_path: Path
    log_path: Path
    backups: tuple[Path, ...]


def write_manifest_outputs(preview: ManifestPreview) -> ManifestWriteResult:
    """Write a preview's concise artifacts atomically.

    Artifacts from a previous write, including the manifest itself, are first renamed
    with a ``.bak-<timestamp>`` suffix, so nothing is overwritten in place and the set of
    artifacts always describes one write. A preview with blocking failures writes only
    its diagnostics.
    """
    manifest_dir = preview.output_root / "manifest"
    log_dir = preview.output_root / "qc" / "logs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "regions_manifest.csv"
    checks_path = manifest_dir / "input_checks.csv"
    errors_path = manifest_dir / "errors.csv"
    metadata_path = manifest_dir / "run_metadata.json"
    log_path = log_dir / "manifest.log"
    targets = (manifest_path, checks_path, errors_path, metadata_path, log_path)
    existing = [path for path in targets if path.exists()]

    backups: list[Path] = []
    for path in existing:
        backup = _backup_path(path)
        path.replace(backup)
        backups.append(backup)

    if preview.valid:
        _write_csv(manifest_path, [row.to_dict() for row in preview.rows])
    _write_csv(checks_path, [row.to_dict() for row in preview.checks], default_fields=_check_fields())
    _write_csv(errors_path, [row.to_dict() for row in preview.errors], default_fields=_error_fields())

    project_hash = _sha256(preview.project_path)
    manifest_hash = _sha256(manifest_path) if preview.valid else None
    now = datetime.now(tz=timezone.utc).isoformat()
    metadata = {
        "stage": "manifest",
        "status": "complete" if preview.valid else "failed",
        "finished_utc": now,
        "package_version": __version__,
        "project_path": str(preview.project_path),
        "project_sha256": project_hash,
        "data_root": str(preview.data_root),
        "output_root": str(preview.output_root),
        "reference_status": preview.reference_status,
        "subjects": preview.subjects_found,
        "sessions": preview.sessions_found,
        "regions": preview.regions_found,
        "manifest_rows": len(preview.rows) if preview.valid else 0,
        "warnings": preview.warning_count,
        "failures": preview.failure_count,
        "manifest_sha256": manifest_hash,
        "outputs": {
            "regions_manifest": str(manifest_path) if preview.valid else None,
            "input_checks": str(checks_path),
            "errors": str(errors_path),
            "log": str(log_path),
        },
    }
    _atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log = (
        "RadiomicsViz manifest stage\n"
        f"status: {metadata['status']}\n"
        f"project: {preview.project_path}\n"
        f"subjects: {preview.subjects_found}\n"
        f"sessions: {preview.sessions_found}\n"
        f"regions: {preview.regions_found}\n"
        f"rows: {metadata['manifest_rows']}\n"
        f"warnings: {preview.warning_count}\n"
        f"failures: {preview.failure_count}\n"
        f"reference_status: {preview.reference_status}\n"
    )
    _atomic_text(log_path, log)
    return ManifestWriteResult(
        manifest_path=manifest_path if preview.valid else None,
        checks_path=checks_path, errors_path=errors_path, metadata_path=metadata_path,
        log_path=log_path, backups=tuple(backups),
    )


def _write_csv(
    path: Path, rows: list[dict[str, object]], default_fields: Iterable[str] = (),
) -> None:
    fields = list(rows[0]) if rows else list(default_fields)
    temp = _temporary_path(path)
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_text(path: Path, text: str) -> None:
    temp = _temporary_path(path)
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _temporary_path(path: Path) -> Path:
    # A unique name, so a file left by an interrupted write cannot block this one.
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_fields() -> tuple[str, ...]:
    return (
        "check_id", "status", "subject_id", "session_id", "region_id", "modality",
        "path", "measured_value", "threshold", "message",
    )


def _error_fields() -> tuple[str, ...]:
    return (
        "error_code", "message", "subject_id", "session_id", "region_id", "modality", "path",
    )
