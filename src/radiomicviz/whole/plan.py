"""Load authenticated region-modality units for whole-region extraction."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from radiomicviz.config._common import SAFE_ID_PATTERN

_REQUIRED_COLUMNS = {
    "subject_id",
    "session_id",
    "region_id",
    "region_type",
    "modality",
    "image_path",
    "mask_path",
    "label",
    "image_size_bytes",
    "image_mtime_utc",
    "mask_size_bytes",
    "mask_mtime_utc",
}


class WholePlanError(ValueError):
    """The canonical manifest cannot define a safe whole-region plan."""


@dataclass(frozen=True)
class WholeUnit:
    """One unique region and modality from the canonical manifest."""

    subject_id: str
    session_id: str
    region_id: str
    region_type: str
    modality: str
    image_path: Path
    mask_path: Path
    label: int
    image_size_bytes: int
    image_mtime_utc: str
    mask_size_bytes: int
    mask_mtime_utc: str
    metadata: tuple[tuple[str, str], ...]

    @property
    def unit_key(self) -> str:
        return (
            f"{self.subject_id}__{self.session_id}__{self.region_id}__{self.modality}"
        )

    def output_dir(self, output_root: Path, variant: str) -> Path:
        return (
            output_root
            / "subjects"
            / self.subject_id
            / "sessions"
            / self.session_id
            / "regions"
            / self.region_id
            / "radiomics"
            / "whole"
            / variant
            / self.modality
        )


@dataclass(frozen=True)
class WholePlan:
    """Authenticated manifest and ordered region-modality units."""

    manifest_path: Path
    manifest_sha256: str
    output_root: Path
    manifest_reference_status: str
    units: tuple[WholeUnit, ...]


def load_whole_plan(path: str | Path) -> WholePlan:
    """Load the canonical manifest and verify all inputs needed by ``whole``."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise WholePlanError(f"manifest does not exist: {manifest_path}")
    if manifest_path.name != "regions_manifest.csv" or manifest_path.parent.name != "manifest":
        raise WholePlanError(
            "expected the canonical <output_root>/manifest/regions_manifest.csv path"
        )
    output_root = manifest_path.parent.parent.resolve()
    manifest_sha256 = _sha256(manifest_path)
    reference_status = _verify_manifest_metadata(manifest_path, manifest_sha256)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - set(fields))
        if missing:
            raise WholePlanError(
                "manifest is missing required column(s): " + ", ".join(missing)
            )
        meta_fields = tuple(field for field in fields if field.startswith("meta_"))
        rows = list(reader)
    if not rows:
        raise WholePlanError("manifest contains no rows")

    units: dict[tuple[str, str, str, str], WholeUnit] = {}
    for row_number, row in enumerate(rows, start=2):
        unit = _parse_unit(row, row_number, meta_fields)
        key = (unit.subject_id, unit.session_id, unit.region_id, unit.modality)
        if key in units:
            raise WholePlanError(
                f"manifest contains duplicate whole-region unit {unit.unit_key!r}"
            )
        units[key] = unit
    ordered = tuple(units[key] for key in sorted(units))
    for unit in ordered:
        _verify_unit_inputs(unit)
    return WholePlan(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        output_root=output_root,
        manifest_reference_status=reference_status,
        units=ordered,
    )


def _parse_unit(
    row: dict[str, str],
    row_number: int,
    meta_fields: tuple[str, ...],
) -> WholeUnit:
    subject = _identifier(row.get("subject_id", ""), "subject_id", row_number)
    session = _identifier(row.get("session_id", ""), "session_id", row_number)
    region = _identifier(row.get("region_id", ""), "region_id", row_number)
    region_type = _identifier(row.get("region_type", ""), "region_type", row_number)
    modality = _identifier(row.get("modality", ""), "modality", row_number)
    image = _absolute_path(row.get("image_path", ""), "image_path", row_number)
    mask = _absolute_path(row.get("mask_path", ""), "mask_path", row_number)
    label = _positive_integer(row.get("label", ""), "label", row_number)
    image_size = _positive_integer(
        row.get("image_size_bytes", ""), "image_size_bytes", row_number
    )
    image_mtime = row.get("image_mtime_utc", "")
    if not image_mtime:
        raise WholePlanError(f"manifest row {row_number}: image_mtime_utc is empty")
    mask_size = _positive_integer(
        row.get("mask_size_bytes", ""), "mask_size_bytes", row_number
    )
    mask_mtime = row.get("mask_mtime_utc", "")
    if not mask_mtime:
        raise WholePlanError(f"manifest row {row_number}: mask_mtime_utc is empty")
    metadata = tuple((field, row.get(field, "")) for field in meta_fields)
    return WholeUnit(
        subject_id=subject,
        session_id=session,
        region_id=region,
        region_type=region_type,
        modality=modality,
        image_path=image,
        mask_path=mask,
        label=label,
        image_size_bytes=image_size,
        image_mtime_utc=image_mtime,
        mask_size_bytes=mask_size,
        mask_mtime_utc=mask_mtime,
        metadata=metadata,
    )


def _verify_unit_inputs(unit: WholeUnit) -> None:
    _verify_fingerprint(
        unit.mask_path,
        unit.mask_size_bytes,
        unit.mask_mtime_utc,
        f"mask for {unit.unit_key}",
    )
    _verify_fingerprint(
        unit.image_path,
        unit.image_size_bytes,
        unit.image_mtime_utc,
        f"analysis-ready image for {unit.unit_key}",
    )


def _verify_manifest_metadata(manifest_path: Path, manifest_hash: str) -> str:
    metadata_path = manifest_path.with_name("run_metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WholePlanError(f"manifest metadata is missing: {metadata_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise WholePlanError(f"cannot read manifest metadata: {exc}") from exc
    if metadata.get("status") != "complete":
        raise WholePlanError("manifest metadata does not record a complete manifest stage")
    if metadata.get("manifest_sha256") != manifest_hash:
        raise WholePlanError(
            "manifest SHA-256 does not match run_metadata.json; rebuild or restore the manifest"
        )
    status = metadata.get("reference_status")
    if not isinstance(status, str) or not status:
        raise WholePlanError("manifest metadata does not record a reference status")
    return status


def _verify_fingerprint(path: Path, size: int, mtime: str, description: str) -> None:
    if not path.is_file():
        raise WholePlanError(f"{description} is missing: {path}")
    current = _fingerprint(path)
    if current["size_bytes"] != size or current["mtime_utc"] != mtime:
        raise WholePlanError(
            f"{description} changed after manifest creation; rebuild the manifest: {path}"
        )


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _identifier(value: str, name: str, row_number: int) -> str:
    if value in {"", ".", ".."} or SAFE_ID_PATTERN.fullmatch(value) is None:
        raise WholePlanError(
            f"manifest row {row_number}: unsafe or empty {name} {value!r}"
        )
    return value


def _absolute_path(value: str, name: str, row_number: int) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise WholePlanError(f"manifest row {row_number}: {name} must be absolute")
    return path.resolve()


def _positive_integer(value: str, name: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WholePlanError(
            f"manifest row {row_number}: {name} must be an integer"
        ) from exc
    if parsed < 1:
        raise WholePlanError(
            f"manifest row {row_number}: {name} must be at least 1"
        )
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
