"""Build authenticated region-method-modality units for habitat ROI radiomics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from radiomicviz._integrity import fingerprint_matches, sha256_file
from radiomicviz.whole.plan import WholePlanError, WholeUnit, load_whole_plan

METHODS = ("fixed_k2", "adaptive_k1to3", "geometric_1mm")
MODALITIES = ("T1", "FLAIR")
_INDEX_COLUMNS = {
    "subject_id", "session_id", "region_id", "region_type", "variant",
    "method", "zone_id", "zone_name", "n_voxels", "mask_label_value",
}


class HabitatFeaturePlanError(ValueError):
    """Upstream habitat outputs cannot define a safe extraction plan."""


@dataclass(frozen=True)
class HabitatZone:
    zone_id: int
    zone_name: str
    label: int
    n_voxels: int
    habitat_grid_voxels: int
    missing: bool
    eligible_for_split: str
    chosen_k: str
    silhouette_score: str
    habitat_valid: str


@dataclass(frozen=True)
class HabitatFeatureUnit:
    subject_id: str
    session_id: str
    region_id: str
    region_type: str
    method: str
    modality: str
    image_path: Path
    image_sha256: str
    mask_path: Path
    mask_sha256: str
    habitat_completion_path: Path
    habitat_completion_sha256: str
    zones: tuple[HabitatZone, ...]
    metadata: tuple[tuple[str, str], ...]

    @property
    def unit_key(self) -> str:
        return (
            f"{self.subject_id}__{self.session_id}__{self.region_id}__"
            f"{self.method}__{self.modality}"
        )

    def output_dir(self, output_root: Path, variant: str) -> Path:
        return (
            output_root / "subjects" / self.subject_id / "sessions" / self.session_id
            / "regions" / self.region_id / "habitat_features" / self.method
            / variant / self.modality
        )


@dataclass(frozen=True)
class HabitatFeaturePlan:
    manifest_path: Path
    manifest_sha256: str
    output_root: Path
    manifest_reference_status: str
    habitat_reference_status: str
    habitat_index_path: Path
    habitat_index_sha256: str
    habitat_metadata_path: Path
    units: tuple[HabitatFeatureUnit, ...]


def load_habitat_feature_plan(
    manifest_path: str | Path, *, variant: str = "b32"
) -> HabitatFeaturePlan:
    """Load completed habitat masks and pair every non-empty zone with T1/FLAIR."""
    try:
        whole = load_whole_plan(manifest_path)
    except WholePlanError as exc:
        raise HabitatFeaturePlanError(str(exc)) from exc
    index_path = whole.output_root / "tables" / "habitats" / variant / "habitat_index.csv"
    metadata_path = whole.output_root / "qc" / f"habitats_{variant}_run_metadata.json"
    metadata = _load_json(metadata_path, "habitat run metadata")
    if metadata.get("status") != "complete":
        raise HabitatFeaturePlanError("habitat stage is not complete")
    if metadata.get("manifest_sha256") != whole.manifest_sha256:
        raise HabitatFeaturePlanError("habitat stage was built from a different manifest")
    if metadata.get("variant") != variant:
        raise HabitatFeaturePlanError("habitat run metadata records a different variant")
    habitat_status = (
        "reference-compatible" if metadata.get("reference_compatible") is True
        else "non-reference"
    )
    if not index_path.is_file():
        raise HabitatFeaturePlanError(f"habitat index is missing: {index_path}")

    with index_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_INDEX_COLUMNS - fields)
        if missing:
            raise HabitatFeaturePlanError(
                "habitat index is missing required column(s): " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise HabitatFeaturePlanError("habitat index contains no rows")

    images: dict[tuple[str, str, str, str], WholeUnit] = {
        (u.subject_id, u.session_id, u.region_id, u.modality): u for u in whole.units
    }
    grouped: dict[tuple[str, str, str, str], list[HabitatZone]] = {}
    region_types: dict[tuple[str, str, str], str] = {}
    seen_zone_keys: set[tuple[str, str, str, str, int]] = set()
    for line, row in enumerate(rows, start=2):
        if row.get("variant") != variant:
            raise HabitatFeaturePlanError(
                f"habitat index row {line} records variant {row.get('variant')!r}"
            )
        method = row.get("method", "")
        if method not in METHODS:
            raise HabitatFeaturePlanError(
                f"habitat index row {line} has unsupported method {method!r}"
            )
        subject = row.get("subject_id", "")
        session = row.get("session_id", "")
        region = row.get("region_id", "")
        key3 = (subject, session, region)
        if not any((subject, session, region, modality) in images for modality in MODALITIES):
            raise HabitatFeaturePlanError(
                f"habitat index row {line} does not match a manifest region"
            )
        zone_id = _integer(row.get("zone_id", ""), "zone_id", line, minimum=0)
        label = _integer(row.get("mask_label_value", ""), "mask_label_value", line, minimum=1)
        n_voxels = _integer(row.get("n_voxels", ""), "n_voxels", line, minimum=0)
        zone_key = (*key3, method, zone_id)
        if zone_key in seen_zone_keys:
            raise HabitatFeaturePlanError(f"duplicate habitat zone at row {line}: {zone_key}")
        seen_zone_keys.add(zone_key)
        region_types[key3] = row.get("region_type", "")
        if n_voxels == 0:
            continue
        zone = HabitatZone(
            zone_id=zone_id,
            zone_name=row.get("zone_name", ""),
            label=label,
            n_voxels=n_voxels,
            habitat_grid_voxels=n_voxels,
            missing=_boolean(row.get("missing", "0")),
            eligible_for_split=row.get("eligible_for_split", ""),
            chosen_k=row.get("chosen_k", ""),
            silhouette_score=row.get("silhouette_score", ""),
            habitat_valid=row.get("habitat_valid", ""),
        )
        grouped.setdefault((*key3, method), []).append(zone)

    units: list[HabitatFeatureUnit] = []
    completion_cache: dict[tuple[str, str, str], tuple[Path, dict, str]] = {}
    mask_cache: dict[tuple[str, str, str, str], tuple[Path, str, tuple[int, ...]]] = {}
    for group_key in sorted(grouped):
        subject, session, region, method = group_key
        zones = tuple(sorted(grouped[group_key], key=lambda zone: zone.zone_id))
        completion_key = (subject, session, region)
        cached = completion_cache.get(completion_key)
        if cached is None:
            completion_path = _region_root(whole.output_root, *completion_key) / "habitats" / "reference" / variant / "completion.json"
            completion = _load_json(completion_path, "habitat completion")
            _validate_completion(completion, whole.manifest_sha256, variant, completion_key)
            cached = (completion_path, completion, sha256_file(completion_path))
            completion_cache[completion_key] = cached
        completion_path, completion, completion_sha = cached
        source_outputs = completion.get("source_grid_outputs", {})
        recorded = source_outputs.get(method)
        if not isinstance(recorded, dict):
            raise HabitatFeaturePlanError(
                f"habitat completion lacks a source-grid {method} mask; rerun habitats"
            )
        mask_key = (*completion_key, method)
        mask_cached = mask_cache.get(mask_key)
        if mask_cached is None:
            mask_path = _source_mask_path(whole.output_root, subject, session, region, method, variant)
            if not fingerprint_matches(recorded, mask_path):
                raise HabitatFeaturePlanError(
                    f"source-grid habitat mask is missing or changed: {mask_path}"
                )
            mask_sha = str(recorded.get("sha256", ""))
            source_counts = _source_zone_counts(mask_path, zones)
            mask_cached = (mask_path, mask_sha, source_counts)
            mask_cache[mask_key] = mask_cached
        mask_path, mask_sha, source_counts = mask_cached
        zones = tuple(
            replace(zone, n_voxels=source_count)
            for zone, source_count in zip(zones, source_counts)
        )
        for modality in MODALITIES:
            image = images.get((subject, session, region, modality))
            if image is None:
                raise HabitatFeaturePlanError(
                    f"manifest region {subject}/{session}/{region} lacks canonical modality {modality}"
                )
            units.append(
                HabitatFeatureUnit(
                    subject_id=subject,
                    session_id=session,
                    region_id=region,
                    region_type=region_types[completion_key],
                    method=method,
                    modality=modality,
                    image_path=image.image_path,
                    image_sha256=image.image_sha256,
                    mask_path=mask_path,
                    mask_sha256=mask_sha,
                    habitat_completion_path=completion_path,
                    habitat_completion_sha256=completion_sha,
                    zones=zones,
                    metadata=image.metadata,
                )
            )
    if not units:
        raise HabitatFeaturePlanError("habitat index contains no non-empty zones")
    return HabitatFeaturePlan(
        manifest_path=whole.manifest_path,
        manifest_sha256=whole.manifest_sha256,
        output_root=whole.output_root,
        manifest_reference_status=whole.manifest_reference_status,
        habitat_reference_status=habitat_status,
        habitat_index_path=index_path,
        habitat_index_sha256=sha256_file(index_path),
        habitat_metadata_path=metadata_path,
        units=tuple(units),
    )


def _region_root(root: Path, subject: str, session: str, region: str) -> Path:
    return root / "subjects" / subject / "sessions" / session / "regions" / region


def _source_mask_path(
    root: Path, subject: str, session: str, region: str, method: str, variant: str
) -> Path:
    base = _region_root(root, subject, session, region) / "habitats" / method
    return (base / "mask_source_grid.nii.gz") if method == "geometric_1mm" else (base / variant / "mask_source_grid.nii.gz")


def _validate_completion(
    data: dict, manifest_sha: str, variant: str, key: tuple[str, str, str]
) -> None:
    expected = {
        "stage": "habitats", "status": "complete", "manifest_sha256": manifest_sha,
        "variant": variant, "unit_key": "__".join(key),
    }
    if any(data.get(name) != value for name, value in expected.items()):
        raise HabitatFeaturePlanError(
            f"habitat completion does not match {key[0]}/{key[1]}/{key[2]}"
        )


def _source_zone_counts(
    mask_path: Path, zones: tuple[HabitatZone, ...]
) -> tuple[int, ...]:
    try:
        values = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    except RuntimeError as exc:
        raise HabitatFeaturePlanError(f"cannot read source-grid habitat mask: {mask_path}") from exc
    return tuple(int(np.count_nonzero(values == zone.label)) for zone in zones)


def _load_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HabitatFeaturePlanError(f"{description} is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise HabitatFeaturePlanError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise HabitatFeaturePlanError(f"{description} is not a JSON object: {path}")
    return value


def _integer(value: str, name: str, line: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HabitatFeaturePlanError(
            f"habitat index row {line}: {name} must be an integer"
        ) from exc
    if str(parsed) != str(value).strip() or parsed < minimum:
        raise HabitatFeaturePlanError(
            f"habitat index row {line}: {name} must be an integer >= {minimum}"
        )
    return parsed


def _boolean(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}
