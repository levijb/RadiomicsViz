"""Build one habitat unit per manifest region from paired modality rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from radiomicviz.whole.plan import WholePlanError, WholeUnit, load_whole_plan


class HabitatPlanError(ValueError):
    """The canonical manifest cannot define a safe habitat plan."""


@dataclass(frozen=True)
class HabitatUnit:
    """One region with its authoritative mask and paired voxelwise inputs."""

    subject_id: str
    session_id: str
    region_id: str
    region_type: str
    mask_path: Path
    label: int
    t1_unit: WholeUnit
    flair_unit: WholeUnit

    @property
    def unit_key(self) -> str:
        return f"{self.subject_id}__{self.session_id}__{self.region_id}"

    def voxelwise_dir(self, output_root: Path, variant: str, modality: str) -> Path:
        unit = self.t1_unit if modality == "T1" else self.flair_unit
        return (
            output_root
            / "subjects"
            / unit.subject_id
            / "sessions"
            / unit.session_id
            / "regions"
            / unit.region_id
            / "radiomics"
            / "voxelwise"
            / variant
            / modality
        )

    def output_dir(self, output_root: Path) -> Path:
        return (
            output_root
            / "subjects"
            / self.subject_id
            / "sessions"
            / self.session_id
            / "regions"
            / self.region_id
            / "habitats"
        )


@dataclass(frozen=True)
class HabitatPlan:
    """Authenticated manifest and ordered region-level habitat units."""

    manifest_path: Path
    manifest_sha256: str
    output_root: Path
    manifest_reference_status: str
    units: tuple[HabitatUnit, ...]


def load_habitat_plan(path: str | Path) -> HabitatPlan:
    """Load canonical rows and require exactly one T1 and FLAIR row per region."""
    try:
        source = load_whole_plan(path)
    except WholePlanError as exc:
        raise HabitatPlanError(str(exc)) from exc

    grouped: dict[tuple[str, str, str], dict[str, WholeUnit]] = {}
    for row in source.units:
        key = (row.subject_id, row.session_id, row.region_id)
        modalities = grouped.setdefault(key, {})
        if row.modality in modalities:
            raise HabitatPlanError(
                f"duplicate {row.modality} row for habitat unit {row.unit_key!r}"
            )
        modalities[row.modality] = row

    units: list[HabitatUnit] = []
    for key in sorted(grouped):
        modalities = grouped[key]
        missing = [name for name in ("T1", "FLAIR") if name not in modalities]
        if missing:
            tag = "__".join(key)
            raise HabitatPlanError(
                f"habitat unit {tag!r} is missing required modality row(s): "
                + ", ".join(missing)
            )
        t1, flair = modalities["T1"], modalities["FLAIR"]
        if (t1.mask_path, t1.label, t1.region_type) != (
            flair.mask_path,
            flair.label,
            flair.region_type,
        ):
            raise HabitatPlanError(
                f"T1 and FLAIR rows disagree on mask, label or region type for {'__'.join(key)}"
            )
        units.append(
            HabitatUnit(
                subject_id=t1.subject_id,
                session_id=t1.session_id,
                region_id=t1.region_id,
                region_type=t1.region_type,
                mask_path=t1.mask_path,
                label=t1.label,
                t1_unit=t1,
                flair_unit=flair,
            )
        )
    return HabitatPlan(
        manifest_path=source.manifest_path,
        manifest_sha256=source.manifest_sha256,
        output_root=source.output_root,
        manifest_reference_status=source.manifest_reference_status,
        units=tuple(units),
    )
