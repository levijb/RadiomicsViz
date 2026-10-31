"""Data contracts for ROI radiomics extracted from habitat zones."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HabitatFeatureCheck:
    check_id: str
    status: str
    unit_key: str
    subject_id: str
    session_id: str
    region_id: str
    method: str
    zone_id: int | str
    zone_name: str
    modality: str
    measured_value: str = ""
    threshold: str = ""
    message: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HabitatFeatureFailure:
    error_code: str
    message: str
    unit_key: str
    subject_id: str
    session_id: str
    region_id: str
    method: str
    zone_id: int | str
    zone_name: str
    modality: str
    path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HabitatFeatureUnitResult:
    unit_key: str
    status: str
    output_dir: Path
    completion_path: Path | None
    rows: tuple[Mapping[str, Any], ...]
    diagnostic_names: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class HabitatFeatureRunResult:
    output_root: Path
    manifest_path: Path
    manifest_sha256: str
    habitat_index_path: Path
    habitat_index_sha256: str
    protocol_id: str
    protocol_version: str
    protocol_sha256: str
    variant: str
    preset_id: str
    preset_path: Path
    preset_sha256: str
    jobs: int
    dry_run: bool
    started_utc: str
    finished_utc: str
    units: tuple[HabitatFeatureUnitResult, ...]
    checks: tuple[HabitatFeatureCheck, ...]
    failures: tuple[HabitatFeatureFailure, ...]
    software: Mapping[str, str]
    reference_deviations: tuple[str, ...] = ()
    log_lines: tuple[str, ...] = ()
    backups: tuple[Path, ...] = ()
    table_path: Path | None = None
    method_table_paths: Mapping[str, Path] | None = None
    reference_table_paths: Mapping[str, Path] | None = None
    checks_path: Path | None = None
    errors_path: Path | None = None
    metadata_path: Path | None = None
    log_path: Path | None = None

    @property
    def completed(self) -> int:
        return sum(unit.status == "complete" for unit in self.units)

    @property
    def resumed(self) -> int:
        return sum(unit.status == "resumed" for unit in self.units)

    @property
    def blocked(self) -> int:
        return sum(unit.status == "blocked" for unit in self.units)

    @property
    def zone_rows(self) -> int:
        return sum(len(unit.rows) for unit in self.units)

    @property
    def successful_zones(self) -> int:
        return sum(
            str(row.get("zone_status", "complete")) == "complete"
            and not str(row.get("error", ""))
            for unit in self.units
            for row in unit.rows
        )

    @property
    def skipped_zones(self) -> int:
        return sum(
            str(row.get("zone_status", "")) == "skipped"
            for unit in self.units
            for row in unit.rows
        )

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def warnings(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    @property
    def reference_compatible(self) -> bool:
        return not self.reference_deviations

    @property
    def valid(self) -> bool:
        return not self.failures
