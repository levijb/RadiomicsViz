"""Data contracts for manifest discovery and its concise quality-control report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    """One region and modality in the canonical manifest."""

    subject_id: str
    session_id: str
    region_id: str
    region_type: str
    modality: str
    image_path: str
    mask_path: str
    label: int
    space_id: str
    image_size_bytes: int
    image_mtime_utc: str
    mask_size_bytes: int
    mask_mtime_utc: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckRow:
    """One manifest input check."""

    check_id: str
    status: str
    subject_id: str = ""
    session_id: str = ""
    region_id: str = ""
    modality: str = ""
    path: str = ""
    measured_value: str = ""
    threshold: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorRow:
    """One blocking problem found while constructing the manifest."""

    error_code: str
    message: str
    subject_id: str = ""
    session_id: str = ""
    region_id: str = ""
    modality: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestPreview:
    """In-memory manifest and QC result, produced before anything is written."""

    project_path: Path
    data_root: Path
    output_root: Path
    rows: tuple[ManifestRow, ...]
    checks: tuple[CheckRow, ...]
    errors: tuple[ErrorRow, ...]
    reference_status: str
    subjects_found: int
    sessions_found: int
    regions_found: int

    @property
    def valid(self) -> bool:
        """Whether the canonical manifest may be written."""
        return not self.errors and bool(self.rows)

    @property
    def warning_count(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    @property
    def failure_count(self) -> int:
        return len(self.errors)

    @property
    def modalities_found(self) -> tuple[str, ...]:
        return tuple(sorted({row.modality for row in self.rows}))
