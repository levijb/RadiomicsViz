"""Safely materialize a combined lesion mask into per-lesion binary masks.

Two explicit scientific interpretations are supported:

``labeled_map``
    Every positive integer in the source map is one lesion. This preserves the
    behavior of the historical ``materialize_labeled_lesions.py`` workflow.

``connected_components``
    The configured foreground value in a binary mask is split with the configured
    6- or 26-neighbor definition. Touching lesions are one component; this command
    deliberately does not introduce watershed or other inferred boundaries.

Source data are read-only. A complete tree is written and read back in a private
staging directory before it replaces any canonical materialization. Existing output
requires a matching completion record or an explicit ``force`` request.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from radiomicviz.config._common import SAFE_ID_PATTERN
from radiomicviz.config.project import ProjectConfig

COMPLETION_SCHEMA_VERSION = 1


class MaterializationError(ValueError):
    """A combined-mask input cannot be safely materialized or reused."""


class LockUnavailableError(MaterializationError):
    """Another process holds a materialization lock for the same session."""


@dataclass(frozen=True)
class RegionPlan:
    """One binary output that will be derived from a combined source mask."""

    region_id: str
    source_label: int | None
    component_id: int | None
    n_voxels: int


@dataclass(frozen=True)
class MaterializationUnit:
    """One subject/session combined mask and its planned outputs."""

    subject_id: str
    subject_name: str
    session_id: str
    session_name: str
    source_path: Path
    source_sha256: str
    output_dir: Path
    regions: tuple[RegionPlan, ...]
    status: str
    status_reason: str


@dataclass(frozen=True)
class MaterializationPlan:
    """Validated work plan for every discovered subject/session."""

    project: ProjectConfig
    data_root: Path
    output_root: Path
    units: tuple[MaterializationUnit, ...]


@dataclass(frozen=True)
class UnitResult:
    """Result for one materialization unit."""

    subject_id: str
    session_id: str
    output_dir: Path
    status: str
    n_regions: int
    message: str = ""


@dataclass(frozen=True)
class MaterializationResult:
    """Cohort-level materialization result."""

    plan: MaterializationPlan
    units: tuple[UnitResult, ...]
    dry_run: bool

    @property
    def completed(self) -> int:
        return sum(unit.status == "completed" for unit in self.units)

    @property
    def resumed(self) -> int:
        return sum(unit.status == "resumed" for unit in self.units)

    @property
    def blocked(self) -> int:
        return sum(unit.status == "blocked" for unit in self.units)

    @property
    def failed(self) -> int:
        return sum(unit.status == "failed" for unit in self.units)

    @property
    def valid(self) -> bool:
        return self.blocked == 0 and self.failed == 0


@dataclass(frozen=True)
class _Session:
    subject_id: str
    subject_name: str
    session_id: str
    session_name: str


def build_materialization_plan(
    project: ProjectConfig,
    *,
    output_root: str | Path | None = None,
) -> MaterializationPlan:
    """Discover and validate combined-mask inputs without writing files."""
    if project.source is None:
        raise MaterializationError(
            "lesion materialization requires a project configuration loaded from a file"
        )
    if project.regions.mode == "per_region_masks":
        raise MaterializationError(
            "regions.mode is 'per_region_masks'; lesion materialization is unnecessary "
            "because the configured masks are already separated"
        )

    base = project.source.resolve().parent
    data_root = _resolve_root(project.data_root, base)
    resolved_output = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else _resolve_root(project.output_root, base)
    )
    if not data_root.is_dir():
        raise MaterializationError(f"data_root does not exist or is not a directory: {data_root}")

    sessions = _discover_sessions(project, data_root)
    if not sessions:
        raise MaterializationError("no valid subject/session inputs were discovered")

    units: list[MaterializationUnit] = []
    for session in sessions:
        matches = _match_source(project, data_root, session)
        if len(matches) != 1:
            rendered = project.regions.mask_pattern.format(
                subject=session.subject_name, session=session.session_name
            )
            if not matches:
                raise MaterializationError(
                    f"{session.subject_id}/{session.session_id}: combined mask pattern "
                    f"{rendered!r} matched no files"
                )
            raise MaterializationError(
                f"{session.subject_id}/{session.session_id}: combined mask pattern "
                f"{rendered!r} matched {len(matches)} files; exactly one is required"
            )

        source_path = matches[0]
        source_hash = _sha256_file(source_path)
        _image, array = _load_combined_mask(source_path)
        regions = _derive_region_plan(project, array)
        output_dir = materialized_session_root(
            resolved_output,
            session.subject_id,
            session.session_id,
            project.regions.region_type,
        )
        valid, reason, _marker = _validate_complete_tree(
            output_dir,
            project=project,
            source_path=source_path,
            source_sha256=source_hash,
            verify_hashes=True,
        )
        if valid:
            status, status_reason = "resume", "matching verified materialization exists"
        elif output_dir.exists():
            status, status_reason = "blocked", reason
        else:
            status, status_reason = "run", "no existing materialization"
        units.append(
            MaterializationUnit(
                subject_id=session.subject_id,
                subject_name=session.subject_name,
                session_id=session.session_id,
                session_name=session.session_name,
                source_path=source_path,
                source_sha256=source_hash,
                output_dir=output_dir,
                regions=regions,
                status=status,
                status_reason=status_reason,
            )
        )

    return MaterializationPlan(
        project=project,
        data_root=data_root,
        output_root=resolved_output,
        units=tuple(units),
    )


def run_materialization(
    plan: MaterializationPlan,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> MaterializationResult:
    """Execute or preview a validated materialization plan."""
    results: list[UnitResult] = []
    for unit in plan.units:
        if unit.status == "resume" and not force:
            results.append(_unit_result(unit, "resumed", unit.status_reason))
            continue
        if unit.status == "blocked" and not force:
            results.append(
                _unit_result(
                    unit,
                    "blocked",
                    f"{unit.status_reason}; rerun with --force to archive and replace it",
                )
            )
            continue
        if dry_run:
            action = "would replace existing output" if force and unit.output_dir.exists() else "would run"
            results.append(_unit_result(unit, action, unit.status_reason))
            continue
        try:
            status = _materialize_unit(plan.project, unit, force=force)
            results.append(_unit_result(unit, status, ""))
        except (MaterializationError, OSError, RuntimeError, ValueError) as exc:
            results.append(_unit_result(unit, "failed", f"{type(exc).__name__}: {exc}"))
    return MaterializationResult(plan=plan, units=tuple(results), dry_run=dry_run)


def materialized_session_root(
    output_root: Path,
    subject_id: str,
    session_id: str,
    region_type: str,
) -> Path:
    """Return the canonical materialized-mask tree for one session."""
    for kind, value in (
        ("subject", subject_id),
        ("session", session_id),
        ("region type", region_type),
    ):
        if value in {".", ".."} or SAFE_ID_PATTERN.fullmatch(value) is None:
            raise MaterializationError(f"unsafe {kind} identifier: {value!r}")
    return (
        output_root
        / "materialized_regions"
        / subject_id
        / session_id
        / region_type
    ).resolve()


def materialized_masks_for_session(
    project: ProjectConfig,
    *,
    data_root: Path,
    output_root: Path,
    subject_id: str,
    subject_name: str,
    session_id: str,
    session_name: str,
) -> list[Path]:
    """Return verified materialized masks for a combined-mask manifest session.

    The source combined-mask hash and every output-mask hash are checked, preventing a
    stale materialization from entering a new manifest.
    """
    session = _Session(subject_id, subject_name, session_id, session_name)
    matches = _match_source(project, data_root, session)
    if len(matches) != 1:
        raise MaterializationError(
            f"combined source mask matched {len(matches)} files; exactly one is required"
        )
    source_path = matches[0]
    output_dir = materialized_session_root(
        output_root, subject_id, session_id, project.regions.region_type
    )
    valid, reason, marker = _validate_complete_tree(
        output_dir,
        project=project,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        verify_hashes=True,
    )
    if not valid or marker is None:
        raise MaterializationError(reason)
    return [
        _safe_relative_file(output_dir, record["mask_relative_path"])
        for record in marker["regions"]
    ]


def _unit_result(unit: MaterializationUnit, status: str, message: str) -> UnitResult:
    return UnitResult(
        subject_id=unit.subject_id,
        session_id=unit.session_id,
        output_dir=unit.output_dir,
        status=status,
        n_regions=len(unit.regions),
        message=message,
    )


def _materialize_unit(
    project: ProjectConfig,
    unit: MaterializationUnit,
    *,
    force: bool,
) -> str:
    parent = unit.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{unit.output_dir.name}.materialize.lock"
    with _exclusive_file_lock(lock_path):
        current_hash = _sha256_file(unit.source_path)
        if current_hash != unit.source_sha256:
            raise MaterializationError("source mask changed after the plan was built; preview again")

        valid, _reason, _marker = _validate_complete_tree(
            unit.output_dir,
            project=project,
            source_path=unit.source_path,
            source_sha256=current_hash,
            verify_hashes=True,
        )
        if valid and not force:
            return "resumed"
        if unit.output_dir.exists() and not force:
            raise MaterializationError(
                "existing materialization is incomplete or does not match; use --force"
            )

        transaction = uuid.uuid4().hex[:12]
        stage = parent / f".{unit.output_dir.name}.tmp-{transaction}"
        if stage.exists():
            raise MaterializationError(f"staging directory already exists: {stage}")
        stage.mkdir()
        try:
            image, array = _load_combined_mask(unit.source_path)
            regions = _derive_regions(project, array)
            if tuple(region[0] for region in regions) != tuple(
                region.region_id for region in unit.regions
            ):
                raise MaterializationError("derived region identifiers changed after planning")
            records = _write_staging_tree(
                stage,
                project,
                unit,
                image,
                regions,
                transaction,
            )
            if _sha256_file(unit.source_path) != current_hash:
                raise MaterializationError("source mask changed during materialization")
            marker = _completion_payload(project, unit, records, transaction)
            _write_json(stage / ".materialization" / "complete.json", marker)
            stage_valid, stage_reason, _ = _validate_complete_tree(
                stage,
                project=project,
                source_path=unit.source_path,
                source_sha256=current_hash,
                verify_hashes=True,
            )
            if not stage_valid:
                raise MaterializationError(f"staging verification failed: {stage_reason}")
            _commit_stage(stage, unit.output_dir, force=force)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return "completed"


def _write_staging_tree(
    stage: Path,
    project: ProjectConfig,
    unit: MaterializationUnit,
    source_image: sitk.Image,
    regions: list[tuple[str, np.ndarray, int | None, int | None]],
    transaction: str,
) -> list[dict]:
    records: list[dict] = []
    inventory_rows: list[dict] = []
    voxel_volume = float(np.prod(source_image.GetSpacing()))
    structure6 = ndi.generate_binary_structure(3, 1)
    structure26 = ndi.generate_binary_structure(3, 3)
    for region_id, component, source_label, component_id in regions:
        n_voxels = int(np.count_nonzero(component))
        if n_voxels <= 0:
            raise MaterializationError(f"{region_id}: derived mask is empty")
        region_dir = stage / region_id
        region_dir.mkdir()
        mask_path = region_dir / "mask.nii.gz"
        mask_image = sitk.GetImageFromArray(component.astype(np.uint8))
        mask_image.CopyInformation(source_image)
        sitk.WriteImage(mask_image, str(mask_path), True)

        readback_image = sitk.ReadImage(str(mask_path))
        readback = sitk.GetArrayFromImage(readback_image)
        if set(int(value) for value in np.unique(readback)) - {0, 1}:
            raise MaterializationError(f"{region_id}: saved mask is not binary")
        if int(np.count_nonzero(readback)) != n_voxels:
            raise MaterializationError(f"{region_id}: saved voxel count differs from source")
        if not _same_geometry(source_image, readback_image):
            raise MaterializationError(f"{region_id}: saved mask geometry changed")

        components6 = int(ndi.label(readback != 0, structure=structure6)[1])
        components26 = int(ndi.label(readback != 0, structure=structure26)[1])
        coordinates = np.argwhere(component)
        centroid = [float(value) for value in coordinates.mean(axis=0)]
        metadata = {
            "subject_id": unit.subject_id,
            "session_id": unit.session_id,
            "region_id": region_id,
            "region_type": project.regions.region_type,
            "source_mask": str(unit.source_path),
            "source_label": source_label,
            "component_id": component_id,
            "n_voxels": n_voxels,
            "volume_mm3": round(n_voxels * voxel_volume, 6),
            "centroid_index_zyx": centroid,
            "n_connected_components_6": components6,
            "n_connected_components_26": components26,
        }
        _write_json(region_dir / "metadata.json", metadata)
        relative = f"{region_id}/mask.nii.gz"
        record = {
            "region_id": region_id,
            "source_label": source_label,
            "component_id": component_id,
            "n_voxels": n_voxels,
            "mask_relative_path": relative,
            "mask_sha256": _sha256_file(mask_path),
        }
        records.append(record)
        inventory_rows.append({**metadata, **record})

    materialization_dir = stage / ".materialization"
    materialization_dir.mkdir()
    _write_inventory(materialization_dir / "inventory.csv", inventory_rows)
    _write_json(
        materialization_dir / "provenance.json",
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "transaction_id": transaction,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_mask": str(unit.source_path),
            "source_mask_sha256": unit.source_sha256,
            "mode": project.regions.mode,
            "foreground_label": project.regions.label,
            "connectivity": project.regions.connectivity,
            "n_regions": len(records),
        },
    )
    return records


def _completion_payload(
    project: ProjectConfig,
    unit: MaterializationUnit,
    records: list[dict],
    transaction: str,
) -> dict:
    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "transaction_id": transaction,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": unit.subject_id,
        "session_id": unit.session_id,
        "region_type": project.regions.region_type,
        "mode": project.regions.mode,
        "foreground_label": project.regions.label,
        "connectivity": project.regions.connectivity,
        "source_mask": str(unit.source_path),
        "source_mask_sha256": unit.source_sha256,
        "regions": records,
    }


def _commit_stage(stage: Path, final: Path, *, force: bool) -> None:
    backup: Path | None = None
    if final.exists():
        if not force:
            raise MaterializationError(f"output already exists: {final}")
        backup = _unique_backup_path(final)
        os.replace(final, backup)
    try:
        os.replace(stage, final)
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            os.replace(backup, final)
        raise


def _validate_complete_tree(
    tree: Path,
    *,
    project: ProjectConfig,
    source_path: Path,
    source_sha256: str,
    verify_hashes: bool,
) -> tuple[bool, str, dict | None]:
    marker_path = tree / ".materialization" / "complete.json"
    if not marker_path.is_file():
        return False, "materialization completion record is missing", None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"materialization completion record is unreadable: {exc}", None
    expected = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "region_type": project.regions.region_type,
        "mode": project.regions.mode,
        "foreground_label": project.regions.label,
        "connectivity": project.regions.connectivity,
        "source_mask": str(source_path),
        "source_mask_sha256": source_sha256,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            return False, f"completion record {key} does not match the current project/input", marker
    records = marker.get("regions")
    if not isinstance(records, list):
        return False, "completion record regions must be a list", marker
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return False, "completion record contains a non-mapping region", marker
        region_id = record.get("region_id")
        relative = record.get("mask_relative_path")
        if not isinstance(region_id, str) or not _safe_identifier(region_id):
            return False, "completion record contains an unsafe region_id", marker
        if region_id in seen:
            return False, f"duplicate region_id in completion record: {region_id}", marker
        seen.add(region_id)
        try:
            mask_path = _safe_relative_file(tree, relative)
        except MaterializationError as exc:
            return False, str(exc), marker
        if mask_path.parent.name != region_id or mask_path.name != "mask.nii.gz":
            return False, f"unexpected materialized mask path for {region_id}", marker
        if not mask_path.is_file():
            return False, f"materialized mask is missing: {relative}", marker
        expected_hash = record.get("mask_sha256")
        if verify_hashes and (
            not isinstance(expected_hash, str) or _sha256_file(mask_path) != expected_hash
        ):
            return False, f"materialized mask checksum mismatch: {relative}", marker
    return True, "ok", marker


def _derive_region_plan(project: ProjectConfig, array: np.ndarray) -> tuple[RegionPlan, ...]:
    return tuple(
        RegionPlan(
            region_id=region_id,
            source_label=source_label,
            component_id=component_id,
            n_voxels=int(np.count_nonzero(component)),
        )
        for region_id, component, source_label, component_id in _derive_regions(project, array)
    )


def _derive_regions(
    project: ProjectConfig,
    array: np.ndarray,
) -> list[tuple[str, np.ndarray, int | None, int | None]]:
    prefix = project.regions.region_type
    if project.regions.mode == "labeled_map":
        labels = sorted(int(value) for value in np.unique(array) if int(value) != 0)
        return [
            (f"{prefix}_{label:03d}", array == label, label, None)
            for label in labels
        ]
    if project.regions.mode == "connected_components":
        unique = np.unique(array)
        allowed = np.array([0, project.regions.label], dtype=np.int64)
        invalid = unique[~np.isin(unique, allowed)]
        if invalid.size:
            raise MaterializationError(
                "connected-components source contains values other than background 0 "
                f"and foreground {project.regions.label}: {_values_text(unique)}"
            )
        rank = 1 if project.regions.connectivity == 6 else 3
        labeled, count = ndi.label(
            array == project.regions.label,
            structure=ndi.generate_binary_structure(3, rank),
        )
        return [
            (f"{prefix}_{component:03d}", labeled == component, None, component)
            for component in range(1, int(count) + 1)
        ]
    raise MaterializationError(f"unsupported materialization mode: {project.regions.mode}")


def _load_combined_mask(path: Path) -> tuple[sitk.Image, np.ndarray]:
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        raise MaterializationError(f"cannot read combined mask {path}: {exc}") from exc
    if image.GetDimension() != 3:
        raise MaterializationError(
            f"combined mask must be 3D, found {image.GetDimension()}D: {path}"
        )
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all():
        raise MaterializationError(f"combined mask contains non-finite values: {path}")
    if not np.equal(array, np.round(array)).all():
        raise MaterializationError(f"combined mask contains fractional values: {path}")
    if (array < 0).any():
        raise MaterializationError(f"combined mask contains negative values: {path}")
    if array.size and float(np.max(array)) > np.iinfo(np.int64).max:
        raise MaterializationError(f"combined mask label exceeds int64 range: {path}")
    return image, array.astype(np.int64)


def _discover_sessions(project: ProjectConfig, data_root: Path) -> list[_Session]:
    subject_paths = _safe_glob(data_root, project.subjects.discovery_pattern)
    if not subject_paths:
        raise MaterializationError(
            f"no subjects matched {project.subjects.discovery_pattern!r} under {data_root}"
        )
    sessions: list[_Session] = []
    seen_subjects: set[str] = set()
    for subject_path in subject_paths:
        if not subject_path.is_dir():
            raise MaterializationError(f"subject discovery matched a non-directory: {subject_path}")
        subject_id = _extract_id(project.subjects.id_regex, "subject", subject_path.name)
        if subject_id is None or not _safe_identifier(subject_id):
            raise MaterializationError(f"invalid subject identifier from {subject_path.name!r}")
        if subject_id in seen_subjects:
            raise MaterializationError(f"duplicate subject identifier: {subject_id}")
        seen_subjects.add(subject_id)
        if project.sessions.mode == "single":
            sessions.append(
                _Session(
                    subject_id,
                    subject_path.name,
                    project.sessions.default_session_id,
                    project.sessions.default_session_id,
                )
            )
            continue

        session_paths = _safe_glob(subject_path, project.sessions.discovery_pattern)
        if not session_paths:
            raise MaterializationError(f"{subject_id}: no sessions were discovered")
        seen_sessions: set[str] = set()
        for session_path in session_paths:
            if not session_path.is_dir():
                raise MaterializationError(
                    f"session discovery matched a non-directory: {session_path}"
                )
            session_id = _extract_id(
                project.sessions.id_regex, "session", session_path.name
            )
            if session_id is None or not _safe_identifier(session_id):
                raise MaterializationError(
                    f"invalid session identifier from {session_path.name!r}"
                )
            if session_id in seen_sessions:
                raise MaterializationError(
                    f"{subject_id}: duplicate session identifier {session_id}"
                )
            seen_sessions.add(session_id)
            sessions.append(
                _Session(subject_id, subject_path.name, session_id, session_path.name)
            )
    return sessions


def _match_source(
    project: ProjectConfig,
    data_root: Path,
    session: _Session,
) -> list[Path]:
    pattern = project.regions.mask_pattern.format(
        subject=session.subject_name, session=session.session_name
    )
    return _safe_glob(data_root, pattern)


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    if not _is_safe_relative_pattern(pattern):
        raise MaterializationError(
            f"input pattern must be relative and must not contain '..': {pattern}"
        )
    root = root.resolve()
    paths: list[Path] = []
    for path in root.glob(pattern):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise MaterializationError(
                f"input pattern resolved outside its root: {resolved}"
            ) from exc
        paths.append(resolved)
    return sorted(paths, key=lambda path: str(path).casefold())


def _is_safe_relative_pattern(pattern: str) -> bool:
    posix = PurePosixPath(pattern.replace("\\", "/"))
    windows = PureWindowsPath(pattern)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
    )


def _safe_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not _is_safe_relative_pattern(relative):
        raise MaterializationError("completion record contains an unsafe mask path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterializationError("completion record mask path escapes its output tree") from exc
    return candidate


def _extract_id(pattern: str, group: str, name: str) -> str | None:
    match = re.fullmatch(pattern, name)
    return match.group(group) if match is not None else None


def _safe_identifier(value: str) -> bool:
    return value not in {".", ".."} and SAFE_ID_PATTERN.fullmatch(value) is not None


def _resolve_root(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetDimension() == right.GetDimension()
        and left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-7, rtol=0)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-7, rtol=0)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-7, rtol=0)
    )


def _values_text(values: np.ndarray, limit: int = 20) -> str:
    rendered = [str(int(value)) for value in values[:limit]]
    if values.size > limit:
        rendered.append("...")
    return ", ".join(rendered)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_inventory(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "subject_id",
        "session_id",
        "region_id",
        "region_type",
        "source_label",
        "component_id",
        "n_voxels",
        "volume_mm3",
        "centroid_index_zyx",
        "n_connected_components_6",
        "n_connected_components_26",
        "mask_relative_path",
        "mask_sha256",
        "source_mask",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["centroid_index_zyx"] = json.dumps(
                serialized["centroid_index_zyx"], separators=(",", ":")
            )
            writer.writerow(serialized)


def _unique_backup_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
        suffix += 1
    return candidate


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a nonblocking advisory lock released automatically on exit."""
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LockUnavailableError(f"lock already held: {lock_path}") from exc
            acquired = True
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise LockUnavailableError(f"lock already held: {lock_path}") from exc
            acquired = True
        yield
    finally:
        try:
            if acquired and os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
