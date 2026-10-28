"""Deterministic discovery and input validation for the regions manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

import numpy as np
import SimpleITK as sitk

from radiomicviz.config._common import SAFE_ID_PATTERN
from radiomicviz.config.project import ProjectConfig
from radiomicviz.manifest.models import CheckRow, ErrorRow, ManifestPreview, ManifestRow

REFERENCE_SPACING_MM = (1.0, 1.0, 1.0)
SPACING_TOLERANCE_MM = 0.001
GEOMETRY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class _ImageInfo:
    path: Path
    dimension: int
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


@dataclass(frozen=True)
class _Session:
    subject_id: str
    subject_name: str
    session_id: str
    session_name: str


class _Collector:
    def __init__(self) -> None:
        self.checks: list[CheckRow] = []
        self.errors: list[ErrorRow] = []

    def check(self, check_id: str, status: str, **values: str) -> None:
        self.checks.append(CheckRow(check_id=check_id, status=status, **values))

    def fail(
        self,
        error_code: str,
        message: str,
        *,
        subject_id: str = "",
        session_id: str = "",
        region_id: str = "",
        modality: str = "",
        path: str = "",
        measured_value: str = "",
        threshold: str = "",
    ) -> None:
        scope = {
            "subject_id": subject_id,
            "session_id": session_id,
            "region_id": region_id,
            "modality": modality,
            "path": path,
        }
        self.errors.append(ErrorRow(error_code=error_code, message=message, **scope))
        self.check(
            error_code,
            "FAIL",
            measured_value=measured_value,
            threshold=threshold,
            message=message,
            **scope,
        )


def build_manifest_preview(
    project: ProjectConfig,
    *,
    output_root: str | Path | None = None,
) -> ManifestPreview:
    """Discover inputs and validate them without writing any files."""
    if project.source is None:
        raise ValueError("manifest discovery requires a project configuration loaded from a file")

    base = project.source.resolve().parent
    data_root = _resolve_root(project.data_root, base)
    resolved_output = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else _resolve_root(project.output_root, base)
    )
    collect = _Collector()
    rows: list[ManifestRow] = []
    subjects_found: set[str] = set()
    sessions_found: set[tuple[str, str]] = set()
    regions_found: set[tuple[str, str, str]] = set()
    reference_status = "reference-compatible"

    if not data_root.is_dir():
        collect.fail(
            "data_root_missing",
            "data_root does not exist or is not a directory",
            path=str(data_root),
        )
        return _preview(
            project, data_root, resolved_output, rows, collect, reference_status,
            subjects_found, sessions_found, regions_found,
        )

    subject_paths = _glob(data_root, project.subjects.discovery_pattern, collect, "subject_discovery")
    if not subject_paths:
        collect.fail(
            "subjects_not_found",
            f"no subjects matched {project.subjects.discovery_pattern!r}",
            path=str(data_root),
        )

    subject_ids: dict[str, Path] = {}
    sessions: list[_Session] = []
    for subject_path in subject_paths:
        if not subject_path.is_dir():
            collect.fail(
                "subject_not_directory",
                "subject discovery matched a non-directory",
                path=str(subject_path),
            )
            continue
        subject_id = _extract_id(project.subjects.id_regex, "subject", subject_path.name)
        if subject_id is None:
            collect.fail(
                "subject_id_no_match",
                "subject folder name does not fully match subjects.id_regex",
                path=str(subject_path),
                measured_value=subject_path.name,
            )
            continue
        if not _safe_identifier(subject_id):
            collect.fail(
                "subject_id_unsafe",
                "subjects.id_regex produced an unsafe identifier",
                path=str(subject_path),
                measured_value=subject_id,
                threshold="letters, digits, '_', '.', '-' except '.' and '..'",
            )
            continue
        if subject_id in subject_ids:
            collect.fail(
                "duplicate_subject_id",
                f"subject_id {subject_id!r} was also produced by {subject_ids[subject_id]}",
                subject_id=subject_id,
                path=str(subject_path),
            )
            continue
        subject_ids[subject_id] = subject_path
        subjects_found.add(subject_id)
        collect.check(
            "subject_discovered", "PASS", subject_id=subject_id, path=str(subject_path),
            message="subject folder discovered",
        )
        sessions.extend(_discover_sessions(project, subject_path, subject_id, collect))

    image_cache: dict[Path, _ImageInfo | None] = {}
    mask_cache: dict[Path, tuple[_ImageInfo | None, np.ndarray | None]] = {}
    seen_image_paths: dict[tuple[str, str, Path], str] = {}

    for session in sessions:
        sessions_found.add((session.subject_id, session.session_id))
        images: dict[str, tuple[Path, _ImageInfo]] = {}

        for modality in project.modalities:
            scope = {
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "modality": modality.name,
            }
            image_matches = _match_pattern(
                data_root, modality.image_pattern, session, collect, "image", **scope
            )
            if len(image_matches) != 1:
                _wrong_match_count(collect, "image", modality.image_pattern, image_matches, scope)
                continue
            image_path = image_matches[0]
            duplicate_key = (session.subject_id, session.session_id, image_path)
            previous_modality = seen_image_paths.get(duplicate_key)
            if previous_modality is not None:
                collect.fail(
                    "duplicate_image",
                    f"the same image is assigned to {previous_modality} and {modality.name}",
                    path=str(image_path), **scope,
                )
                continue
            seen_image_paths[duplicate_key] = modality.name

            image_info = _read_image_info(image_path, image_cache, collect, "image", **scope)
            if image_info is None:
                continue
            if image_info.dimension != 3:
                collect.fail(
                    "image_dimension", f"expected a 3D image, found {image_info.dimension}D",
                    path=str(image_path), measured_value=str(image_info.dimension), threshold="3", **scope,
                )
                continue
            collect.check(
                "image_dimension", "PASS", path=str(image_path), measured_value="3",
                threshold="3", message="analysis-ready image is 3D", **scope,
            )

            spacing_ok = _spacing_is_reference(image_info.spacing)
            if not spacing_ok:
                status = "WARN" if project.allow_nonreference_spacing else "FAIL"
                message = (
                    "spacing is outside 1.0 +/- 0.001 mm; run is stamped non-reference"
                    if project.allow_nonreference_spacing
                    else "spacing is outside 1.0 +/- 0.001 mm"
                )
                collect.check(
                    "reference_spacing", status, path=str(image_path),
                    measured_value=_tuple_text(image_info.spacing),
                    threshold="1.0 +/- 0.001 mm per axis", message=message, **scope,
                )
                if project.allow_nonreference_spacing:
                    reference_status = "non-reference-spacing"
                else:
                    collect.errors.append(
                        ErrorRow(error_code="reference_spacing", message=message, path=str(image_path), **scope)
                    )
            else:
                collect.check(
                    "reference_spacing", "PASS", path=str(image_path),
                    measured_value=_tuple_text(image_info.spacing),
                    threshold="1.0 +/- 0.001 mm per axis",
                    message="spacing is reference-compatible", **scope,
                )

            images[modality.name] = (image_path, image_info)

        mask_matches = _match_pattern(
            data_root, project.regions.mask_pattern, session, collect, "region_mask",
            subject_id=session.subject_id, session_id=session.session_id,
        )
        if not mask_matches:
            collect.fail(
                "region_masks_not_found", f"no region masks matched {project.regions.mask_pattern!r}",
                subject_id=session.subject_id, session_id=session.session_id, path=str(data_root),
            )
            continue

        seen_regions: dict[str, Path] = {}
        occupancy: np.ndarray | None = None
        for mask_path in mask_matches:
            region_id = _extract_id(project.regions.id_regex, "region", mask_path.name)
            region_scope = {"subject_id": session.subject_id, "session_id": session.session_id}
            if region_id is None:
                collect.fail(
                    "region_id_no_match", "mask file name does not fully match regions.id_regex",
                    path=str(mask_path), measured_value=mask_path.name, **region_scope,
                )
                continue
            if not _safe_identifier(region_id):
                collect.fail(
                    "region_id_unsafe", "regions.id_regex produced an unsafe identifier",
                    path=str(mask_path), measured_value=region_id,
                    threshold="letters, digits, '_', '.', '-' except '.' and '..'", **region_scope,
                )
                continue
            region_scope["region_id"] = region_id
            if region_id in seen_regions:
                collect.fail(
                    "duplicate_region_id",
                    f"region_id {region_id!r} was also produced by {seen_regions[region_id]}",
                    path=str(mask_path), **region_scope,
                )
                continue
            seen_regions[region_id] = mask_path

            mask_info, mask_array = _read_mask(
                mask_path, mask_cache, collect, "region_mask", **region_scope
            )
            if mask_info is None or mask_array is None:
                continue
            if mask_info.dimension != 3:
                collect.fail(
                    "mask_dimension", f"expected a 3D mask, found {mask_info.dimension}D",
                    path=str(mask_path), measured_value=str(mask_info.dimension), threshold="3",
                    **region_scope,
                )
                continue

            unique_values = np.unique(mask_array)
            allowed = np.array([0, project.regions.label])
            invalid_values = unique_values[~np.isin(unique_values, allowed)]
            if invalid_values.size:
                collect.fail(
                    "mask_values", "mask contains values other than background 0 and the configured label",
                    path=str(mask_path), measured_value=_values_text(unique_values),
                    threshold=f"0 and {project.regions.label}", **region_scope,
                )
                continue
            foreground = mask_array == project.regions.label
            voxel_count = int(np.count_nonzero(foreground))
            if voxel_count == 0:
                collect.fail(
                    "mask_empty", "mask has no voxels equal to the configured label",
                    path=str(mask_path), measured_value="0", threshold="> 0", **region_scope,
                )
                continue
            collect.check(
                "mask_values", "PASS", path=str(mask_path),
                measured_value=_values_text(unique_values), threshold=f"0 and {project.regions.label}",
                message="mask values are valid", **region_scope,
            )
            size_status = "WARN" if voxel_count < 50 else "PASS"
            size_message = (
                "region has fewer than 10 voxels; voxelwise behavior requires later testing"
                if voxel_count < 10
                else "region has fewer than 50 voxels; habitat splitting will fall back"
                if voxel_count < 50
                else "region size passes the reference habitat split threshold"
            )
            collect.check(
                "region_voxel_count", size_status, path=str(mask_path),
                measured_value=str(voxel_count), threshold="10 and 50 voxel downstream thresholds",
                message=size_message, **region_scope,
            )

            components_6 = _component_count(foreground, fully_connected=False)
            components_26 = _component_count(foreground, fully_connected=True)
            collect.check(
                "mask_components", "PASS" if components_26 == 1 else "WARN", path=str(mask_path),
                measured_value=f"6-connected={components_6}; 26-connected={components_26}",
                threshold="recorded; one 26-connected component expected for a lesion",
                message=("component counts recorded" if components_26 == 1 else
                         "one region mask contains multiple disconnected components"),
                **region_scope,
            )

            if occupancy is None:
                occupancy = np.zeros(foreground.shape, dtype=bool)
            overlap = int(np.count_nonzero(occupancy & foreground)) if occupancy.shape == foreground.shape else 0
            collect.check(
                "region_overlap", "WARN" if overlap else "PASS", path=str(mask_path),
                measured_value=str(overlap), threshold="0 voxels",
                message=("region overlaps a previously discovered region" if overlap else
                         "no overlap with previously discovered regions"),
                **region_scope,
            )
            if occupancy.shape == foreground.shape:
                occupancy |= foreground

            region_rows = 0
            for modality_name, (image_path, image_info) in images.items():
                modality_scope = {**region_scope, "modality": modality_name}
                if not _geometry_matches(image_info, mask_info):
                    collect.fail(
                        "mask_image_geometry", "region mask geometry does not match the image",
                        path=str(mask_path), measured_value=_geometry_text(mask_info),
                        threshold=_geometry_text(image_info), **modality_scope,
                    )
                    continue
                collect.check(
                    "mask_image_geometry", "PASS", path=str(mask_path),
                    measured_value=_geometry_text(mask_info), threshold=_geometry_text(image_info),
                    message="mask and source image geometry match", **modality_scope,
                )
                image_stat = _fingerprint(image_path)
                mask_stat = _fingerprint(mask_path)
                rows.append(
                    ManifestRow(
                        subject_id=session.subject_id, session_id=session.session_id,
                        region_id=region_id, region_type=project.regions.region_type,
                        modality=modality_name, image_path=str(image_path),
                        mask_path=str(mask_path), label=project.regions.label,
                        space_id=project.space_id,
                        image_size_bytes=image_stat[0], image_mtime_utc=image_stat[1],
                        mask_size_bytes=mask_stat[0], mask_mtime_utc=mask_stat[1],
                    )
                )
                region_rows += 1
            if region_rows:
                regions_found.add((session.subject_id, session.session_id, region_id))

    rows.sort(key=lambda row: (row.subject_id, row.session_id, row.region_id, row.modality))
    if not rows and not collect.errors:
        collect.fail("manifest_empty", "discovery produced no manifest rows")
    return _preview(
        project, data_root, resolved_output, rows, collect, reference_status,
        subjects_found, sessions_found, regions_found,
    )


def _preview(
    project: ProjectConfig, data_root: Path, output_root: Path, rows: list[ManifestRow],
    collect: _Collector, reference_status: str, subjects: set[str],
    sessions: set[tuple[str, str]], regions: set[tuple[str, str, str]],
) -> ManifestPreview:
    return ManifestPreview(
        project_path=project.source.resolve(), data_root=data_root, output_root=output_root,
        rows=tuple(rows), checks=tuple(collect.checks), errors=tuple(collect.errors),
        reference_status=reference_status, subjects_found=len(subjects),
        sessions_found=len(sessions), regions_found=len(regions),
    )


def _discover_sessions(
    project: ProjectConfig, subject_path: Path, subject_id: str, collect: _Collector,
) -> list[_Session]:
    if project.sessions.mode == "single":
        return [_Session(subject_id, subject_path.name, project.sessions.default_session_id,
                         project.sessions.default_session_id)]
    paths = _glob(
        subject_path, project.sessions.discovery_pattern, collect, "session_discovery",
        subject_id=subject_id,
    )
    if not paths:
        collect.fail(
            "sessions_not_found", f"no sessions matched {project.sessions.discovery_pattern!r}",
            subject_id=subject_id, path=str(subject_path),
        )
        return []
    result: list[_Session] = []
    seen: dict[str, Path] = {}
    for path in paths:
        if not path.is_dir():
            collect.fail(
                "session_not_directory", "session discovery matched a non-directory",
                subject_id=subject_id, path=str(path),
            )
            continue
        session_id = _extract_id(project.sessions.id_regex, "session", path.name)
        if session_id is None:
            collect.fail(
                "session_id_no_match", "session folder name does not fully match sessions.id_regex",
                subject_id=subject_id, path=str(path), measured_value=path.name,
            )
            continue
        if not _safe_identifier(session_id):
            collect.fail(
                "session_id_unsafe", "sessions.id_regex produced an unsafe identifier",
                subject_id=subject_id, path=str(path), measured_value=session_id,
                threshold="letters, digits, '_', '.', '-' except '.' and '..'",
            )
            continue
        if session_id in seen:
            collect.fail(
                "duplicate_session_id", f"session_id {session_id!r} was also produced by {seen[session_id]}",
                subject_id=subject_id, session_id=session_id, path=str(path),
            )
            continue
        seen[session_id] = path
        result.append(_Session(subject_id, subject_path.name, session_id, path.name))
        collect.check(
            "session_discovered", "PASS", subject_id=subject_id, session_id=session_id,
            path=str(path), message="session folder discovered",
        )
    return result


def _match_pattern(
    root: Path, pattern: str, session: _Session, collect: _Collector, check_id: str, **scope: str,
) -> list[Path]:
    expanded = pattern.format(subject=session.subject_name, session=session.session_name)
    return _glob(root, expanded, collect, check_id, **scope)


def _glob(root: Path, pattern: str, collect: _Collector, check_id: str, **scope: str) -> list[Path]:
    if not _is_safe_relative_pattern(pattern):
        collect.fail(
            f"{check_id}_pattern", "pattern must be relative and must not contain '..'",
            path=pattern, **scope,
        )
        return []
    try:
        return sorted((path.resolve() for path in root.glob(pattern)), key=lambda path: str(path).casefold())
    except (OSError, ValueError) as exc:
        collect.fail(
            f"{check_id}_pattern", f"pattern could not be evaluated ({exc})", path=pattern, **scope,
        )
        return []


def _is_safe_relative_pattern(pattern: str) -> bool:
    posix = PurePosixPath(pattern.replace("\\", "/"))
    windows = PureWindowsPath(pattern)
    return not posix.is_absolute() and not windows.is_absolute() and not windows.drive and ".." not in posix.parts


def _extract_id(pattern: str, group: str, name: str) -> str | None:
    match = re.fullmatch(pattern, name)
    return match.group(group) if match is not None else None


def _safe_identifier(value: str) -> bool:
    return value not in {".", ".."} and SAFE_ID_PATTERN.fullmatch(value) is not None


def _wrong_match_count(
    collect: _Collector, kind: str, pattern: str, matches: list[Path], scope: dict[str, str],
) -> None:
    if matches:
        collect.fail(
            f"{kind}_ambiguous", f"pattern {pattern!r} matched {len(matches)} files; exactly one is required",
            path="; ".join(str(path) for path in matches), measured_value=str(len(matches)),
            threshold="1", **scope,
        )
    else:
        collect.fail(
            f"{kind}_missing", f"pattern {pattern!r} matched no files",
            measured_value="0", threshold="1", **scope,
        )


def _read_image_info(
    path: Path, cache: dict[Path, _ImageInfo | None], collect: _Collector, kind: str, **scope: str,
) -> _ImageInfo | None:
    if path in cache:
        return cache[path]
    if not path.is_file():
        collect.fail(f"{kind}_not_file", "path is not a file", path=str(path), **scope)
        cache[path] = None
        return None
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        collect.fail(f"{kind}_unreadable", f"SimpleITK could not read the image ({exc})", path=str(path), **scope)
        cache[path] = None
        return None
    info = _info(path, image)
    cache[path] = info
    return info


def _read_mask(
    path: Path, cache: dict[Path, tuple[_ImageInfo | None, np.ndarray | None]],
    collect: _Collector, kind: str, **scope: str,
) -> tuple[_ImageInfo | None, np.ndarray | None]:
    if path in cache:
        return cache[path]
    if not path.is_file():
        collect.fail(f"{kind}_not_file", "path is not a file", path=str(path), **scope)
        cache[path] = (None, None)
        return cache[path]
    try:
        image = sitk.ReadImage(str(path))
        array = np.asarray(sitk.GetArrayFromImage(image))
    except RuntimeError as exc:
        collect.fail(f"{kind}_unreadable", f"SimpleITK could not read the mask ({exc})", path=str(path), **scope)
        cache[path] = (None, None)
        return cache[path]
    cache[path] = (_info(path, image), array)
    return cache[path]


def _info(path: Path, image: sitk.Image) -> _ImageInfo:
    return _ImageInfo(
        path=path, dimension=image.GetDimension(),
        size=tuple(int(value) for value in image.GetSize()),
        spacing=tuple(float(value) for value in image.GetSpacing()),
        origin=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )


def _geometry_matches(left: _ImageInfo, right: _ImageInfo) -> bool:
    return (
        left.dimension == right.dimension and left.size == right.size
        and np.allclose(left.spacing, right.spacing, atol=GEOMETRY_TOLERANCE, rtol=0)
        and np.allclose(left.origin, right.origin, atol=GEOMETRY_TOLERANCE, rtol=0)
        and np.allclose(left.direction, right.direction, atol=GEOMETRY_TOLERANCE, rtol=0)
    )


def _spacing_is_reference(spacing: Iterable[float]) -> bool:
    values = tuple(spacing)
    return len(values) == 3 and bool(
        np.allclose(values, REFERENCE_SPACING_MM, atol=SPACING_TOLERANCE_MM, rtol=0)
    )


def _component_count(mask: np.ndarray, *, fully_connected: bool) -> int:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    connected = sitk.ConnectedComponent(image, fully_connected)
    array = sitk.GetArrayViewFromImage(connected)
    return int(array.max()) if array.size else 0


def _fingerprint(path: Path) -> tuple[int, str]:
    stat = path.stat()
    timestamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return stat.st_size, timestamp


def _resolve_root(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _tuple_text(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.9g}" for value in values) + "]"


def _geometry_text(info: _ImageInfo) -> str:
    return (f"size={info.size}; spacing={_tuple_text(info.spacing)}; "
            f"origin={_tuple_text(info.origin)}; direction={_tuple_text(info.direction)}")


def _values_text(values: np.ndarray) -> str:
    shown = ", ".join(str(value.item()) for value in values[:12])
    return shown + (", ..." if values.size > 12 else "")
