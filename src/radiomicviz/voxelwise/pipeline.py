"""Voxelwise PyRadiomics execution with atomic, resumable unit outputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol as TypingProtocol
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from radiomicviz import __version__
from radiomicviz.protocols import Protocol, VoxelwisePreset
from radiomicviz.voxelwise.models import (
    VoxelwiseCheck,
    VoxelwiseFailure,
    VoxelwiseRunResult,
    VoxelwiseUnitResult,
)
from radiomicviz.voxelwise.plan import VoxelwisePlan, VoxelwiseUnit
from radiomicviz.voxelwise.report import (
    prepare_voxelwise_report_targets,
    write_voxelwise_report,
)


class VoxelwiseStageError(RuntimeError):
    """Voxelwise extraction cannot safely start or finish."""


class FeatureExtractor(TypingProtocol):
    def execute(
        self, image: str, mask: str, *, label: int, voxelBased: bool
    ) -> Mapping[str, Any]:
        ...


ExtractorFactory = Callable[[Path], FeatureExtractor]


@dataclass(frozen=True)
class _ProcessedUnit:
    result: VoxelwiseUnitResult
    checks: tuple[VoxelwiseCheck, ...]
    failure: VoxelwiseFailure | None
    log_lines: tuple[str, ...]


@dataclass(frozen=True)
class _DefaultExtractionTask:
    unit: VoxelwiseUnit
    plan: VoxelwisePlan
    protocol: Protocol
    preset: VoxelwisePreset
    software: Mapping[str, str]
    force: bool


class _SessionProgress:
    """Report each subject/session once its last unit has finished."""

    def __init__(
        self,
        units: tuple[VoxelwiseUnit, ...],
        progress: Callable[[str], None] | None,
    ) -> None:
        self._progress = progress
        self._session_of = {
            unit.unit_key: (unit.subject_id, unit.session_id) for unit in units
        }
        self._units = Counter(self._session_of.values())
        self._finished: Counter[tuple[str, str]] = Counter()
        self._failed: Counter[tuple[str, str]] = Counter()
        self._reported = 0

    @property
    def sessions(self) -> int:
        return len(self._units)

    def __call__(self, item: _ProcessedUnit) -> _ProcessedUnit:
        if self._progress is None:
            return item
        key = self._session_of[item.result.unit_key]
        self._finished[key] += 1
        self._failed[key] += item.failure is not None
        if self._finished[key] == self._units[key]:
            self._reported += 1
            subject, session = key
            self._progress(
                f"[{self._reported}/{self.sessions}] {subject}/{session} done: "
                f"{self._units[key]} unit(s), {self._failed[key]} failed"
            )
        return item


def run_voxelwise(
    plan: VoxelwisePlan,
    protocol: Protocol,
    *,
    variant: str = "b32",
    jobs: int = 1,
    dry_run: bool = False,
    force: bool = False,
    allow_version_mismatch: bool = False,
    extractor_factory: ExtractorFactory | None = None,
    software: Mapping[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> VoxelwiseRunResult:
    """Extract one NRRD feature map per enabled feature and manifest unit.

    Every unit receives its own extractor. Real concurrent extraction uses separate
    processes because PyRadiomics parameter validation has process-global state.
    Regions below the preset's minimum ROI size receive a resumable ``skipped``
    completion record and a QC warning rather than failing the invocation.
    """
    if jobs < 1:
        raise VoxelwiseStageError("jobs must be at least 1")
    preset = protocol.voxelwise_preset(variant)
    actual_software = dict(software or _software_versions())
    deviations = list(_version_deviations(protocol.expected_software, actual_software))
    if plan.manifest_reference_status != "reference-compatible":
        deviations.append(
            f"manifest reference status is {plan.manifest_reference_status}"
        )
    if deviations and not allow_version_mismatch:
        listing = "\n".join(f"  - {item}" for item in deviations)
        raise VoxelwiseStageError(
            "software or input state does not match the reference protocol:\n"
            f"{listing}\nUse --allow-version-mismatch only for an explicitly non-reference run."
        )

    started = _utc_now()
    planned = tuple(
        _plan_unit(unit, plan.output_root, preset, protocol, actual_software, force)
        for unit in plan.units
    )
    if dry_run:
        return VoxelwiseRunResult(
            output_root=plan.output_root,
            manifest_path=plan.manifest_path,
            manifest_sha256=plan.manifest_sha256,
            protocol_id=protocol.protocol_id,
            protocol_version=protocol.protocol_version,
            protocol_sha256=protocol.sha256,
            variant=variant,
            preset_id=preset.preset_id,
            preset_path=preset.parameter_path,
            preset_sha256=preset.parameter_sha256,
            jobs=jobs,
            dry_run=True,
            started_utc=started,
            finished_utc=_utc_now(),
            units=planned,
            checks=(),
            failures=(),
            software=actual_software,
            reference_deviations=tuple(deviations),
        )

    targets = prepare_voxelwise_report_targets(plan.output_root, variant)
    factory = extractor_factory or _default_extractor_factory
    tasks = list(zip(plan.units, planned))
    track = _SessionProgress(plan.units, progress)
    if progress is not None:
        progress(
            f"Running voxelwise extraction (variant {variant}): {len(tasks)} unit(s) "
            f"across {track.sessions} subject/session(s), jobs={jobs}"
        )

    def process(pair: tuple[VoxelwiseUnit, VoxelwiseUnitResult]) -> _ProcessedUnit:
        unit, preview = pair
        if preview.status == "would-resume":
            return _resume_unit(unit, preview.output_dir, preset)
        if preview.status == "blocked":
            return _blocked_unit(unit, preview.output_dir, variant)
        return _extract_unit(
            unit,
            plan,
            protocol,
            preset,
            actual_software,
            factory,
            force=force,
        )

    if jobs == 1:
        processed = [track(process(pair)) for pair in tasks]
    elif extractor_factory is None:
        completed: dict[str, _ProcessedUnit] = {}
        pending: list[_DefaultExtractionTask] = []
        for unit, preview in tasks:
            if preview.status == "would-resume":
                completed[unit.unit_key] = track(
                    _resume_unit(unit, preview.output_dir, preset)
                )
            elif preview.status == "blocked":
                completed[unit.unit_key] = track(
                    _blocked_unit(unit, preview.output_dir, variant)
                )
            else:
                pending.append(
                    _DefaultExtractionTask(
                        unit=unit,
                        plan=plan,
                        protocol=protocol,
                        preset=preset,
                        software=actual_software,
                        force=force,
                    )
                )
        if pending:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for item in pool.map(_run_default_extraction_task, pending):
                    completed[item.result.unit_key] = track(item)
        processed = [completed[unit.unit_key] for unit, _preview in tasks]
    else:
        with ThreadPoolExecutor(
            max_workers=jobs, thread_name_prefix="radiomicviz-voxelwise"
        ) as pool:
            processed = [track(item) for item in pool.map(process, tasks)]

    results = tuple(item.result for item in processed)
    checks = tuple(check for item in processed for check in item.checks)
    failures = tuple(item.failure for item in processed if item.failure is not None)
    log_lines = tuple(line for item in processed for line in item.log_lines)
    run_result = VoxelwiseRunResult(
        output_root=plan.output_root,
        manifest_path=plan.manifest_path,
        manifest_sha256=plan.manifest_sha256,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_sha256=protocol.sha256,
        variant=variant,
        preset_id=preset.preset_id,
        preset_path=preset.parameter_path,
        preset_sha256=preset.parameter_sha256,
        jobs=jobs,
        dry_run=False,
        started_utc=started,
        finished_utc=_utc_now(),
        units=results,
        checks=checks,
        failures=failures,
        software=actual_software,
        reference_deviations=tuple(deviations),
        log_lines=log_lines,
    )
    return write_voxelwise_report(run_result, targets)


def _run_default_extraction_task(task: _DefaultExtractionTask) -> _ProcessedUnit:
    """Pickle-safe process-pool entry point for one real extraction unit."""
    return _extract_unit(
        task.unit,
        task.plan,
        task.protocol,
        task.preset,
        task.software,
        _default_extractor_factory,
        force=task.force,
    )


def _plan_unit(
    unit: VoxelwiseUnit,
    output_root: Path,
    preset: VoxelwisePreset,
    protocol: Protocol,
    software: Mapping[str, str],
    force: bool,
) -> VoxelwiseUnitResult:
    output_dir = _output_dir(unit, output_root, preset.variant)
    completion_path = output_dir / "completion.json"
    if force or not output_dir.exists():
        status = "would-run"
    elif _completion_matches(completion_path, unit, preset, protocol, software):
        status = "would-resume"
    else:
        status = "blocked"
    return VoxelwiseUnitResult(
        unit_key=unit.unit_key,
        status=status,
        output_dir=output_dir,
        completion_path=completion_path if output_dir.exists() else None,
    )


def _extract_unit(
    unit: VoxelwiseUnit,
    plan: VoxelwisePlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
    software: Mapping[str, str],
    factory: ExtractorFactory,
    *,
    force: bool,
) -> _ProcessedUnit:
    final_dir = _output_dir(unit, plan.output_root, preset.variant)
    archived: Path | None = None
    if final_dir.exists():
        if not force:
            return _blocked_unit(unit, final_dir, preset.variant)
        archived = _archive_directory(final_dir)
    work_root = plan.output_root / ".tmp" / "voxelwise"
    work_root.mkdir(parents=True, exist_ok=True)
    short_key = hashlib.sha256(unit.unit_key.encode("utf-8")).hexdigest()[:12]
    work_dir = work_root / f"u-{short_key}-{uuid4().hex[:8]}"
    work_dir.mkdir()
    started = _utc_now()
    checks: list[VoxelwiseCheck] = []
    log = [f"[{unit.unit_key}] extraction started"]
    try:
        mask_voxels = _mask_voxel_count(unit)
        if mask_voxels < preset.minimum_roi_size:
            return _commit_skip(
                unit,
                plan,
                protocol,
                preset,
                software,
                final_dir,
                work_dir,
                archived,
                started,
                mask_voxels,
                checks,
                log,
            )

        extractor = factory(preset.parameter_path)
        vector = OrderedDict(
            extractor.execute(
                str(unit.image_path),
                str(unit.mask_path),
                label=unit.label,
                voxelBased=True,
            )
        )
        maps = OrderedDict(
            (name, value) for name, value in vector.items() if isinstance(value, sitk.Image)
        )
        _validate_map_schema(unit, maps, vector, preset, checks)
        expected_geometry = _expected_map_geometry(unit, preset.kernel_radius)
        output_fingerprints: dict[str, dict[str, object]] = {}
        for name in preset.expected_map_names:
            target = work_dir / f"{name}.nrrd"
            _write_and_validate_map(
                unit, name, maps[name], target, expected_geometry
            )
            output_fingerprints[name] = _fingerprint(
                target, recorded_path=final_dir / target.name
            )
        checks.append(
            _check(
                unit,
                "voxelwise_maps",
                "PASS",
                measured=f"{len(output_fingerprints)} readable float64 NRRDs",
                threshold=f"{preset.expected_map_count} maps with expected geometry and finite voxels",
                message="voxelwise feature maps passed output validation",
                path=final_dir,
            )
        )
        finished = _utc_now()
        completion = _completion_base(
            unit,
            plan,
            protocol,
            preset,
            software,
            status="complete",
            mask_voxels=mask_voxels,
            started=started,
            finished=finished,
            archived=archived,
        )
        completion.update(
            {
                "map_count": len(output_fingerprints),
                "map_names": list(preset.expected_map_names),
                "map_schema_sha256": preset.expected_map_schema_sha256,
                "outputs": output_fingerprints,
            }
        )
        _atomic_text(
            work_dir / "completion.json",
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.replace(final_dir)
        log.append(
            f"[{unit.unit_key}] extraction completed with {len(output_fingerprints)} maps"
        )
        return _ProcessedUnit(
            result=VoxelwiseUnitResult(
                unit_key=unit.unit_key,
                status="complete",
                output_dir=final_dir,
                completion_path=final_dir / "completion.json",
                mask_voxels=mask_voxels,
                map_names=preset.expected_map_names,
            ),
            checks=tuple(checks),
            failure=None,
            log_lines=tuple(log),
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        log.append(
            f"[{unit.unit_key}] extraction failed: {type(exc).__name__}: {message}"
        )
        return _ProcessedUnit(
            result=VoxelwiseUnitResult(
                unit_key=unit.unit_key,
                status="failed",
                output_dir=final_dir,
                completion_path=None,
                message=message,
            ),
            checks=tuple(checks),
            failure=VoxelwiseFailure(
                error_code="voxelwise_extraction_failed",
                message=message,
                unit_key=unit.unit_key,
                subject_id=unit.subject_id,
                session_id=unit.session_id,
                region_id=unit.region_id,
                modality=unit.modality,
                path=str(unit.mask_path),
            ),
            log_lines=tuple(log + [f"[{unit.unit_key}] staging retained at {work_dir}"]),
        )


def _commit_skip(
    unit: VoxelwiseUnit,
    plan: VoxelwisePlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
    software: Mapping[str, str],
    final_dir: Path,
    work_dir: Path,
    archived: Path | None,
    started: str,
    mask_voxels: int,
    checks: list[VoxelwiseCheck],
    log: list[str],
) -> _ProcessedUnit:
    reason = (
        f"region has {mask_voxels} voxels; voxelwise preset requires at least "
        f"{preset.minimum_roi_size}"
    )
    checks.append(
        _check(
            unit,
            "voxelwise_minimum_roi_size",
            "WARN",
            measured=str(mask_voxels),
            threshold=f">={preset.minimum_roi_size}",
            message=reason,
            path=unit.mask_path,
        )
    )
    completion = _completion_base(
        unit,
        plan,
        protocol,
        preset,
        software,
        status="skipped",
        mask_voxels=mask_voxels,
        started=started,
        finished=_utc_now(),
        archived=archived,
    )
    completion.update(
        {
            "skip_reason": reason,
            "map_count": 0,
            "map_names": [],
            "map_schema_sha256": preset.expected_map_schema_sha256,
            "outputs": {},
        }
    )
    _atomic_text(
        work_dir / "completion.json",
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.replace(final_dir)
    log.append(f"[{unit.unit_key}] skipped: {reason}")
    return _ProcessedUnit(
        result=VoxelwiseUnitResult(
            unit_key=unit.unit_key,
            status="skipped",
            output_dir=final_dir,
            completion_path=final_dir / "completion.json",
            mask_voxels=mask_voxels,
            message=reason,
        ),
        checks=tuple(checks),
        failure=None,
        log_lines=tuple(log),
    )


def _resume_unit(
    unit: VoxelwiseUnit, output_dir: Path, preset: VoxelwisePreset
) -> _ProcessedUnit:
    completion_path = output_dir / "completion.json"
    data = json.loads(completion_path.read_text(encoding="utf-8"))
    skipped = data.get("status") == "skipped"
    names = tuple(data.get("map_names", ()))
    status = "resumed-skipped" if skipped else "resumed"
    check = _check(
        unit,
        "voxelwise_resume",
        "WARN" if skipped else "PASS",
        measured=(
            f"skipped at {data.get('mask_voxels')} voxels"
            if skipped
            else f"{len(names)} maps"
        ),
        threshold=(
            f">={preset.minimum_roi_size} voxels"
            if skipped
            else f"{preset.expected_map_count} maps"
        ),
        message=(
            str(data.get("skip_reason", "matching skipped unit was reused"))
            if skipped
            else "matching completed unit was reused"
        ),
        path=completion_path,
    )
    return _ProcessedUnit(
        result=VoxelwiseUnitResult(
            unit_key=unit.unit_key,
            status=status,
            output_dir=output_dir,
            completion_path=completion_path,
            mask_voxels=data.get("mask_voxels"),
            map_names=names,
            message=str(data.get("skip_reason", "")),
        ),
        checks=(check,),
        failure=None,
        log_lines=(f"[{unit.unit_key}] matching result resumed",),
    )


def _blocked_unit(
    unit: VoxelwiseUnit, output_dir: Path, variant: str
) -> _ProcessedUnit:
    message = (
        "output exists without a matching completion record; use --force to archive "
        "and recompute"
    )
    return _ProcessedUnit(
        result=VoxelwiseUnitResult(
            unit_key=unit.unit_key,
            status="blocked",
            output_dir=output_dir,
            completion_path=output_dir / "completion.json",
            message=message,
        ),
        checks=(),
        failure=VoxelwiseFailure(
            error_code="voxelwise_output_blocked",
            message=message,
            unit_key=unit.unit_key,
            subject_id=unit.subject_id,
            session_id=unit.session_id,
            region_id=unit.region_id,
            modality=unit.modality,
            path=str(output_dir),
        ),
        log_lines=(f"[{unit.unit_key}] blocked: {message}",),
    )


def _validate_map_schema(
    unit: VoxelwiseUnit,
    maps: Mapping[str, sitk.Image],
    vector: Mapping[str, Any],
    preset: VoxelwisePreset,
    checks: list[VoxelwiseCheck],
) -> None:
    actual = tuple(maps)
    missing = sorted(set(preset.expected_map_names) - set(actual))
    unexpected = sorted(set(actual) - set(preset.expected_map_names))
    schema_ok = not missing and not unexpected and len(actual) == preset.expected_map_count
    checks.append(
        _check(
            unit,
            "voxelwise_map_schema",
            "PASS" if schema_ok else "FAIL",
            measured=f"{len(actual)} maps; missing={missing}; unexpected={unexpected}",
            threshold=f"{preset.expected_map_count} exact protocol map names",
            message="voxelwise feature-map schema",
            path=preset.parameter_path,
        )
    )
    settings = vector.get("diagnostics_Configuration_Settings")
    spacing = settings.get("resampledPixelSpacing") if isinstance(settings, Mapping) else None
    settings_ok = isinstance(settings, Mapping) and (
        settings.get("binCount") == preset.bin_count
        and settings.get("normalize") is False
        and settings.get("label") == unit.label
        and spacing in (None, [])
    )
    checks.append(
        _check(
            unit,
            "voxelwise_effective_settings",
            "PASS" if settings_ok else "FAIL",
            measured=str(settings),
            threshold=(
                f"binCount={preset.bin_count}; no resampling; normalize=False; "
                f"label={unit.label}; voxelBased=True at execute()"
            ),
            message="effective PyRadiomics settings match the selected preset",
            path=preset.parameter_path,
        )
    )
    problems: list[str] = []
    if not schema_ok:
        problems.append("feature-map names do not match the protocol")
    if not settings_ok:
        problems.append("effective PyRadiomics settings do not match the preset")
    if problems:
        raise VoxelwiseStageError("; ".join(problems))


def _write_and_validate_map(
    unit: VoxelwiseUnit,
    name: str,
    image: sitk.Image,
    target: Path,
    expected_geometry: tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]],
) -> None:
    sitk.WriteImage(image, str(target))
    if not target.is_file() or target.stat().st_size == 0:
        raise VoxelwiseStageError(f"empty feature map written for {name}")
    saved = sitk.ReadImage(str(target))
    if saved.GetPixelID() != sitk.sitkFloat64:
        raise VoxelwiseStageError(
            f"feature map {name} is {saved.GetPixelIDTypeAsString()}, expected 64-bit float"
        )
    expected_size, expected_spacing, expected_origin, expected_direction = expected_geometry
    geometry_ok = (
        tuple(saved.GetSize()) == expected_size
        and _close(saved.GetSpacing(), expected_spacing)
        and _close(saved.GetOrigin(), expected_origin)
        and _close(saved.GetDirection(), expected_direction)
    )
    if not geometry_ok:
        raise VoxelwiseStageError(
            f"feature map {name} geometry does not equal the padded region crop"
        )
    values = sitk.GetArrayViewFromImage(saved)
    if values.size == 0 or not np.isfinite(values).any():
        raise VoxelwiseStageError(f"feature map {name} contains no finite voxels")


def _expected_map_geometry(
    unit: VoxelwiseUnit, radius: int
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    image = sitk.ReadImage(str(unit.image_path))
    mask = sitk.ReadImage(str(unit.mask_path))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(unit.label):
        raise VoxelwiseStageError(f"mask does not contain label {unit.label}")
    bounding_box = stats.GetBoundingBox(unit.label)
    dimension = image.GetDimension()
    start = bounding_box[:dimension]
    size = bounding_box[dimension:]
    image_size = image.GetSize()
    lower = tuple(max(0, start[axis] - radius) for axis in range(dimension))
    upper = tuple(
        min(image_size[axis], start[axis] + size[axis] + radius)
        for axis in range(dimension)
    )
    expected_size = tuple(upper[axis] - lower[axis] for axis in range(dimension))
    expected_origin = tuple(image.TransformIndexToPhysicalPoint(lower))
    return (
        expected_size,
        tuple(image.GetSpacing()),
        expected_origin,
        tuple(image.GetDirection()),
    )


def _completion_matches(
    path: Path,
    unit: VoxelwiseUnit,
    preset: VoxelwisePreset,
    protocol: Protocol,
    software: Mapping[str, str],
) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "stage": "voxelwise",
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "protocol_sha256": protocol.sha256,
        "preset_sha256": preset.parameter_sha256,
        "software": dict(software),
        "input_image": _fingerprint(unit.image_path),
        "input_mask": _fingerprint(unit.mask_path),
        "label": unit.label,
        "map_schema_sha256": preset.expected_map_schema_sha256,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        return False
    status = data.get("status")
    if status == "skipped":
        return (
            isinstance(data.get("mask_voxels"), int)
            and data["mask_voxels"] < preset.minimum_roi_size
            and data.get("map_count") == 0
            and data.get("map_names") == []
            and data.get("outputs") == {}
            and not any(path.parent.glob("*.nrrd"))
        )
    if status != "complete":
        return False
    if data.get("map_names") != list(preset.expected_map_names):
        return False
    if data.get("map_count") != preset.expected_map_count:
        return False
    outputs = data.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(preset.expected_map_names):
        return False
    for name in preset.expected_map_names:
        map_path = path.parent / f"{name}.nrrd"
        if not map_path.is_file() or outputs.get(name) != _fingerprint(map_path):
            return False
    return True


def _completion_base(
    unit: VoxelwiseUnit,
    plan: VoxelwisePlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
    software: Mapping[str, str],
    *,
    status: str,
    mask_voxels: int,
    started: str,
    finished: str,
    archived: Path | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "voxelwise",
        "status": status,
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "package_version": __version__,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": plan.manifest_sha256,
        "preset_id": preset.preset_id,
        "preset_sha256": preset.parameter_sha256,
        "software": dict(software),
        "input_image": _fingerprint(unit.image_path),
        "input_mask": _fingerprint(unit.mask_path),
        "label": unit.label,
        "mask_voxels": mask_voxels,
        "minimum_roi_size": preset.minimum_roi_size,
        "kernel_radius": preset.kernel_radius,
        "started_utc": started,
        "finished_utc": finished,
        "supersedes": str(archived.resolve()) if archived else None,
    }


def _default_extractor_factory(parameter_path: Path) -> FeatureExtractor:
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise VoxelwiseStageError(
            "PyRadiomics is not installed; create the documented validation environment"
        ) from exc
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    return RadiomicsFeatureExtractor(str(parameter_path))


def _software_versions() -> dict[str, str]:
    try:
        import pywt
        import radiomics
    except ImportError as exc:
        raise VoxelwiseStageError(
            f"voxelwise extraction dependency is missing: {exc.name}"
        ) from exc
    return {
        "python": platform.python_version(),
        "pyradiomics": str(radiomics.__version__).lstrip("v"),
        "numpy": np.__version__,
        "simpleitk": sitk.Version_VersionString(),
        "pywavelets": pywt.__version__,
    }


def _version_deviations(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(
        f"{name} is {actual.get(name, 'missing')}; protocol expects {required}"
        for name, required in expected.items()
        if actual.get(name) != required
    )


def _output_dir(unit: VoxelwiseUnit, output_root: Path, variant: str) -> Path:
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
        / unit.modality
    )


def _mask_voxel_count(unit: VoxelwiseUnit) -> int:
    mask = sitk.ReadImage(str(unit.mask_path))
    values = sitk.GetArrayViewFromImage(mask)
    count = int(np.count_nonzero(values == unit.label))
    if count == 0:
        raise VoxelwiseStageError(
            f"mask contains no voxels with label {unit.label}: {unit.mask_path}"
        )
    return count


def _close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= 1e-4 for a, b in zip(left, right)
    )


def _archive_directory(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def _fingerprint(path: Path, *, recorded_path: Path | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str((recorded_path or path).resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _check(
    unit: VoxelwiseUnit,
    check_id: str,
    status: str,
    *,
    measured: str,
    threshold: str,
    message: str,
    path: Path,
) -> VoxelwiseCheck:
    return VoxelwiseCheck(
        check_id=check_id,
        status=status,
        unit_key=unit.unit_key,
        subject_id=unit.subject_id,
        session_id=unit.session_id,
        region_id=unit.region_id,
        modality=unit.modality,
        measured_value=measured,
        threshold=threshold,
        message=message,
        path=str(path),
    )


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
