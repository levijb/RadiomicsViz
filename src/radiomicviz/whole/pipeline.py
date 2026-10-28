"""Whole-region PyRadiomics execution with atomic, resumable unit outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol as TypingProtocol
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from radiomicviz import __version__
from radiomicviz.protocols import Protocol, RoiPreset
from radiomicviz.whole.models import (
    WholeCheck,
    WholeFailure,
    WholeRunResult,
    WholeUnitResult,
)
from radiomicviz.whole.plan import WholePlan, WholeUnit
from radiomicviz.whole.report import prepare_whole_report_targets, write_whole_report


class WholeStageError(RuntimeError):
    """Whole-region extraction cannot safely start or finish."""


class FeatureExtractor(TypingProtocol):
    def execute(
        self, image: str, mask: str, *, label: int
    ) -> Mapping[str, Any]:
        ...


ExtractorFactory = Callable[[Path], FeatureExtractor]


@dataclass(frozen=True)
class _ProcessedUnit:
    result: WholeUnitResult
    checks: tuple[WholeCheck, ...]
    failure: WholeFailure | None
    log_lines: tuple[str, ...]


@dataclass(frozen=True)
class _DefaultExtractionTask:
    unit: WholeUnit
    plan: WholePlan
    protocol: Protocol
    preset: RoiPreset
    software: Mapping[str, str]
    force: bool


def run_whole(
    plan: WholePlan,
    protocol: Protocol,
    *,
    variant: str = "b32",
    jobs: int = 1,
    dry_run: bool = False,
    force: bool = False,
    allow_version_mismatch: bool = False,
    extractor_factory: ExtractorFactory | None = None,
    software: Mapping[str, str] | None = None,
) -> WholeRunResult:
    """Run whole-region extraction for every manifest region and modality.

    The unit of resume is one subject, session, region, modality and variant. ``jobs``
    controls concurrent units; every unit receives its own extractor instance.
    """
    if jobs < 1:
        raise WholeStageError("jobs must be at least 1")
    preset = protocol.roi_preset(variant)
    actual_software = dict(software or _software_versions())
    deviations = list(_version_deviations(protocol.expected_software, actual_software))
    if plan.manifest_reference_status != "reference-compatible":
        deviations.append(
            f"manifest reference status is {plan.manifest_reference_status}"
        )
    if deviations and not allow_version_mismatch:
        listing = "\n".join(f"  - {item}" for item in deviations)
        raise WholeStageError(
            "software or input state does not match the reference protocol:\n"
            f"{listing}\nUse --allow-version-mismatch only for an explicitly non-reference run."
        )

    started = _utc_now()
    planned = tuple(
        _plan_unit(unit, plan.output_root, preset, protocol, actual_software, force)
        for unit in plan.units
    )
    if dry_run:
        return WholeRunResult(
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

    targets = prepare_whole_report_targets(plan.output_root, variant)
    factory = extractor_factory or _default_extractor_factory
    tasks = list(zip(plan.units, planned))

    def process(pair: tuple[WholeUnit, WholeUnitResult]) -> _ProcessedUnit:
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
        processed = [process(pair) for pair in tasks]
    elif extractor_factory is None:
        # PyRadiomics parameter validation uses process-global YAML parser state and is
        # not safe when two extractors initialize in threads. Separate processes also
        # match the isolation of the historical batch implementation.
        completed: dict[str, _ProcessedUnit] = {}
        pending: list[_DefaultExtractionTask] = []
        for unit, preview in tasks:
            if preview.status == "would-resume":
                completed[unit.unit_key] = _resume_unit(unit, preview.output_dir, preset)
            elif preview.status == "blocked":
                completed[unit.unit_key] = _blocked_unit(
                    unit, preview.output_dir, variant
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
                    completed[item.result.unit_key] = item
        processed = [completed[unit.unit_key] for unit, _preview in tasks]
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="radiomicviz-whole") as pool:
            processed = list(pool.map(process, tasks))

    results = tuple(item.result for item in processed)
    checks = tuple(check for item in processed for check in item.checks)
    failures = tuple(item.failure for item in processed if item.failure is not None)
    log_lines = tuple(line for item in processed for line in item.log_lines)
    run_result = WholeRunResult(
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
    return write_whole_report(run_result, targets)


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
    unit: WholeUnit,
    output_root: Path,
    preset: RoiPreset,
    protocol: Protocol,
    software: Mapping[str, str],
    force: bool,
) -> WholeUnitResult:
    output_dir = unit.output_dir(output_root, preset.variant)
    completion_path = output_dir / "completion.json"
    if force or not output_dir.exists():
        status = "would-run"
    elif _completion_matches(completion_path, unit, preset, protocol, software):
        status = "would-resume"
    else:
        status = "blocked"
    return WholeUnitResult(
        unit_key=unit.unit_key,
        status=status,
        output_dir=output_dir,
        completion_path=completion_path if output_dir.exists() else None,
        row={},
    )


def _extract_unit(
    unit: WholeUnit,
    plan: WholePlan,
    protocol: Protocol,
    preset: RoiPreset,
    software: Mapping[str, str],
    factory: ExtractorFactory,
    *,
    force: bool,
) -> _ProcessedUnit:
    final_dir = unit.output_dir(plan.output_root, preset.variant)
    archived: Path | None = None
    if final_dir.exists():
        if not force:
            return _blocked_unit(unit, final_dir, preset.variant)
        archived = _archive_directory(final_dir)
    work_root = plan.output_root / ".tmp" / "whole"
    work_root.mkdir(parents=True, exist_ok=True)
    short_key = hashlib.sha256(unit.unit_key.encode("utf-8")).hexdigest()[:12]
    work_dir = work_root / f"u-{short_key}-{uuid4().hex[:8]}"
    work_dir.mkdir()
    started = _utc_now()
    checks: list[WholeCheck] = []
    log = [f"[{unit.unit_key}] extraction started"]
    try:
        mask_voxels = _mask_voxel_count(unit)
        extractor = factory(preset.parameter_path)
        vector = OrderedDict(
            extractor.execute(
                str(unit.image_path),
                str(unit.mask_path),
                label=unit.label,
            )
        )
        diagnostics = tuple(key for key in vector if key.startswith("diagnostics_"))
        features = tuple(key for key in vector if not key.startswith("diagnostics_"))
        _validate_schema(unit, vector, diagnostics, features, preset, mask_voxels, checks)
        row = _base_row(unit, preset.variant, mask_voxels=mask_voxels)
        row.update((key, _csv_value(value)) for key, value in vector.items())
        row["error"] = ""
        result_path = work_dir / "whole_features.csv"
        _write_one_row_csv(result_path, row)
        finished = _utc_now()
        completion = {
            "schema_version": 1,
            "stage": "whole",
            "status": "complete",
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
            "diagnostics_count": len(diagnostics),
            "feature_count": len(features),
            "feature_schema_sha256": _schema_hash(features),
            "started_utc": started,
            "finished_utc": finished,
            "supersedes": str(archived.resolve()) if archived else None,
            "outputs": {
                "whole_features": _fingerprint(
                    result_path, recorded_path=final_dir / result_path.name
                )
            },
        }
        _atomic_text(
            work_dir / "completion.json",
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.replace(final_dir)
        checks.append(
            _check(
                unit,
                "whole_unit",
                "PASS",
                measured=f"{len(features)} features",
                threshold=f"{preset.expected_feature_count} features",
                message="whole-region extraction completed",
                path=final_dir / "whole_features.csv",
            )
        )
        log.append(f"[{unit.unit_key}] extraction completed with {len(features)} features")
        return _ProcessedUnit(
            result=WholeUnitResult(
                unit_key=unit.unit_key,
                status="complete",
                output_dir=final_dir,
                completion_path=final_dir / "completion.json",
                row=row,
                diagnostic_names=diagnostics,
                feature_names=features,
            ),
            checks=tuple(checks),
            failure=None,
            log_lines=tuple(log),
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        log.append(f"[{unit.unit_key}] extraction failed: {type(exc).__name__}: {message}")
        row = _base_row(unit, preset.variant, mask_voxels="")
        row["error"] = message
        return _ProcessedUnit(
            result=WholeUnitResult(
                unit_key=unit.unit_key,
                status="failed",
                output_dir=final_dir,
                completion_path=None,
                row=row,
            ),
            checks=tuple(checks),
            failure=WholeFailure(
                error_code="whole_extraction_failed",
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


def _resume_unit(unit: WholeUnit, output_dir: Path, preset: RoiPreset) -> _ProcessedUnit:
    result_path = output_dir / "whole_features.csv"
    with result_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    if len(rows) != 1:
        return _blocked_unit(unit, output_dir, preset.variant)
    diagnostics = tuple(field for field in fields if field.startswith("diagnostics_"))
    features = tuple(field for field in fields if _is_feature_name(field))
    check = _check(
        unit,
        "whole_resume",
        "PASS",
        measured=f"{len(features)} features",
        threshold=f"{preset.expected_feature_count} features",
        message="matching completed unit was reused",
        path=result_path,
    )
    return _ProcessedUnit(
        result=WholeUnitResult(
            unit_key=unit.unit_key,
            status="resumed",
            output_dir=output_dir,
            completion_path=output_dir / "completion.json",
            row=rows[0],
            diagnostic_names=diagnostics,
            feature_names=features,
        ),
        checks=(check,),
        failure=None,
        log_lines=(f"[{unit.unit_key}] matching result resumed",),
    )


def _blocked_unit(unit: WholeUnit, output_dir: Path, variant: str) -> _ProcessedUnit:
    message = "output exists without a matching completion record; use --force to archive and recompute"
    row = _base_row(unit, variant, mask_voxels="")
    row["error"] = message
    return _ProcessedUnit(
        result=WholeUnitResult(
            unit_key=unit.unit_key,
            status="blocked",
            output_dir=output_dir,
            completion_path=output_dir / "completion.json",
            row=row,
        ),
        checks=(),
        failure=WholeFailure(
            error_code="whole_output_blocked",
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


def _validate_schema(
    unit: WholeUnit,
    vector: Mapping[str, Any],
    diagnostics: tuple[str, ...],
    features: tuple[str, ...],
    preset: RoiPreset,
    mask_voxels: int,
    checks: list[WholeCheck],
) -> None:
    feature_hash = _schema_hash(features)
    shape_count = sum(name.startswith("original_shape_") for name in features)
    requirements = (
        (
            "whole_diagnostics_count",
            len(diagnostics),
            preset.expected_diagnostics_count,
            "diagnostic columns",
        ),
        (
            "whole_feature_count",
            len(features),
            preset.expected_feature_count,
            "features",
        ),
        (
            "whole_shape_feature_count",
            shape_count,
            preset.expected_shape_feature_count,
            "shape features",
        ),
    )
    failures: list[str] = []
    for check_id, actual, expected, noun in requirements:
        ok = actual == expected
        checks.append(
            _check(
                unit,
                check_id,
                "PASS" if ok else "FAIL",
                measured=str(actual),
                threshold=str(expected),
                message=f"found {actual} {noun}; expected {expected}",
                path=unit.mask_path,
            )
        )
        if not ok:
            failures.append(f"{actual} {noun} (expected {expected})")
    schema_ok = feature_hash == preset.expected_feature_schema_sha256
    checks.append(
        _check(
            unit,
            "whole_feature_schema",
            "PASS" if schema_ok else "FAIL",
            measured=feature_hash,
            threshold=preset.expected_feature_schema_sha256,
            message="ordered feature-name SHA-256",
            path=preset.parameter_path,
        )
    )
    if not schema_ok:
        failures.append("ordered feature-name SHA-256 does not match the reference")

    settings = vector.get("diagnostics_Configuration_Settings")
    settings_ok = isinstance(settings, Mapping) and (
        settings.get("binCount") == preset.bin_count
        and list(settings.get("resampledPixelSpacing", [])) == [1, 1, 1]
        and settings.get("interpolator") == "sitkBSpline"
        and settings.get("normalize") is False
        and settings.get("label") == unit.label
    )
    checks.append(
        _check(
            unit,
            "whole_effective_settings",
            "PASS" if settings_ok else "FAIL",
            measured=str(settings),
            threshold=(
                f"binCount={preset.bin_count}; resampledPixelSpacing=[1, 1, 1]; "
                f"interpolator=sitkBSpline; normalize=False; label={unit.label}"
            ),
            message="effective PyRadiomics settings match the selected preset",
            path=preset.parameter_path,
        )
    )
    if not settings_ok:
        failures.append("effective PyRadiomics settings do not match the preset")

    original_count = vector.get("diagnostics_Mask-original_VoxelNum")
    interpolated_count = vector.get("diagnostics_Mask-interpolated_VoxelNum")
    count_ok = _integer_equal(original_count, mask_voxels)
    interpolated_ok = _integer_equal(interpolated_count, mask_voxels)
    checks.append(
        _check(
            unit,
            "whole_mask_voxel_count",
            "PASS" if count_ok and interpolated_ok else "WARN",
            measured=(
                f"source={mask_voxels}; diagnostics original={original_count}; "
                f"interpolated={interpolated_count}"
            ),
            threshold="diagnostic counts equal source mask count",
            message=(
                "mask voxel counts are preserved"
                if count_ok and interpolated_ok
                else "mask voxel count changed during PyRadiomics correction or resampling"
            ),
            path=unit.mask_path,
        )
    )
    nonfinite = sum(_is_nonfinite(vector[name]) for name in features)
    checks.append(
        _check(
            unit,
            "whole_nonfinite_features",
            "PASS" if nonfinite == 0 else "WARN",
            measured=str(nonfinite),
            threshold="0 preferred; retained and reported for small or degenerate regions",
            message="non-finite feature values are retained rather than silently dropping the row",
            path=unit.mask_path,
        )
    )
    if failures:
        raise WholeStageError("; ".join(failures))


def _completion_matches(
    path: Path,
    unit: WholeUnit,
    preset: RoiPreset,
    protocol: Protocol,
    software: Mapping[str, str],
) -> bool:
    result_path = path.with_name("whole_features.csv")
    if not path.is_file() or not result_path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "stage": "whole",
        "status": "complete",
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "protocol_sha256": protocol.sha256,
        "preset_sha256": preset.parameter_sha256,
        "software": dict(software),
        "input_image": _fingerprint(unit.image_path),
        "input_mask": _fingerprint(unit.mask_path),
        "label": unit.label,
        "feature_schema_sha256": preset.expected_feature_schema_sha256,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        return False
    return data.get("outputs", {}).get("whole_features") == _fingerprint(result_path)


def _default_extractor_factory(parameter_path: Path) -> FeatureExtractor:
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise WholeStageError(
            "PyRadiomics is not installed; create the documented validation environment"
        ) from exc
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    return RadiomicsFeatureExtractor(str(parameter_path))


def _software_versions() -> dict[str, str]:
    try:
        import pywt
        import radiomics
        import scipy
        import trimesh
    except ImportError as exc:
        raise WholeStageError(
            f"whole-region extraction dependency is missing: {exc.name}"
        ) from exc
    return {
        "python": platform.python_version(),
        "pyradiomics": str(radiomics.__version__).lstrip("v"),
        "numpy": np.__version__,
        "simpleitk": sitk.Version_VersionString(),
        "pywavelets": pywt.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
    }


def _version_deviations(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(
        f"{name} is {actual.get(name, 'missing')}; protocol expects {required}"
        for name, required in expected.items()
        if actual.get(name) != required
    )


def _base_row(
    unit: WholeUnit, variant: str, *, mask_voxels: int | str
) -> OrderedDict[str, Any]:
    row: OrderedDict[str, Any] = OrderedDict(
        (
            ("subject_id", unit.subject_id),
            ("session_id", unit.session_id),
            ("region_id", unit.region_id),
            ("region_type", unit.region_type),
            ("modality", unit.modality),
            ("variant", variant),
            ("image_path", str(unit.image_path)),
            ("mask_path", str(unit.mask_path)),
            ("label", unit.label),
            ("mask_voxels", mask_voxels),
        )
    )
    row.update(unit.metadata)
    return row


def _mask_voxel_count(unit: WholeUnit) -> int:
    mask = sitk.ReadImage(str(unit.mask_path))
    values = sitk.GetArrayViewFromImage(mask)
    count = int(np.count_nonzero(values == unit.label))
    if count == 0:
        raise WholeStageError(
            f"mask contains no voxels with label {unit.label}: {unit.mask_path}"
        )
    return count


def _schema_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _is_feature_name(name: str) -> bool:
    return name.startswith(
        (
            "original_",
            "log-",
            "wavelet-",
            "square_",
            "squareroot_",
            "logarithm_",
            "exponential_",
            "gradient_",
            "lbp-3D",
        )
    )


def _integer_equal(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def _is_nonfinite(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_one_row_csv(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _check(
    unit: WholeUnit,
    check_id: str,
    status: str,
    *,
    measured: str,
    threshold: str,
    message: str,
    path: Path,
) -> WholeCheck:
    return WholeCheck(
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
