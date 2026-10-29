"""Atomic, resumable habitat clustering from paired voxelwise feature maps."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as connected_component_label

from radiomicviz import __version__
from radiomicviz.habitats.models import (
    HabitatCheck,
    HabitatFailure,
    HabitatRunResult,
    HabitatUnitResult,
)
from radiomicviz.habitats.plan import HabitatPlan, HabitatUnit
from radiomicviz.habitats.reference import (
    EXCLUDED_FEATURES,
    GEOMETRY_ATOL,
    GEOMETRY_RTOL,
    METHOD_ADAPTIVE,
    METHOD_FIXED,
    METHOD_GEOMETRIC,
    METHODS,
    SCIENTIFIC_CONFIG,
    SCIENTIFIC_CONFIG_SHA256,
    process_region,
)
from radiomicviz.habitats.report import (
    prepare_habitat_report_targets,
    write_habitat_report,
)
from radiomicviz.protocols import Protocol, VoxelwisePreset


class HabitatStageError(RuntimeError):
    """Habitat clustering cannot safely start or finish."""


@dataclass(frozen=True)
class _ProcessedUnit:
    result: HabitatUnitResult
    checks: tuple[HabitatCheck, ...]
    failure: HabitatFailure | None
    log_lines: tuple[str, ...]


@dataclass(frozen=True)
class _Task:
    unit: HabitatUnit
    plan: HabitatPlan
    protocol: Protocol
    preset: VoxelwisePreset
    software: Mapping[str, str]
    force: bool


class _SessionProgress:
    def __init__(
        self,
        units: tuple[HabitatUnit, ...],
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
                f"[{self._reported}/{len(self._units)}] {subject}/{session} done: "
                f"{self._units[key]} region(s), {self._failed[key]} failed"
            )
        return item


def run_habitats(
    plan: HabitatPlan,
    protocol: Protocol,
    *,
    variant: str = "b32",
    jobs: int = 1,
    dry_run: bool = False,
    force: bool = False,
    allow_version_mismatch: bool = False,
    software: Mapping[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> HabitatRunResult:
    """Run the reference fixed, adaptive and geometric methods for every region."""
    if jobs < 1:
        raise HabitatStageError("jobs must be at least 1")
    preset = protocol.voxelwise_preset(variant)
    actual_software = dict(software or _software_versions())
    deviations = list(_version_deviations(protocol.expected_software, actual_software))
    for name, expected in {"scipy": "1.11.4", "scikit-learn": "1.1.3"}.items():
        if actual_software.get(name) != expected:
            deviations.append(
                f"{name} is {actual_software.get(name, 'missing')}; reference habitat stack expects {expected}"
            )
    if plan.manifest_reference_status != "reference-compatible":
        deviations.append(f"manifest reference status is {plan.manifest_reference_status}")
    deviations.extend(_upstream_software_deviations(plan, preset, protocol))
    deviations = list(dict.fromkeys(deviations))
    if deviations and not allow_version_mismatch:
        listing = "\n".join(f"  - {item}" for item in deviations)
        raise HabitatStageError(
            "software or input state does not match the reference protocol:\n"
            f"{listing}\nUse --allow-version-mismatch only for an explicitly non-reference run."
        )

    started = _utc_now()
    planned = tuple(
        _plan_unit(unit, plan, protocol, preset, actual_software, force)
        for unit in plan.units
    )
    if dry_run:
        return _run_result(
            plan,
            protocol,
            variant,
            jobs,
            started,
            tuple(item.result for item in planned),
            tuple(check for item in planned for check in item.checks),
            tuple(item.failure for item in planned if item.failure is not None),
            actual_software,
            tuple(deviations),
            tuple(line for item in planned for line in item.log_lines),
            dry_run=True,
        )

    targets = prepare_habitat_report_targets(plan.output_root, variant)
    if progress is not None:
        progress(
            f"Running habitat clustering (variant {variant}): {len(plan.units)} region(s) "
            f"with {jobs} concurrent job(s)"
        )
    progress_tracker = _SessionProgress(plan.units, progress)
    finished: list[_ProcessedUnit] = []
    runnable: list[_Task] = []
    for item, unit in zip(planned, plan.units):
        if item.result.status in {"resumed", "resumed-skipped", "blocked"}:
            finished.append(progress_tracker(item))
        else:
            runnable.append(_Task(unit, plan, protocol, preset, actual_software, force))

    if jobs == 1:
        for task in runnable:
            finished.append(progress_tracker(_process_unit(task)))
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="radiomicviz-habitats") as pool:
            for item in pool.map(_process_unit, runnable):
                finished.append(progress_tracker(item))
    order = {unit.unit_key: index for index, unit in enumerate(plan.units)}
    finished.sort(key=lambda item: order[item.result.unit_key])
    result = _run_result(
        plan,
        protocol,
        variant,
        jobs,
        started,
        tuple(item.result for item in finished),
        tuple(check for item in finished for check in item.checks),
        tuple(item.failure for item in finished if item.failure is not None),
        actual_software,
        tuple(deviations),
        tuple(line for item in finished for line in item.log_lines),
        dry_run=False,
    )
    return write_habitat_report(result, targets)


def _run_result(
    plan: HabitatPlan,
    protocol: Protocol,
    variant: str,
    jobs: int,
    started: str,
    units: tuple[HabitatUnitResult, ...],
    checks: tuple[HabitatCheck, ...],
    failures: tuple[HabitatFailure, ...],
    software: Mapping[str, str],
    deviations: tuple[str, ...],
    logs: tuple[str, ...],
    *,
    dry_run: bool,
) -> HabitatRunResult:
    return HabitatRunResult(
        output_root=plan.output_root,
        manifest_path=plan.manifest_path,
        manifest_sha256=plan.manifest_sha256,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_sha256=protocol.sha256,
        variant=variant,
        implementation_id="reference",
        scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
        jobs=jobs,
        dry_run=dry_run,
        started_utc=started,
        finished_utc=_utc_now(),
        units=units,
        checks=checks,
        failures=failures,
        software=dict(software),
        reference_deviations=deviations,
        log_lines=logs,
    )


def _plan_unit(
    unit: HabitatUnit,
    plan: HabitatPlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
    software: Mapping[str, str],
    force: bool,
) -> _ProcessedUnit:
    completion = _completion_path(unit, plan.output_root, preset.variant)
    if not force and _completion_matches(completion, unit, plan, protocol, preset, software):
        data = json.loads(completion.read_text(encoding="utf-8"))
        skipped = data["status"] == "skipped"
        return _ProcessedUnit(
            HabitatUnitResult(
                unit_key=unit.unit_key,
                status="resumed-skipped" if skipped else "resumed",
                output_dir=unit.output_dir(plan.output_root),
                completion_path=completion,
                mask_voxels=data.get("mask_voxels"),
                rows=tuple(data.get("rows", [])),
                qc=data.get("qc"),
                message="matching completion record resumed",
            ),
            (_check(unit, "habitat_resume", "PASS", "matching habitat completion record"),),
            None,
            (f"RESUME {unit.unit_key}",),
        )
    if not force and _unit_outputs_exist(unit, plan.output_root, preset.variant):
        failure = _failure(
            unit,
            "habitat_output_blocked",
            "habitat outputs exist without a matching completion record; use --force to archive and recompute",
            completion,
        )
        return _ProcessedUnit(
            HabitatUnitResult(unit.unit_key, "blocked", unit.output_dir(plan.output_root), completion),
            (_check(unit, "habitat_resume", "FAIL", failure.message, completion),),
            failure,
            (f"BLOCKED {unit.unit_key}: {failure.message}",),
        )
    status = "replace" if force and _unit_outputs_exist(unit, plan.output_root, preset.variant) else "planned"
    return _ProcessedUnit(
        HabitatUnitResult(unit.unit_key, status, unit.output_dir(plan.output_root), None),
        (),
        None,
        (f"{status.upper()} {unit.unit_key}",),
    )


def _process_unit(task: _Task) -> _ProcessedUnit:
    unit, plan, protocol, preset = task.unit, task.plan, task.protocol, task.preset
    checks: list[HabitatCheck] = []
    logs: list[str] = []
    work_root = plan.output_root / ".tmp" / "habitats"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = work_root / f"{unit.unit_key}.{uuid4().hex}"
    work_dir.mkdir()
    archived: list[str] = []
    try:
        inputs = _load_voxelwise_inputs(unit, plan, protocol, preset)
        if inputs["status"] == "skipped":
            if task.force:
                archived.extend(_archive_outputs(unit, plan.output_root, preset.variant))
            completion = _completion_path(unit, plan.output_root, preset.variant)
            completion.parent.mkdir(parents=True, exist_ok=True)
            data = _completion_base(task, "skipped", int(inputs["mask_voxels"]), archived)
            skip_qc = {
                "subject_id": unit.subject_id,
                "session_id": unit.session_id,
                "region_id": unit.region_id,
                "n_voxels": int(inputs["mask_voxels"]),
                "representation_status": "not_available_voxelwise_stage_skipped",
            }
            data.update({"reason": "voxelwise_maps_unavailable_at_or_below_minimum_roi_size", "rows": [], "qc": skip_qc, "outputs": {}})
            _atomic_json(completion, data)
            checks.append(_check(unit, "habitat_input_maps", "WARN", data["reason"], completion))
            shutil.rmtree(work_dir)
            return _ProcessedUnit(
                HabitatUnitResult(unit.unit_key, "skipped", unit.output_dir(plan.output_root), completion, int(inputs["mask_voxels"]), (), skip_qc, str(data["reason"])),
                tuple(checks),
                None,
                (f"SKIP {unit.unit_key}: {data['reason']}",),
            )

        raw = inputs["raw"]
        names = inputs["feature_names"]
        mask = inputs["mask"]
        reference = inputs["reference"]
        mask_voxels = int(np.count_nonzero(mask))
        messages: list[str] = []
        computation = process_region(
            raw,
            names,
            mask,
            tuple(reference.GetSpacing()[::-1]),
            progress=messages.append,
        )
        if len(computation.rows) != 7:
            raise HabitatStageError(
                f"reference core returned {len(computation.rows)} metadata rows, expected 7"
            )
        staged: dict[str, Path] = {}
        finals = _mask_paths(unit, plan.output_root, preset.variant)
        for method in METHODS:
            method_dir = work_dir / method
            method_dir.mkdir()
            staged_path = method_dir / "mask.nii.gz"
            image = sitk.GetImageFromArray(computation.masks[method].astype(np.uint8))
            image.CopyInformation(reference)
            sitk.WriteImage(image, str(staged_path))
            _validate_output_mask(staged_path, reference, mask_voxels, method)
            staged[method] = staged_path
        if task.force:
            archived.extend(_archive_outputs(unit, plan.output_root, preset.variant))
        _promote_masks(staged, finals)
        rows = tuple(
            {
                "subject_id": unit.subject_id,
                "session_id": unit.session_id,
                "region_id": unit.region_id,
                "region_type": unit.region_type,
                "variant": preset.variant,
                "implementation_id": "reference",
                "mask_path": str(finals[str(row["method"])].resolve()),
                **_json_safe(row),
            }
            for row in computation.rows
        )
        qc = {
            "subject_id": unit.subject_id,
            "session_id": unit.session_id,
            "region_id": unit.region_id,
            **_json_safe(inputs["mask_qc"]),
            **_json_safe(computation.qc),
        }
        completion = _completion_path(unit, plan.output_root, preset.variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        data = _completion_base(task, "complete", mask_voxels, archived)
        data.update(
            {
                "feature_names": names,
                "feature_count": len(names),
                "rows": rows,
                "qc": qc,
                "outputs": {
                    method: _fingerprint(path) for method, path in finals.items()
                },
            }
        )
        _atomic_json(completion, data)
        checks.extend(
            (
                _check(unit, "habitat_input_maps", "PASS", f"loaded {len(names)} paired T1+FLAIR feature maps"),
                _check(unit, "habitat_metadata_rows", "PASS", "seven method-zone rows written", completion, measured="7", threshold="7"),
                _check(unit, "habitat_mask_coverage", "PASS", "all three label masks cover the complete region", completion, measured=str(mask_voxels), threshold=str(mask_voxels)),
            )
        )
        shutil.rmtree(work_dir)
        logs.extend(f"{unit.unit_key}: {message}" for message in messages)
        logs.append(f"COMPLETE {unit.unit_key}: {mask_voxels} voxels, {len(names)} features")
        return _ProcessedUnit(
            HabitatUnitResult(unit.unit_key, "complete", unit.output_dir(plan.output_root), completion, mask_voxels, rows, qc),
            tuple(checks),
            None,
            tuple(logs),
        )
    except Exception as exc:
        failure = _failure(unit, "habitat_clustering_failed", f"{type(exc).__name__}: {exc}", work_dir)
        checks.append(_check(unit, "habitat_clustering", "FAIL", failure.message, work_dir))
        return _ProcessedUnit(
            HabitatUnitResult(unit.unit_key, "failed", unit.output_dir(plan.output_root), None, message=failure.message),
            tuple(checks),
            failure,
            (f"FAIL {unit.unit_key}: {failure.message}; staging retained at {work_dir}",),
        )


def _load_voxelwise_inputs(
    unit: HabitatUnit,
    plan: HabitatPlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
) -> dict[str, object]:
    directories = {
        modality: unit.voxelwise_dir(plan.output_root, preset.variant, modality)
        for modality in ("T1", "FLAIR")
    }
    records = {
        modality: _validated_voxelwise_completion(
            directories[modality] / "completion.json",
            unit.t1_unit if modality == "T1" else unit.flair_unit,
            protocol,
            preset,
        )
        for modality in ("T1", "FLAIR")
    }
    statuses = {str(record["status"]) for record in records.values()}
    if statuses == {"skipped"}:
        counts = {int(record["mask_voxels"]) for record in records.values()}
        if len(counts) != 1:
            raise HabitatStageError("T1 and FLAIR skipped completions disagree on mask voxel count")
        return {"status": "skipped", "mask_voxels": counts.pop()}
    if statuses != {"complete"}:
        raise HabitatStageError(
            f"T1/FLAIR voxelwise completion statuses do not match: {sorted(statuses)}"
        )

    matrices: list[np.ndarray] = []
    modality_names: dict[str, list[str]] = {}
    reference: sitk.Image | None = None
    reference_geometry: dict[str, object] | None = None
    shape: tuple[int, ...] | None = None
    for modality in ("T1", "FLAIR"):
        arrays: list[np.ndarray] = []
        names: list[str] = []
        # The reference script discovers files with sorted(os.listdir(...)).
        # Preserve that lexicographic column order rather than completion-record order.
        for map_name in sorted(records[modality]["map_names"]):
            parsed = str(map_name).split("original_", 1)[-1]
            if any(excluded in parsed for excluded in EXCLUDED_FEATURES):
                continue
            path = directories[modality] / f"{map_name}.nrrd"
            image = sitk.ReadImage(str(path))
            geometry = _geometry(image)
            if reference is None:
                reference, reference_geometry = image, geometry
                shape = sitk.GetArrayViewFromImage(image).shape
            elif not _geometry_close(reference_geometry, geometry):
                raise HabitatStageError(
                    f"feature-map geometry mismatch in {path}"
                )
            arrays.append(sitk.GetArrayFromImage(image).reshape(-1))
            names.append(parsed)
        if not arrays:
            raise HabitatStageError(f"no usable voxelwise feature maps in {directories[modality]}")
        matrices.append(np.asarray(arrays).T)
        modality_names[modality] = names
    if modality_names["T1"] != modality_names["FLAIR"]:
        raise HabitatStageError("T1 and FLAIR voxelwise feature schemas differ")
    assert reference is not None and shape is not None
    mask_source = sitk.ReadImage(str(unit.mask_path))
    source_values = sitk.GetArrayFromImage(mask_source)
    source_binary = (source_values == unit.label).astype(np.uint8)
    source_image = sitk.GetImageFromArray(source_binary)
    source_image.CopyInformation(mask_source)
    resampled = sitk.Resample(
        source_image,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    mask = sitk.GetArrayFromImage(resampled).astype(np.uint8)
    if int(mask.sum()) == 0:
        raise HabitatStageError("authoritative mask has zero voxels on the feature grid")
    raw = np.concatenate(matrices, axis=1)
    flat_mask = mask.reshape(-1) > 0
    support = np.isfinite(matrices[0]).any(axis=1) & np.isfinite(matrices[1]).any(axis=1)
    if float(support[flat_mask].mean()) == 0.0:
        raise HabitatStageError("authoritative mask has zero overlap with usable feature support")
    names = [f"T1_{name}" for name in modality_names["T1"]] + [
        f"FLAIR_{name}" for name in modality_names["FLAIR"]
    ]
    source_stats = _component_stats(source_binary)
    resampled_stats = _component_stats(mask)
    return {
        "status": "complete",
        "raw": raw,
        "feature_names": names,
        "mask": mask,
        "reference": reference,
        "mask_qc": {
            "mask_source_path": str(unit.mask_path.resolve()),
            "mask_source_label": unit.label,
            "mask_source_n_voxels": int(source_binary.sum()),
            "mask_resampled_n_voxels": int(mask.sum()),
            "n_components_source": source_stats[0],
            "largest_component_fraction_source": source_stats[1],
            "second_largest_component_voxels_source": source_stats[2],
            "n_components_resampled": resampled_stats[0],
            "largest_component_fraction_resampled": resampled_stats[1],
            "second_largest_component_voxels_resampled": resampled_stats[2],
            "mask_feature_overlap_fraction": float(support[flat_mask].mean()),
        },
    }


def _validated_voxelwise_completion(
    path: Path,
    unit: object,
    protocol: Protocol,
    preset: VoxelwisePreset,
) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HabitatStageError(f"voxelwise completion is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise HabitatStageError(f"cannot read voxelwise completion {path}: {exc}") from exc
    expected = {
        "stage": "voxelwise",
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "protocol_sha256": protocol.sha256,
        "preset_sha256": preset.parameter_sha256,
        "map_schema_sha256": preset.expected_map_schema_sha256,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise HabitatStageError(f"voxelwise completion {path} has mismatched {key}")
    status = data.get("status")
    if status == "skipped":
        if (
            data.get("map_count") != 0
            or data.get("outputs") != {}
            or not isinstance(data.get("mask_voxels"), int)
            or data["mask_voxels"] > preset.minimum_roi_size
            or any(path.parent.glob("*.nrrd"))
        ):
            raise HabitatStageError(f"invalid skipped voxelwise completion: {path}")
        return data
    if status != "complete" or data.get("map_names") != list(preset.expected_map_names):
        raise HabitatStageError(f"voxelwise completion is not a valid complete unit: {path}")
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        raise HabitatStageError(f"voxelwise completion has invalid outputs: {path}")
    for name in preset.expected_map_names:
        map_path = path.parent / f"{name}.nrrd"
        if outputs.get(name) != _voxelwise_fingerprint(map_path):
            raise HabitatStageError(f"voxelwise map changed after completion: {map_path}")
    return data


def _upstream_software_deviations(
    plan: HabitatPlan,
    preset: VoxelwisePreset,
    protocol: Protocol,
) -> tuple[str, ...]:
    """Carry non-reference voxelwise software forward into habitat provenance."""
    deviations: list[str] = []
    for unit in plan.units:
        for modality in ("T1", "FLAIR"):
            path = unit.voxelwise_dir(plan.output_root, preset.variant, modality) / "completion.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            upstream = data.get("software")
            if not isinstance(upstream, dict):
                deviations.append(
                    f"upstream voxelwise unit {unit.unit_key}/{modality} has no software record"
                )
                continue
            for item in _version_deviations(protocol.expected_software, upstream):
                deviations.append(
                    f"upstream voxelwise unit {unit.unit_key}/{modality}: {item}"
                )
    return tuple(deviations)


def _completion_base(
    task: _Task,
    status: str,
    mask_voxels: int,
    archived: list[str],
) -> dict[str, object]:
    unit, plan, protocol, preset = task.unit, task.plan, task.protocol, task.preset
    return {
        "schema_version": 1,
        "stage": "habitats",
        "status": status,
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "implementation_id": "reference",
        "package_version": __version__,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": plan.manifest_sha256,
        "voxelwise_preset_id": preset.preset_id,
        "voxelwise_preset_sha256": preset.parameter_sha256,
        "scientific_config": dict(SCIENTIFIC_CONFIG),
        "scientific_config_sha256": SCIENTIFIC_CONFIG_SHA256,
        "software": dict(task.software),
        "input_mask": _fingerprint(unit.mask_path),
        "input_voxelwise_completions": {
            modality: _fingerprint(unit.voxelwise_dir(plan.output_root, preset.variant, modality) / "completion.json")
            for modality in ("T1", "FLAIR")
        },
        "label": unit.label,
        "mask_voxels": mask_voxels,
        "archived_outputs": archived,
        "finished_utc": _utc_now(),
    }


def _completion_matches(
    path: Path,
    unit: HabitatUnit,
    plan: HabitatPlan,
    protocol: Protocol,
    preset: VoxelwisePreset,
    software: Mapping[str, str],
) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        input_records = {
            modality: _fingerprint(unit.voxelwise_dir(plan.output_root, preset.variant, modality) / "completion.json")
            for modality in ("T1", "FLAIR")
        }
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "stage": "habitats",
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "implementation_id": "reference",
        "protocol_sha256": protocol.sha256,
        "voxelwise_preset_sha256": preset.parameter_sha256,
        "scientific_config_sha256": SCIENTIFIC_CONFIG_SHA256,
        "software": dict(software),
        "input_mask": _fingerprint(unit.mask_path),
        "input_voxelwise_completions": input_records,
        "label": unit.label,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        return False
    if data.get("status") == "skipped":
        return (
            data.get("outputs") == {}
            and data.get("rows") == []
            and not any(path_value.exists() for path_value in _mask_paths(unit, plan.output_root, preset.variant).values())
        )
    if data.get("status") != "complete" or len(data.get("rows", [])) != 7:
        return False
    outputs = data.get("outputs")
    finals = _mask_paths(unit, plan.output_root, preset.variant)
    try:
        return isinstance(outputs, dict) and all(
            outputs.get(method) == _fingerprint(path_value)
            for method, path_value in finals.items()
        )
    except OSError:
        return False


def _mask_paths(unit: HabitatUnit, output_root: Path, variant: str) -> dict[str, Path]:
    root = unit.output_dir(output_root)
    return {
        METHOD_FIXED: root / METHOD_FIXED / variant / "mask.nii.gz",
        METHOD_ADAPTIVE: root / METHOD_ADAPTIVE / variant / "mask.nii.gz",
        METHOD_GEOMETRIC: root / METHOD_GEOMETRIC / "mask.nii.gz",
    }


def _completion_path(unit: HabitatUnit, output_root: Path, variant: str) -> Path:
    return unit.output_dir(output_root) / "reference" / variant / "completion.json"


def _unit_outputs_exist(unit: HabitatUnit, output_root: Path, variant: str) -> bool:
    return _completion_path(unit, output_root, variant).exists() or any(
        path.exists() for path in _mask_paths(unit, output_root, variant).values()
    )


def _archive_outputs(unit: HabitatUnit, output_root: Path, variant: str) -> list[str]:
    targets = [
        unit.output_dir(output_root) / METHOD_FIXED / variant,
        unit.output_dir(output_root) / METHOD_ADAPTIVE / variant,
        unit.output_dir(output_root) / METHOD_GEOMETRIC,
        unit.output_dir(output_root) / "reference" / variant,
    ]
    archived: list[str] = []
    for target in targets:
        if target.exists():
            archived.append(str(_archive_path(target).resolve()))
    return archived


def _promote_masks(staged: Mapping[str, Path], finals: Mapping[str, Path]) -> None:
    for method in METHODS:
        final = finals[method]
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise HabitatStageError(f"refusing to replace existing habitat mask without --force: {final}")
        os.replace(staged[method], final)


def _validate_output_mask(path: Path, reference: sitk.Image, n_voxels: int, method: str) -> None:
    image = sitk.ReadImage(str(path))
    if image.GetPixelID() != sitk.sitkUInt8 or not _geometry_close(_geometry(reference), _geometry(image)):
        raise HabitatStageError(f"{method} output is not uint8 on the voxelwise feature grid")
    values = sitk.GetArrayViewFromImage(image)
    labels = set(int(value) for value in np.unique(values))
    allowed = {0, 1, 2, 3} if method == METHOD_ADAPTIVE else {0, 1, 2}
    if not labels <= allowed or int(np.count_nonzero(values)) != n_voxels:
        raise HabitatStageError(f"{method} output labels do not cover the authoritative region")


def _geometry(image: sitk.Image) -> dict[str, object]:
    return {"size": image.GetSize(), "spacing": image.GetSpacing(), "origin": image.GetOrigin(), "direction": image.GetDirection()}


def _component_stats(binary: np.ndarray) -> tuple[int, float | None, int | None]:
    labeled, count = connected_component_label(binary)
    if count == 0:
        return 0, None, None
    sizes = sorted(int(np.count_nonzero(labeled == value)) for value in range(1, count + 1))
    total = int(binary.sum())
    return count, sizes[-1] / total if total else None, sizes[-2] if len(sizes) > 1 else 0


def _geometry_close(left: Mapping[str, object] | None, right: Mapping[str, object]) -> bool:
    if left is None or left["size"] != right["size"]:
        return False
    return all(
        np.allclose(left[key], right[key], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        for key in ("spacing", "origin", "direction")
    )


def _check(
    unit: HabitatUnit,
    check_id: str,
    status: str,
    message: str,
    path: Path | None = None,
    *,
    measured: str = "",
    threshold: str = "",
) -> HabitatCheck:
    return HabitatCheck(check_id, status, unit.unit_key, unit.subject_id, unit.session_id, unit.region_id, measured, threshold, message, str(path or ""))


def _failure(unit: HabitatUnit, code: str, message: str, path: Path | None = None) -> HabitatFailure:
    return HabitatFailure(code, message, unit.unit_key, unit.subject_id, unit.session_id, unit.region_id, str(path or ""))


def _software_versions() -> dict[str, str]:
    try:
        import pywt
        import radiomics
        import scipy
        import sklearn
    except ImportError as exc:
        raise HabitatStageError(f"habitat dependency is missing: {exc.name}") from exc
    return {
        "python": platform.python_version(),
        "pyradiomics": str(radiomics.__version__).lstrip("v"),
        "numpy": np.__version__,
        "simpleitk": sitk.Version_VersionString(),
        "pywavelets": pywt.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _version_deviations(expected: Mapping[str, str], actual: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        f"{name} is {actual.get(name, 'missing')}; protocol expects {required}"
        for name, required in expected.items()
        if actual.get(name) != required
    )


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(), "sha256": _sha256(path)}


def _voxelwise_fingerprint(path: Path) -> dict[str, object]:
    """Match the upstream voxelwise completion-record fingerprint schema."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _archive_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
