"""Whole-ROI PyRadiomics extraction for every non-empty habitat zone."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol as TypingProtocol
from uuid import uuid4

import numpy as np

from radiomicviz import __version__
from radiomicviz._platform import platform_identifier, reference_platform_deviation
from radiomicviz._integrity import file_fingerprint, fingerprint_matches
from radiomicviz._pyradiomics_logging import install_pyradiomics_advisory_filters
from radiomicviz._roi import inspect_roi
from radiomicviz.habitat_features.models import (
    HabitatFeatureCheck,
    HabitatFeatureFailure,
    HabitatFeatureRunResult,
    HabitatFeatureUnitResult,
)
from radiomicviz.habitat_features.plan import (
    HabitatFeaturePlan,
    HabitatFeatureUnit,
    HabitatZone,
)
from radiomicviz.habitat_features.report import (
    prepare_habitat_feature_report_targets,
    write_habitat_feature_report,
)
from radiomicviz.protocols import Protocol, RoiPreset


class HabitatFeatureStageError(RuntimeError):
    """Habitat-zone extraction cannot safely start or finish."""


class FeatureExtractor(TypingProtocol):
    def execute(self, image: str, mask: str, *, label: int) -> Mapping[str, Any]: ...


ExtractorFactory = Callable[[Path], FeatureExtractor]


@dataclass(frozen=True)
class _ProcessedUnit:
    result: HabitatFeatureUnitResult
    checks: tuple[HabitatFeatureCheck, ...]
    failures: tuple[HabitatFeatureFailure, ...]
    log_lines: tuple[str, ...]


@dataclass(frozen=True)
class _ExtractionPlan:
    output_root: Path
    manifest_sha256: str
    manifest_reference_status: str
    habitat_reference_status: str
    habitat_index_sha256: str


@dataclass(frozen=True)
class _DefaultExtractionTask:
    unit: HabitatFeatureUnit
    plan: _ExtractionPlan
    protocol: Protocol
    preset: RoiPreset
    software: Mapping[str, str]
    force: bool


class _SessionProgress:
    def __init__(
        self,
        units: tuple[HabitatFeatureUnit, ...],
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
        self._failed[key] += len(item.failures)
        if self._finished[key] == self._units[key]:
            self._reported += 1
            subject, session = key
            self._progress(
                f"[{self._reported}/{self.sessions}] {subject}/{session} done: "
                f"{self._units[key]} method-modality unit(s), "
                f"{self._failed[key]} zone failure(s)"
            )
        return item


def run_habitat_features(
    plan: HabitatFeaturePlan,
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
) -> HabitatFeatureRunResult:
    """Extract the reference ROI feature set for each non-empty habitat label."""
    if jobs < 1:
        raise HabitatFeatureStageError("jobs must be at least 1")
    if os.name == "nt" and jobs > 61:
        raise HabitatFeatureStageError("jobs cannot exceed 61 on Windows")
    preset = protocol.roi_preset(variant)
    actual_software = dict(software or _software_versions())
    deviations = list(_version_deviations(protocol.expected_software, actual_software))
    if platform_issue := reference_platform_deviation(actual_software):
        deviations.append(platform_issue)
    if plan.manifest_reference_status != "reference-compatible":
        deviations.append(
            f"manifest reference status is {plan.manifest_reference_status}"
        )
    if plan.habitat_reference_status != "reference-compatible":
        deviations.append(
            f"habitat reference status is {plan.habitat_reference_status}"
        )
    if deviations and not allow_version_mismatch:
        listing = "\n".join(f"  - {item}" for item in deviations)
        raise HabitatFeatureStageError(
            "software or upstream state does not match the reference protocol:\n"
            f"{listing}\nUse --allow-nonreference only for an explicitly non-reference run."
        )

    started = _utc_now()
    planned = tuple(
        _plan_unit(unit, plan, preset, protocol, actual_software, force)
        for unit in plan.units
    )
    if dry_run:
        return _run_result(
            plan, protocol, preset, jobs, True, started, planned, (), (),
            actual_software, tuple(deviations), (),
        )

    targets = prepare_habitat_feature_report_targets(plan.output_root, variant)
    extraction_plan = _ExtractionPlan(
        plan.output_root,
        plan.manifest_sha256,
        plan.manifest_reference_status,
        plan.habitat_reference_status,
        plan.habitat_index_sha256,
    )
    factory = extractor_factory or _default_extractor_factory
    tasks = list(zip(plan.units, planned))
    track = _SessionProgress(plan.units, progress)
    if progress is not None:
        progress(
            f"Running habitat ROI extraction (variant {variant}): {len(tasks)} "
            f"method-modality unit(s), {sum(len(u.zones) for u in plan.units)} zone(s), "
            f"jobs={jobs}"
        )

    def process(pair: tuple[HabitatFeatureUnit, HabitatFeatureUnitResult]) -> _ProcessedUnit:
        unit, preview = pair
        if preview.status == "would-resume":
            return _resume_unit(unit, preview.output_dir, preset)
        if preview.status == "blocked":
            return _blocked_unit(unit, preview.output_dir, variant)
        return _extract_unit(
            unit, extraction_plan, protocol, preset, actual_software, factory,
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
                        unit, extraction_plan, protocol, preset, actual_software, force
                    )
                )
        if pending:
            pool = ProcessPoolExecutor(max_workers=jobs)
            broken: BrokenProcessPool | None = None
            try:
                for item in pool.map(_run_default_task, pending):
                    completed[item.result.unit_key] = track(item)
            except BrokenProcessPool as exc:
                broken = exc
                for task in pending:
                    if task.unit.unit_key not in completed:
                        item = _worker_failure(task.unit, plan.output_root, variant, exc)
                        completed[task.unit.unit_key] = track(item)
            finally:
                pool.shutdown(wait=broken is None, cancel_futures=True)
        processed = [completed[unit.unit_key] for unit, _ in tasks]
    else:
        with ThreadPoolExecutor(
            max_workers=jobs, thread_name_prefix="radiomicviz-habitat-features"
        ) as pool:
            processed = [track(item) for item in pool.map(process, tasks)]

    units = tuple(item.result for item in processed)
    checks = [check for item in processed for check in item.checks]
    checks.extend(_duplicate_feature_checks(units))
    failures = tuple(failure for item in processed for failure in item.failures)
    logs = tuple(line for item in processed for line in item.log_lines)
    result = _run_result(
        plan, protocol, preset, jobs, False, started, units, tuple(checks), failures,
        actual_software, tuple(deviations), logs,
    )
    return write_habitat_feature_report(result, targets)


def _run_default_task(task: _DefaultExtractionTask) -> _ProcessedUnit:
    return _extract_unit(
        task.unit, task.plan, task.protocol, task.preset, task.software,
        _default_extractor_factory, force=task.force,
    )


def _run_result(
    plan: HabitatFeaturePlan,
    protocol: Protocol,
    preset: RoiPreset,
    jobs: int,
    dry_run: bool,
    started: str,
    units: tuple[HabitatFeatureUnitResult, ...],
    checks: tuple[HabitatFeatureCheck, ...],
    failures: tuple[HabitatFeatureFailure, ...],
    software: Mapping[str, str],
    deviations: tuple[str, ...],
    logs: tuple[str, ...],
) -> HabitatFeatureRunResult:
    return HabitatFeatureRunResult(
        output_root=plan.output_root,
        manifest_path=plan.manifest_path,
        manifest_sha256=plan.manifest_sha256,
        habitat_index_path=plan.habitat_index_path,
        habitat_index_sha256=plan.habitat_index_sha256,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_sha256=protocol.sha256,
        variant=preset.variant,
        preset_id=preset.preset_id,
        preset_path=preset.parameter_path,
        preset_sha256=preset.parameter_sha256,
        jobs=jobs,
        dry_run=dry_run,
        started_utc=started,
        finished_utc=_utc_now(),
        units=units,
        checks=checks,
        failures=failures,
        software=software,
        reference_deviations=deviations,
        log_lines=logs,
    )


def _plan_unit(
    unit: HabitatFeatureUnit,
    plan: HabitatFeaturePlan,
    preset: RoiPreset,
    protocol: Protocol,
    software: Mapping[str, str],
    force: bool,
) -> HabitatFeatureUnitResult:
    output_dir = unit.output_dir(plan.output_root, preset.variant)
    completion = output_dir / "completion.json"
    if force or not output_dir.exists():
        status = "would-run"
    elif _completion_matches(completion, unit, plan, preset, protocol, software):
        status = "would-resume"
    else:
        status = "blocked"
    return HabitatFeatureUnitResult(
        unit_key=unit.unit_key,
        status=status,
        output_dir=output_dir,
        completion_path=completion if output_dir.exists() else None,
        rows=(),
    )


def _extract_unit(
    unit: HabitatFeatureUnit,
    plan: _ExtractionPlan,
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
    work_root = plan.output_root / ".tmp" / "habitat_features"
    work_root.mkdir(parents=True, exist_ok=True)
    short = hashlib.sha256(unit.unit_key.encode("utf-8")).hexdigest()[:12]
    work_dir = work_root / f"u-{short}-{uuid4().hex[:8]}"
    work_dir.mkdir()
    started = _utc_now()
    rows: list[OrderedDict[str, Any]] = []
    zone_records: list[dict[str, Any]] = []
    checks: list[HabitatFeatureCheck] = []
    failures: list[HabitatFeatureFailure] = []
    log = [f"[{unit.unit_key}] extraction started"]
    diagnostics_schema: tuple[str, ...] = ()
    feature_schema: tuple[str, ...] = ()
    try:
        for zone in unit.zones:
            row = _base_row(unit, zone, preset.variant, _reference_status(plan, protocol, software))
            zone_started = _utc_now()
            inspection = None
            if zone.n_voxels < preset.minimum_roi_size:
                checks.append(
                    _check(
                        unit, zone, "habitat_feature_small_zone", "WARN",
                        measured=str(zone.n_voxels),
                        threshold=f">={preset.minimum_roi_size} preferred",
                        message="small habitat zones are attempted and retained by the reference analysis",
                        path=unit.mask_path,
                    )
                )
            try:
                inspection = inspect_roi(unit.image_path, unit.mask_path, zone.label)
                if inspection.voxel_count != zone.n_voxels:
                    raise HabitatFeatureStageError(
                        f"source mask contains {inspection.voxel_count} voxels for label "
                        f"{zone.label}; authenticated source-grid count is {zone.n_voxels}"
                    )
                if inspection.dimensions < 2:
                    checks.append(
                        _check(
                            unit, zone, "habitat_feature_roi_dimensions", "WARN",
                            measured=str(inspection.dimensions), threshold=">=2 required by PyRadiomics",
                            message="zone will still be attempted and any extraction error retained",
                            path=unit.mask_path,
                        )
                    )
                extractor = factory(preset.parameter_path)
                vector = OrderedDict(
                    extractor.execute(
                        str(unit.image_path), str(unit.mask_path), label=zone.label
                    )
                )
                diagnostics = tuple(k for k in vector if k.startswith("diagnostics_"))
                features = tuple(k for k in vector if _is_feature_name(k))
                _validate_vector(
                    unit, zone, vector, diagnostics, features, preset, checks
                )
                if diagnostics_schema and diagnostics != diagnostics_schema:
                    raise HabitatFeatureStageError("diagnostic schema changed between zones")
                if feature_schema and features != feature_schema:
                    raise HabitatFeatureStageError("feature schema changed between zones")
                diagnostics_schema = diagnostics
                feature_schema = features
                row["diagnostics_original_voxels"] = _csv_value(
                    vector.get("diagnostics_Mask-original_VoxelNum", "")
                )
                row["diagnostics_interpolated_voxels"] = _csv_value(
                    vector.get("diagnostics_Mask-interpolated_VoxelNum", "")
                )
                row["zone_status"] = "complete"
                row["error"] = ""
                row.update((key, _csv_value(value)) for key, value in vector.items())
                zone_records.append(
                    {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "label": zone.label,
                        "source_zone_voxels": zone.n_voxels,
                        "habitat_grid_voxels": zone.habitat_grid_voxels,
                        "status": "complete",
                        "error": "",
                        "diagnostics_count": len(diagnostics),
                        "feature_count": len(features),
                        "feature_schema_sha256": _schema_hash(features),
                        "started_utc": zone_started,
                        "finished_utc": _utc_now(),
                    }
                )
            except Exception as exc:  # one failed zone never drops its row or siblings
                message = str(exc) or type(exc).__name__
                expected_skip = inspection is not None and (
                    inspection.voxel_count < preset.minimum_roi_size
                    or inspection.dimensions < 2
                )
                row["diagnostics_original_voxels"] = ""
                row["diagnostics_interpolated_voxels"] = ""
                row["zone_status"] = "skipped" if expected_skip else "failed"
                row["error"] = message
                if expected_skip:
                    checks.append(
                        _check(
                            unit,
                            zone,
                            "habitat_feature_expected_skip",
                            "WARN",
                            measured=(
                                f"voxels={inspection.voxel_count}; "
                                f"dimensions={inspection.dimensions}"
                            ),
                            threshold=(
                                f">={preset.minimum_roi_size} voxels and >=2 dimensions"
                            ),
                            message=(
                                "PyRadiomics rejected an objectively small or "
                                "one-dimensional habitat zone; the row was retained "
                                "as an expected skip"
                            ),
                            path=unit.mask_path,
                        )
                    )
                else:
                    failures.append(
                        _failure(
                            unit,
                            zone,
                            "habitat_feature_extraction_failed",
                            message,
                        )
                    )
                zone_records.append(
                    {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "label": zone.label,
                        "source_zone_voxels": zone.n_voxels,
                        "habitat_grid_voxels": zone.habitat_grid_voxels,
                        "status": "skipped" if expected_skip else "failed",
                        "error": message,
                        "started_utc": zone_started,
                        "finished_utc": _utc_now(),
                    }
                )
                log.append(
                    f"[{unit.unit_key} zone {zone.zone_id}] "
                    f"{'expected skip' if expected_skip else 'failed'}: "
                    f"{type(exc).__name__}: {message}"
                )
            rows.append(row)

        result_path = work_dir / "features.csv"
        _write_rows(result_path, rows)
        completion = {
            "schema_version": 1,
            "stage": "habitat-features",
            "status": "complete" if not failures else "complete_with_failures",
            "unit_key": unit.unit_key,
            "variant": preset.variant,
            "method": unit.method,
            "modality": unit.modality,
            "package_version": __version__,
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "protocol_sha256": protocol.sha256,
            "manifest_sha256": plan.manifest_sha256,
            "habitat_index_sha256": plan.habitat_index_sha256,
            "preset_id": preset.preset_id,
            "preset_sha256": preset.parameter_sha256,
            "software": dict(software),
            "reference_status": _reference_status(plan, protocol, software),
            "input_image": file_fingerprint(unit.image_path, sha256=unit.image_sha256),
            "input_mask": file_fingerprint(unit.mask_path, sha256=unit.mask_sha256),
            "habitat_completion": file_fingerprint(
                unit.habitat_completion_path, sha256=unit.habitat_completion_sha256
            ),
            "zones": zone_records,
            "started_utc": started,
            "finished_utc": _utc_now(),
            "supersedes": str(archived.resolve()) if archived else None,
            "outputs": {
                "features": file_fingerprint(
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
                unit, None, "habitat_feature_unit",
                "FAIL" if failures else ("WARN" if any(
                    row.get("zone_status") == "skipped" for row in rows
                ) else "PASS"),
                measured=(
                    f"{len(rows)} zone row(s); "
                    f"{sum(row.get('zone_status') == 'skipped' for row in rows)} skipped; "
                    f"{len(failures)} failed"
                ),
                threshold="one retained row per non-empty zone",
                message="habitat ROI extraction unit completed",
                path=final_dir / "features.csv",
            )
        )
        log.append(
            f"[{unit.unit_key}] committed {len(rows)} zone row(s), "
            f"{sum(row.get('zone_status') == 'skipped' for row in rows)} skipped, "
            f"{len(failures)} failed"
        )
        return _ProcessedUnit(
            HabitatFeatureUnitResult(
                unit.unit_key, "complete", final_dir, final_dir / "completion.json",
                tuple(rows), diagnostics_schema, feature_schema,
            ),
            tuple(checks), tuple(failures), tuple(log),
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        return _worker_failure(unit, plan.output_root, preset.variant, exc)
    finally:
        if work_dir.exists() and work_dir != final_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def _resume_unit(
    unit: HabitatFeatureUnit, output_dir: Path, preset: RoiPreset
) -> _ProcessedUnit:
    completion = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
    with (output_dir / "features.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    normalized_diagnostic_fields = {
        "diagnostics_original_voxels",
        "diagnostics_interpolated_voxels",
    }
    diagnostics = tuple(
        field
        for field in fields
        if field.startswith("diagnostics_")
        and field not in normalized_diagnostic_fields
    )
    features = tuple(field for field in fields if _is_feature_name(field))
    failures: list[HabitatFeatureFailure] = []
    skipped = 0
    zone_by_id = {str(zone.zone_id): zone for zone in unit.zones}
    for row in rows:
        if row.get("zone_status") == "skipped":
            skipped += 1
        elif row.get("error"):
            zone = zone_by_id.get(str(row.get("zone_id")))
            if zone is not None:
                failures.append(
                    _failure(
                        unit, zone, "habitat_feature_extraction_failed",
                        str(row["error"]),
                    )
                )
    check = _check(
        unit, None, "habitat_feature_resume",
        "FAIL" if failures else ("WARN" if skipped else "PASS"),
        measured=(
            f"{len(rows)} zone row(s); {skipped} skipped; {len(failures)} failed"
        ),
        threshold="matching authenticated completion",
        message="matching habitat-feature result was reused",
        path=output_dir / "features.csv",
    )
    return _ProcessedUnit(
        HabitatFeatureUnitResult(
            unit.unit_key, "resumed", output_dir, output_dir / "completion.json",
            tuple(rows), diagnostics, features,
        ),
        (check,), tuple(failures),
        (f"[{unit.unit_key}] matching result resumed",),
    )


def _blocked_unit(
    unit: HabitatFeatureUnit, output_dir: Path, variant: str
) -> _ProcessedUnit:
    message = (
        "output exists without a matching completion record; use --force to archive "
        "and recompute"
    )
    rows = tuple(
        dict(
            _base_row(unit, zone, variant, ""),
            zone_status="blocked",
            error=message,
        )
        for zone in unit.zones
    )
    failure = HabitatFeatureFailure(
        "habitat_feature_output_blocked", message, unit.unit_key,
        unit.subject_id, unit.session_id, unit.region_id, unit.method,
        "", "", unit.modality, str(output_dir),
    )
    return _ProcessedUnit(
        HabitatFeatureUnitResult(
            unit.unit_key, "blocked", output_dir, output_dir / "completion.json", rows
        ),
        (), (failure,), (f"[{unit.unit_key}] blocked: {message}",),
    )


def _worker_failure(
    unit: HabitatFeatureUnit, output_root: Path, variant: str, exc: BaseException
) -> _ProcessedUnit:
    message = (
        "a process-pool worker exited unexpectedly; completed units were preserved "
        f"({type(exc).__name__}: {exc})"
    )
    output_dir = unit.output_dir(output_root, variant)
    rows = tuple(
        dict(
            _base_row(unit, zone, variant, ""),
            zone_status="failed",
            error=message,
        )
        for zone in unit.zones
    )
    failures = tuple(
        _failure(unit, zone, "habitat_feature_worker_lost", message)
        for zone in unit.zones
    )
    return _ProcessedUnit(
        HabitatFeatureUnitResult(unit.unit_key, "failed", output_dir, None, rows),
        (), failures, (f"[{unit.unit_key}] failed: {message}",),
    )


def _validate_vector(
    unit: HabitatFeatureUnit,
    zone: HabitatZone,
    vector: Mapping[str, Any],
    diagnostics: tuple[str, ...],
    features: tuple[str, ...],
    preset: RoiPreset,
    checks: list[HabitatFeatureCheck],
) -> None:
    shape_count = sum(name.startswith("original_shape_") for name in features)
    requirements = (
        ("habitat_feature_diagnostics_count", len(diagnostics), preset.expected_diagnostics_count),
        ("habitat_feature_count", len(features), preset.expected_feature_count),
        ("habitat_shape_feature_count", shape_count, preset.expected_shape_feature_count),
    )
    failures: list[str] = []
    for check_id, actual, expected in requirements:
        ok = actual == expected
        checks.append(
            _check(
                unit, zone, check_id, "PASS" if ok else "FAIL",
                measured=str(actual), threshold=str(expected),
                message=f"found {actual}; expected {expected}", path=unit.mask_path,
            )
        )
        if not ok:
            failures.append(f"{check_id}: {actual} (expected {expected})")
    feature_hash = _schema_hash(features)
    schema_ok = feature_hash == preset.expected_feature_schema_sha256
    checks.append(
        _check(
            unit, zone, "habitat_feature_schema", "PASS" if schema_ok else "FAIL",
            measured=feature_hash, threshold=preset.expected_feature_schema_sha256,
            message="ordered feature-name SHA-256", path=preset.parameter_path,
        )
    )
    if not schema_ok:
        failures.append("feature schema does not match whole-region reference schema")
    settings = vector.get("diagnostics_Configuration_Settings")
    settings_ok = isinstance(settings, Mapping) and (
        settings.get("binCount") == preset.bin_count
        and list(settings.get("resampledPixelSpacing", [])) == [1, 1, 1]
        and settings.get("interpolator") == "sitkBSpline"
        and settings.get("normalize") is False
        and settings.get("label") == zone.label
    )
    checks.append(
        _check(
            unit, zone, "habitat_feature_effective_settings",
            "PASS" if settings_ok else "FAIL", measured=str(settings),
            threshold=(
                f"binCount={preset.bin_count}; resampledPixelSpacing=[1, 1, 1]; "
                f"interpolator=sitkBSpline; normalize=False; label={zone.label}"
            ),
            message="effective PyRadiomics settings match the reference ROI preset",
            path=preset.parameter_path,
        )
    )
    if not settings_ok:
        failures.append("effective settings do not match the preset")
    original = vector.get("diagnostics_Mask-original_VoxelNum")
    interpolated = vector.get("diagnostics_Mask-interpolated_VoxelNum")
    counts_ok = _integer_equal(original, zone.n_voxels) and _integer_equal(
        interpolated, zone.n_voxels
    )
    checks.append(
        _check(
            unit, zone, "habitat_feature_mask_voxels",
            "PASS" if counts_ok else "WARN",
            measured=(
                f"source={zone.n_voxels}; diagnostics original={original}; "
                f"interpolated={interpolated}"
            ),
            threshold="diagnostic counts equal source zone count",
            message=(
                "zone voxel count preserved"
                if counts_ok else "zone count changed during correction or resampling"
            ),
            path=unit.mask_path,
        )
    )
    nonfinite = sum(_is_nonfinite(vector[name]) for name in features)
    checks.append(
        _check(
            unit, zone, "habitat_feature_nonfinite", "PASS" if nonfinite == 0 else "WARN",
            measured=str(nonfinite), threshold="0 preferred",
            message="non-finite feature values are retained and reported",
            path=unit.mask_path,
        )
    )
    if failures:
        raise HabitatFeatureStageError("; ".join(failures))


def _completion_matches(
    path: Path,
    unit: HabitatFeatureUnit,
    plan: HabitatFeaturePlan,
    preset: RoiPreset,
    protocol: Protocol,
    software: Mapping[str, str],
) -> bool:
    result_path = path.with_name("features.csv")
    if not path.is_file() or not result_path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "stage": "habitat-features",
        "unit_key": unit.unit_key,
        "variant": preset.variant,
        "method": unit.method,
        "modality": unit.modality,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": plan.manifest_sha256,
        "habitat_index_sha256": plan.habitat_index_sha256,
        "preset_sha256": preset.parameter_sha256,
        "software": dict(software),
    }
    if any(data.get(key) != value for key, value in expected.items()):
        return False
    if data.get("status") not in {"complete", "complete_with_failures"}:
        return False
    if not fingerprint_matches(
        data.get("input_image"), unit.image_path, expected_sha256=unit.image_sha256
    ) or not fingerprint_matches(
        data.get("input_mask"), unit.mask_path, expected_sha256=unit.mask_sha256
    ) or not fingerprint_matches(
        data.get("habitat_completion"), unit.habitat_completion_path,
        expected_sha256=unit.habitat_completion_sha256,
    ):
        return False
    records = data.get("zones")
    if not isinstance(records, list) or len(records) != len(unit.zones):
        return False
    expected_zones = [
        (z.zone_id, z.zone_name, z.label, z.n_voxels, z.habitat_grid_voxels)
        for z in unit.zones
    ]
    actual_zones = [
        (
            r.get("zone_id"), r.get("zone_name"), r.get("label"),
            r.get("source_zone_voxels"), r.get("habitat_grid_voxels"),
        )
        for r in records if isinstance(r, dict)
    ]
    if actual_zones != expected_zones:
        return False
    return fingerprint_matches(data.get("outputs", {}).get("features"), result_path)


def _base_row(
    unit: HabitatFeatureUnit,
    zone: HabitatZone,
    variant: str,
    reference_status: str,
) -> OrderedDict[str, Any]:
    row: OrderedDict[str, Any] = OrderedDict(
        (
            ("subject_id", unit.subject_id),
            ("session_id", unit.session_id),
            ("region_id", unit.region_id),
            ("region_type", unit.region_type),
            ("method", unit.method),
            ("zone_id", zone.zone_id),
            ("zone_name", zone.zone_name),
            ("mask_label_value", zone.label),
            ("source_zone_voxels", zone.n_voxels),
            ("habitat_grid_voxels", zone.habitat_grid_voxels),
            ("zone_flagged_small", zone.n_voxels < 10),
            ("habitat_missing_flag", zone.missing),
            ("eligible_for_split", zone.eligible_for_split),
            ("chosen_k", zone.chosen_k),
            ("silhouette_score", zone.silhouette_score),
            ("habitat_valid", zone.habitat_valid),
            ("modality", unit.modality),
            ("variant", variant),
            ("reference_status", reference_status),
            ("image_path", str(unit.image_path)),
            ("mask_path", str(unit.mask_path)),
        )
    )
    row.update(unit.metadata)
    return row


def _duplicate_feature_checks(
    units: tuple[HabitatFeatureUnitResult, ...],
) -> tuple[HabitatFeatureCheck, ...]:
    feature_names = next((unit.feature_names for unit in units if unit.feature_names), ())
    if not feature_names:
        return ()
    groups: defaultdict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        for row in unit.rows:
            if not row.get("error") and all(name in row for name in feature_names):
                groups[tuple(str(row.get(name, "")) for name in feature_names)].append(row)
    checks: list[HabitatFeatureCheck] = []
    for duplicate_rows in groups.values():
        if len(duplicate_rows) < 2:
            continue
        first = duplicate_rows[0]
        identities = [
            f"{r.get('subject_id')}/{r.get('session_id')}/{r.get('region_id')}/"
            f"{r.get('method')}/z{r.get('zone_id')}/{r.get('modality')}"
            for r in duplicate_rows
        ]
        checks.append(
            HabitatFeatureCheck(
                "habitat_feature_duplicate_vector", "WARN",
                "__".join(
                    str(first.get(k, ""))
                    for k in ("subject_id", "session_id", "region_id", "method", "modality")
                ),
                str(first.get("subject_id", "")), str(first.get("session_id", "")),
                str(first.get("region_id", "")), str(first.get("method", "")),
                str(first.get("zone_id", "")), str(first.get("zone_name", "")),
                str(first.get("modality", "")), measured_value=str(len(duplicate_rows)),
                threshold="unique vectors expected across distinct zones",
                message="identical feature vectors: " + "; ".join(identities),
                path=str(first.get("mask_path", "")),
            )
        )
    return tuple(checks)


def _check(
    unit: HabitatFeatureUnit,
    zone: HabitatZone | None,
    check_id: str,
    status: str,
    *,
    measured: str,
    threshold: str,
    message: str,
    path: Path,
) -> HabitatFeatureCheck:
    return HabitatFeatureCheck(
        check_id, status, unit.unit_key, unit.subject_id, unit.session_id,
        unit.region_id, unit.method, "" if zone is None else zone.zone_id,
        "" if zone is None else zone.zone_name, unit.modality,
        measured, threshold, message, str(path),
    )


def _failure(
    unit: HabitatFeatureUnit,
    zone: HabitatZone,
    code: str,
    message: str,
) -> HabitatFeatureFailure:
    return HabitatFeatureFailure(
        code, message, unit.unit_key, unit.subject_id, unit.session_id,
        unit.region_id, unit.method, zone.zone_id, zone.zone_name, unit.modality,
        str(unit.mask_path),
    )


def _reference_status(
    plan: _ExtractionPlan, protocol: Protocol, software: Mapping[str, str]
) -> str:
    if (
        plan.manifest_reference_status != "reference-compatible"
        or plan.habitat_reference_status != "reference-compatible"
        or _version_deviations(protocol.expected_software, software)
    ):
        return "non-reference"
    return "reference-compatible"


def _default_extractor_factory(parameter_path: Path) -> FeatureExtractor:
    try:
        import SimpleITK as sitk
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise HabitatFeatureStageError(
            "PyRadiomics is not installed; create the documented validation environment"
        ) from exc
    install_pyradiomics_advisory_filters()
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    return RadiomicsFeatureExtractor(str(parameter_path))


def _software_versions() -> dict[str, str]:
    try:
        import platform
        import pywt
        import radiomics
        import scipy
        import SimpleITK as sitk
        import trimesh
    except ImportError as exc:
        raise HabitatFeatureStageError(
            f"habitat-feature extraction dependency is missing: {exc.name}"
        ) from exc
    return {
        "platform": platform_identifier(),
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
        for name, required in expected.items() if actual.get(name) != required
    )


def _is_feature_name(name: str) -> bool:
    return name.startswith(
        (
            "original_", "log-", "wavelet-", "square_", "squareroot_",
            "logarithm_", "exponential_", "gradient_", "lbp-3D",
        )
    )


def _schema_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


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
    return value.item() if isinstance(value, np.generic) else value


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, restval="", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
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


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
