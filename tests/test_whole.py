from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import load_protocol
from radiomicviz.whole import load_whole_plan, run_whole, summarize_whole
from radiomicviz.whole.pipeline import _DefaultExtractionTask, _ExtractionPlan

FEATURE_NAMES = ("original_shape_Elongation", "original_firstorder_Mean")
FAKE_SOFTWARE = {
    "python": "3.10.10",
    "pyradiomics": "3.0.1",
    "numpy": "1.26.4",
    "simpleitk": "2.5.5",
    "pywavelets": "1.8.0",
}


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[1:7, 1:7, 1:7] = np.arange(1, 217, dtype=np.float32).reshape(6, 6, 6)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_mask(path: Path, block) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[block] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _case(tmp_path: Path):
    data_root = tmp_path / "data"
    subject = data_root / "sub-001"
    _write_image(subject / "anat" / "sub-001_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-001_FLAIR_preprocessed.nii.gz")
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", np.s_[2:5, 2:5, 2:5])
    _write_mask(subject / "lesions" / "lesion_002.nii.gz", np.s_[6:7, 1:3, 1:6])
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(tmp_path / "output")
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    written = write_manifest_outputs(
        build_manifest_preview(load_project_config(project_path))
    )
    return written.manifest_path, tmp_path / "output"


def _test_protocol():
    protocol = load_protocol()
    preset = protocol.roi_preset("b32")
    test_preset = replace(
        preset,
        expected_feature_count=2,
        expected_shape_feature_count=1,
        expected_diagnostics_count=4,
        expected_feature_schema_sha256=hashlib.sha256(
            "\n".join(FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
    )
    return replace(
        protocol,
        expected_software=FAKE_SOFTWARE,
        variants={"b32": test_preset},
    )


class FakeExtractor:
    def __init__(self, *, fail_mask: str | None = None):
        self.fail_mask = fail_mask

    def execute(self, image: str, mask: str, *, label: int):
        if self.fail_mask and self.fail_mask in mask:
            raise ValueError("deliberate extraction failure")
        values = sitk.GetArrayFromImage(sitk.ReadImage(mask))
        count = int(np.count_nonzero(values == label))
        return OrderedDict(
            (
                ("diagnostics_Versions_PyRadiomics", "v3.0.1"),
                (
                    "diagnostics_Configuration_Settings",
                    {
                        "binCount": 32,
                        "resampledPixelSpacing": [1, 1, 1],
                        "interpolator": "sitkBSpline",
                        "normalize": False,
                        "label": label,
                    },
                ),
                ("diagnostics_Mask-original_VoxelNum", count),
                ("diagnostics_Mask-interpolated_VoxelNum", count),
                (FEATURE_NAMES[0], 0.7),
                (FEATURE_NAMES[1], 312.5),
            )
        )


def _factory(_parameter_path: Path):
    return FakeExtractor()


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_whole_extracts_every_region_and_modality_and_writes_cohort_tables(tmp_path):
    manifest_path, output_root = _case(tmp_path)

    result = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        jobs=2,
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert result.valid
    assert result.succeeded == 4
    assert result.resumed == 0
    tidy_rows = _rows(result.table_path)
    assert len(tidy_rows) == 4
    assert {row["reference_status"] for row in tidy_rows} == {"reference-compatible"}
    assert all(row["image_path"] for row in tidy_rows)
    assert all("analysis_image_path" not in row for row in tidy_rows)
    assert len(_rows(result.reference_table_path)) == 4
    assert _rows(result.errors_path) == []
    assert {row["status"] for row in _rows(result.checks_path)} == {"PASS"}
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_count"] == 2
    assert metadata["diagnostics_count"] == 4
    assert metadata["completed"] == 4
    for unit in result.units:
        assert (unit.output_dir / "whole_features.csv").is_file()
        assert (unit.output_dir / "completion.json").is_file()
        assert unit.output_dir.is_relative_to(output_root)


def test_whole_rerun_resumes_and_dry_run_writes_nothing(tmp_path):
    manifest_path, output_root = _case(tmp_path)
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()
    first = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    first_mtime = (first.units[0].output_dir / "whole_features.csv").stat().st_mtime_ns

    preview = run_whole(
        plan,
        protocol,
        dry_run=True,
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )
    second = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )

    assert {unit.status for unit in preview.units} == {"would-resume"}
    assert second.valid
    assert second.resumed == 4
    assert second.succeeded == 0
    assert (second.units[0].output_dir / "whole_features.csv").stat().st_mtime_ns == first_mtime
    assert list((output_root / "tables" / "whole" / "b32").glob("*.bak-*"))


@pytest.mark.parametrize("jobs", [1, 2])
def test_progress_reports_each_subject_session_after_its_last_unit(tmp_path, jobs):
    _case(tmp_path)
    subject = tmp_path / "data" / "sub-002"
    _write_image(subject / "anat" / "sub-002_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-002_FLAIR_preprocessed.nii.gz")
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", np.s_[2:5, 2:5, 2:5])
    manifest_path = write_manifest_outputs(
        build_manifest_preview(load_project_config(tmp_path / "project.yaml"))
    ).manifest_path
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()
    options = dict(jobs=jobs, software=FAKE_SOFTWARE)
    messages: list[str] = []

    run_whole(
        plan, protocol, dry_run=True, extractor_factory=_factory,
        progress=messages.append, **options,
    )
    assert messages == []

    first = run_whole(
        plan, protocol, extractor_factory=lambda _path: FakeExtractor(fail_mask="sub-002"),
        progress=messages.append, **options,
    )
    second = run_whole(
        plan, protocol, extractor_factory=_factory, progress=messages.append, **options
    )

    assert (first.succeeded, first.failed) == (4, 2)
    assert (second.resumed, second.succeeded, second.failed) == (4, 2, 0)
    start = (
        "Running whole-region extraction (variant b32): 6 unit(s) across "
        f"2 subject/session(s), jobs={jobs}"
    )
    assert messages == [
        start,
        "[1/2] sub-001/ses-01 done: 4 unit(s), 0 skipped, 0 failed",
        "[2/2] sub-002/ses-01 done: 2 unit(s), 0 skipped, 2 failed",
        start,
        "[1/2] sub-001/ses-01 done: 4 unit(s), 0 skipped, 0 failed",
        "[2/2] sub-002/ses-01 done: 2 unit(s), 0 skipped, 0 failed",
    ]


def test_corrupted_unit_blocks_until_force_then_is_archived(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()
    first = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    corrupt_dir = first.units[0].output_dir
    (corrupt_dir / "whole_features.csv").write_text("broken\n", encoding="utf-8")

    preview = run_whole(
        plan, protocol, dry_run=True, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    blocked = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    forced = run_whole(
        plan,
        protocol,
        force=True,
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert sum(unit.status == "blocked" for unit in preview.units) == 1
    assert blocked.failed == 1
    assert forced.valid
    assert forced.succeeded == 4
    assert list(corrupt_dir.parent.glob(f"{corrupt_dir.name}.bak-*"))


def test_same_size_output_edit_is_detected_by_content_hash(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()
    options = dict(extractor_factory=_factory, software=FAKE_SOFTWARE)
    first = run_whole(plan, protocol, **options)
    result_path = first.units[0].output_dir / "whole_features.csv"
    stat = result_path.stat()
    text = result_path.read_text(encoding="utf-8")
    replacement = "9" if text[-3] != "9" else "8"
    result_path.write_text(text[:-3] + replacement + text[-2:], encoding="utf-8")
    os.utime(result_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    preview = run_whole(plan, protocol, dry_run=True, **options)

    assert any(unit.status == "blocked" for unit in preview.units)


def test_touch_and_output_relocation_do_not_invalidate_content(tmp_path):
    manifest_path, output_root = _case(tmp_path)
    protocol = _test_protocol()
    options = dict(extractor_factory=_factory, software=FAKE_SOFTWARE)
    plan = load_whole_plan(manifest_path)
    run_whole(plan, protocol, **options)
    image_path = plan.units[0].image_path
    stat = image_path.stat()
    os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    moved = tmp_path / "moved-output"
    output_root.rename(moved)

    preview = run_whole(
        load_whole_plan(moved / "manifest" / "regions_manifest.csv"),
        protocol,
        dry_run=True,
        **options,
    )

    assert {unit.status for unit in preview.units} == {"would-resume"}


def test_process_pool_task_does_not_pickle_the_cohort_plan(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()
    task = _DefaultExtractionTask(
        plan.units[0],
        _ExtractionPlan(
            plan.output_root,
            plan.manifest_sha256,
            plan.manifest_reference_status,
        ),
        protocol,
        protocol.roi_preset("b32"),
        FAKE_SOFTWARE,
        False,
    )

    assert len(pickle.dumps(task)) < 16 * 1024


def test_one_dimensional_region_is_a_resumable_skip(tmp_path):
    data_root = tmp_path / "data"
    subject = data_root / "sub-001"
    image_values = np.arange(16**3, dtype=np.float32).reshape(16, 16, 16)
    for name in ("sub-001_T1w_preprocessed.nii.gz", "sub-001_FLAIR_preprocessed.nii.gz"):
        path = subject / "anat" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(image_values)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(path))
    mask_values = np.zeros((16, 16, 16), dtype=np.uint8)
    mask_values[8, 8, 2:14] = 1
    mask_path = subject / "lesions" / "lesion_001.nii.gz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = sitk.GetImageFromArray(mask_values)
    mask.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(mask, str(mask_path))
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(tmp_path / "output")
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    preview = build_manifest_preview(load_project_config(project_path))
    manifest_path = write_manifest_outputs(preview).manifest_path

    result = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert preview.valid
    assert any(
        check.check_id == "region_dimensions" and check.status == "WARN"
        for check in preview.checks
    )
    assert (result.skipped, result.failed) == (2, 0)
    for unit in result.units:
        completion = json.loads(unit.completion_path.read_text(encoding="utf-8"))
        assert completion["skip_code"] == "whole_minimum_roi_dimensions"


def test_constant_roi_intensity_is_a_resumable_skip(tmp_path):
    _manifest_path, _output_root = _case(tmp_path)
    t1_path = tmp_path / "data" / "sub-001" / "anat" / "sub-001_T1w_preprocessed.nii.gz"
    constant = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.float32))
    constant.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(constant, str(t1_path))
    manifest_path = write_manifest_outputs(
        build_manifest_preview(load_project_config(tmp_path / "project.yaml"))
    ).manifest_path

    result = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert (result.succeeded, result.skipped, result.failed) == (2, 2, 0)
    skipped = [unit for unit in result.units if unit.status == "skipped"]
    assert {unit.row["modality"] for unit in skipped} == {"T1"}
    assert all(
        json.loads(unit.completion_path.read_text(encoding="utf-8"))["skip_code"]
        == "whole_intensity_range"
        for unit in skipped
    )


def test_failed_units_remain_in_cohort_table_with_error_text(tmp_path):
    manifest_path, _output_root = _case(tmp_path)

    result = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        extractor_factory=lambda _path: FakeExtractor(fail_mask="lesion_002"),
        software=FAKE_SOFTWARE,
    )

    rows = _rows(result.table_path)
    assert not result.valid
    assert result.failed == 2
    assert len(rows) == 4
    failed = [row for row in rows if row["error"]]
    assert len(failed) == 2
    assert all("deliberate extraction failure" in row["error"] for row in failed)
    assert len(_rows(result.errors_path)) == 2


def test_whole_skips_external_masks_below_ten_voxels_and_resumes_skip(tmp_path):
    _manifest_path, _output_root = _case(tmp_path)
    small_mask = tmp_path / "data" / "sub-001" / "lesions" / "lesion_002.nii.gz"
    _write_mask(small_mask, np.s_[6:7, 1:4, 1:4])
    manifest_path = write_manifest_outputs(
        build_manifest_preview(load_project_config(tmp_path / "project.yaml"))
    ).manifest_path
    plan = load_whole_plan(manifest_path)
    protocol = _test_protocol()

    first = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )

    assert first.valid
    assert (first.succeeded, first.skipped, first.failed) == (2, 2, 0)
    skipped = [unit for unit in first.units if unit.status == "skipped"]
    assert len(skipped) == 2
    assert all(unit.row["mask_voxels"] == 9 for unit in skipped)
    assert all(str(unit.row["error"]).startswith("skipped:") for unit in skipped)
    assert all(
        json.loads(unit.completion_path.read_text(encoding="utf-8"))["status"]
        == "skipped"
        for unit in skipped
    )
    assert _rows(first.errors_path) == []
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["skipped"] == 2

    second = run_whole(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    assert (second.resumed, second.skipped, second.failed) == (2, 2, 0)
    assert sum(unit.status == "resumed-skipped" for unit in second.units) == 2


def test_version_mismatch_fails_by_default_and_can_be_stamped(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    mismatched = dict(FAKE_SOFTWARE, numpy="9.9")

    try:
        run_whole(
            load_whole_plan(manifest_path),
            _test_protocol(),
            extractor_factory=_factory,
            software=mismatched,
        )
    except Exception as exc:
        assert "protocol expects 1.26.4" in str(exc)
    else:
        raise AssertionError("version mismatch unexpectedly ran without an override")

    result = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        allow_version_mismatch=True,
        extractor_factory=_factory,
        software=mismatched,
    )
    assert result.valid
    assert not result.reference_compatible
    assert result.reference_deviations == (
        "numpy is 9.9; protocol expects 1.26.4",
    )


def test_whole_summary_writes_feature_eda_and_archives_prior_reports(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    run = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )
    assert run.valid

    summary = summarize_whole(manifest_path, top_features=2)
    assert (summary.total_units, summary.successful_units) == (4, 4)
    assert (summary.skipped_units, summary.failed_units) == (0, 0)
    assert summary.feature_count == 2
    assert summary.markdown_path.is_file()
    assert len(_rows(summary.features_path)) == 4
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert all(report["validations"].values())
    assert report["top_missing_features"] == []
    assert "No missing or non-finite feature values" in summary.markdown_path.read_text(
        encoding="utf-8"
    )

    repeated = summarize_whole(manifest_path, top_features=2)
    assert len(repeated.backups) == 6
    assert all(path.is_file() for path in repeated.backups)


def test_whole_summary_counts_expected_skips(tmp_path):
    _manifest_path, _output_root = _case(tmp_path)
    small_mask = tmp_path / "data" / "sub-001" / "lesions" / "lesion_002.nii.gz"
    _write_mask(small_mask, np.s_[6:7, 1:4, 1:4])
    manifest_path = write_manifest_outputs(
        build_manifest_preview(load_project_config(tmp_path / "project.yaml"))
    ).manifest_path
    run = run_whole(
        load_whole_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )
    assert run.skipped == 2

    summary = summarize_whole(manifest_path)
    assert (summary.successful_units, summary.skipped_units) == (2, 2)
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert all(report["validations"].values())
