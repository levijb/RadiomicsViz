from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import load_protocol
from radiomicviz.voxelwise import (
    load_voxelwise_plan,
    run_voxelwise,
    summarize_voxelwise,
)

JOINT_AVERAGE = "original_glcm_JointAverage"
MAP_NAMES = (
    "original_firstorder_Mean",
    JOINT_AVERAGE,
    "original_glcm_Contrast",
)
FAKE_SOFTWARE = {
    "python": "3.10.10",
    "pyradiomics": "3.0.1",
    "numpy": "1.26.4",
    "simpleitk": "2.5.5",
    "pywavelets": "1.8.0",
}


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((9, 9, 9), dtype=np.float32)
    array[1:8, 1:8, 1:8] = np.arange(1, 344, dtype=np.float32).reshape(7, 7, 7)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_mask(path: Path, block) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((9, 9, 9), dtype=np.uint8)
    array[block] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_count_mask(path: Path, voxel_count: int) -> None:
    """Write a compact 3-D mask with exactly 1-12 foreground voxels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((9, 9, 9), dtype=np.uint8)
    coordinates = np.argwhere(np.ones((3, 2, 2), dtype=bool))[:voxel_count]
    coordinates += np.asarray((4, 4, 4))
    array[tuple(coordinates.T)] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _case(
    tmp_path: Path,
    *,
    tiny_second: bool = False,
    second_voxel_count: int | None = None,
):
    data_root = tmp_path / "data"
    subject = data_root / "sub-001"
    _write_image(subject / "anat" / "sub-001_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-001_FLAIR_preprocessed.nii.gz")
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", np.s_[2:5, 2:5, 2:5])
    second_path = subject / "lesions" / "lesion_002.nii.gz"
    if second_voxel_count is not None:
        _write_count_mask(second_path, second_voxel_count)
    else:
        second = np.s_[6:7, 6:7, 6:7] if tiny_second else np.s_[4:7, 4:6, 4:6]
        _write_mask(second_path, second)
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
    preset = protocol.voxelwise_preset("b32")
    test_preset = replace(
        preset,
        expected_map_count=len(MAP_NAMES),
        expected_map_names=MAP_NAMES,
        expected_map_schema_sha256=hashlib.sha256(
            "\n".join(MAP_NAMES).encode("utf-8")
        ).hexdigest(),
    )
    return replace(
        protocol,
        expected_software=FAKE_SOFTWARE,
        voxelwise_variants={"b32": test_preset},
    )


class FakeExtractor:
    def __init__(
        self,
        *,
        fail_mask: str | None = None,
        all_nan_map: str | None = None,
    ):
        self.fail_mask = fail_mask
        self.all_nan_map = all_nan_map

    def execute(
        self, image: str, mask: str, *, label: int, voxelBased: bool
    ):
        assert voxelBased is True
        if self.fail_mask and self.fail_mask in mask:
            raise ValueError("deliberate voxelwise failure")
        source = sitk.ReadImage(image)
        mask_image = sitk.ReadImage(mask)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_image)
        bounding_box = stats.GetBoundingBox(label)
        start = bounding_box[:3]
        size = bounding_box[3:]
        lower = tuple(max(0, start[axis] - 1) for axis in range(3))
        upper = tuple(
            min(source.GetSize()[axis], start[axis] + size[axis] + 1)
            for axis in range(3)
        )
        crop_size = tuple(upper[axis] - lower[axis] for axis in range(3))
        source_crop = sitk.RegionOfInterest(source, crop_size, lower)
        mask_crop = sitk.RegionOfInterest(mask_image, crop_size, lower)
        foreground = sitk.GetArrayFromImage(mask_crop) == label

        def feature_map(value: float | None) -> sitk.Image:
            array = np.full(foreground.shape, np.nan, dtype=np.float64)
            if value is not None:
                array[foreground] = value
            result = sitk.GetImageFromArray(array)
            result.CopyInformation(source_crop)
            return result

        result = OrderedDict(
            [
                (
                    "diagnostics_Configuration_Settings",
                    {
                        "binCount": 32,
                        "normalize": False,
                        "label": label,
                        "resampledPixelSpacing": None,
                    },
                )
            ]
        )
        for index, name in enumerate(MAP_NAMES, start=1):
            value = None if name == self.all_nan_map else index + 0.5
            result[name] = feature_map(value)
        return result


def _factory(_parameter_path: Path):
    return FakeExtractor()


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_voxelwise_writes_valid_maps_for_every_region_and_modality(tmp_path):
    manifest_path, output_root = _case(tmp_path)

    result = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        jobs=2,
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert result.valid
    assert (result.succeeded, result.skipped, result.failed) == (4, 0, 0)
    assert _rows(result.errors_path) == []
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["completed"] == 4
    for unit in result.units:
        assert unit.output_dir.is_relative_to(output_root)
        assert (unit.output_dir / "completion.json").is_file()
        assert {path.name for path in unit.output_dir.glob("*.nrrd")} == {
            f"{name}.nrrd" for name in MAP_NAMES
        }
        for path in unit.output_dir.glob("*.nrrd"):
            assert sitk.ReadImage(str(path)).GetPixelID() == sitk.sitkFloat64


def test_tiny_regions_are_recorded_skips_not_failures(tmp_path):
    manifest_path, _output_root = _case(tmp_path, tiny_second=True)

    result = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )

    assert result.valid
    assert (result.succeeded, result.skipped, result.failed) == (2, 2, 0)
    assert _rows(result.errors_path) == []
    skipped = [unit for unit in result.units if unit.status == "skipped"]
    assert all(unit.mask_voxels == 1 for unit in skipped)
    assert all(not list(unit.output_dir.glob("*.nrrd")) for unit in skipped)
    assert all(
        json.loads((unit.output_dir / "completion.json").read_text())["status"]
        == "skipped"
        for unit in skipped
    )
    assert sum(row["status"] == "WARN" for row in _rows(result.checks_path)) == 2


def test_voxelwise_minimum_roi_size_is_strict_and_skips_resume(tmp_path):
    ten_manifest, _ = _case(tmp_path / "ten", second_voxel_count=10)
    ten_plan = load_voxelwise_plan(ten_manifest)
    protocol = _test_protocol()
    options = dict(extractor_factory=_factory, software=FAKE_SOFTWARE)

    first = run_voxelwise(ten_plan, protocol, **options)
    resumed = run_voxelwise(ten_plan, protocol, **options)

    assert first.valid
    assert (first.succeeded, first.skipped, first.failed) == (2, 2, 0)
    exact_ten = [unit for unit in first.units if unit.mask_voxels == 10]
    assert len(exact_ten) == 2
    assert all(unit.status == "skipped" for unit in exact_ten)
    assert all("strictly more than minimumROISize=10" in unit.message for unit in exact_ten)
    assert resumed.resumed == 2
    assert resumed.skipped == 2
    assert sum(unit.status == "resumed-skipped" for unit in resumed.units) == 2

    eleven_manifest, _ = _case(tmp_path / "eleven", second_voxel_count=11)
    eleven = run_voxelwise(
        load_voxelwise_plan(eleven_manifest),
        protocol,
        **options,
    )
    assert eleven.valid
    assert (eleven.succeeded, eleven.skipped, eleven.failed) == (4, 0, 0)


def test_all_nan_joint_average_completes_with_targeted_warning(tmp_path):
    manifest_path, _ = _case(tmp_path)
    plan = load_voxelwise_plan(manifest_path)
    protocol = _test_protocol()
    factory = lambda _path: FakeExtractor(all_nan_map=JOINT_AVERAGE)

    result = run_voxelwise(
        plan,
        protocol,
        extractor_factory=factory,
        software=FAKE_SOFTWARE,
    )

    assert result.valid
    assert (result.succeeded, result.skipped, result.failed) == (4, 0, 0)
    warnings = [
        row
        for row in _rows(result.checks_path)
        if row["check_id"] == "voxelwise_map_empty"
    ]
    assert len(warnings) == 4
    assert all(row["status"] == "WARN" for row in warnings)
    assert all(JOINT_AVERAGE in row["message"] for row in warnings)
    for unit in result.units:
        completion = json.loads(unit.completion_path.read_text(encoding="utf-8"))
        assert completion["map_finite_voxels"][JOINT_AVERAGE] == 0
        assert set(completion["map_finite_voxels"]) == set(MAP_NAMES)
        assert (unit.output_dir / f"{JOINT_AVERAGE}.nrrd").is_file()

    summary = summarize_voxelwise(manifest_path)
    coverage = [
        row for row in _rows(summary.maps_path) if row["map"] == JOINT_AVERAGE
    ]
    assert len(coverage) == 2
    assert all(row["zero_finite_units"] == "2" for row in coverage)
    assert all(float(row["mean_finite_fraction"]) == 0.0 for row in coverage)

    resumed = run_voxelwise(
        plan,
        protocol,
        extractor_factory=factory,
        software=FAKE_SOFTWARE,
    )
    assert resumed.valid and resumed.resumed == 4
    resumed_warnings = [
        row
        for row in _rows(resumed.checks_path)
        if row["check_id"] == "voxelwise_map_empty"
    ]
    assert len(resumed_warnings) == 4


def test_all_nan_unrecognized_map_remains_a_failure(tmp_path):
    manifest_path, _ = _case(tmp_path)

    result = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        extractor_factory=lambda _path: FakeExtractor(all_nan_map=MAP_NAMES[0]),
        software=FAKE_SOFTWARE,
    )

    assert not result.valid
    assert (result.succeeded, result.failed) == (0, 4)
    assert all("contains no finite voxels" in row["message"] for row in _rows(result.errors_path))


def test_voxelwise_resumes_and_corruption_requires_force(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    plan = load_voxelwise_plan(manifest_path)
    protocol = _test_protocol()
    options = dict(extractor_factory=_factory, software=FAKE_SOFTWARE)
    first = run_voxelwise(plan, protocol, **options)
    for unit in first.units:
        completion = json.loads(unit.completion_path.read_text(encoding="utf-8"))
        completion.pop("map_finite_voxels")
        unit.completion_path.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    resumed = run_voxelwise(plan, protocol, **options)
    corrupt_dir = first.units[0].output_dir
    (corrupt_dir / f"{MAP_NAMES[0]}.nrrd").write_bytes(b"broken")

    preview = run_voxelwise(plan, protocol, dry_run=True, **options)
    blocked = run_voxelwise(plan, protocol, **options)
    forced = run_voxelwise(plan, protocol, force=True, **options)

    assert resumed.resumed == 4
    assert sum(unit.status == "blocked" for unit in preview.units) == 1
    assert blocked.failed == 1
    assert forced.valid and forced.succeeded == 4
    assert list(corrupt_dir.parent.glob(f"{corrupt_dir.name}.bak-*"))


def test_voxelwise_failed_units_are_visible_in_errors(tmp_path):
    manifest_path, _output_root = _case(tmp_path)

    result = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        extractor_factory=lambda _path: FakeExtractor(fail_mask="lesion_002"),
        software=FAKE_SOFTWARE,
    )

    assert not result.valid
    assert (result.succeeded, result.failed) == (2, 2)
    rows = _rows(result.errors_path)
    assert len(rows) == 2
    assert all("deliberate voxelwise failure" in row["message"] for row in rows)


def test_voxelwise_summary_writes_map_coverage_and_archives_prior_reports(tmp_path):
    manifest_path, _output_root = _case(tmp_path)
    run = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )
    assert run.valid

    summary = summarize_voxelwise(manifest_path, top_maps=2)
    assert (summary.total_units, summary.successful_units) == (4, 4)
    assert (summary.skipped_units, summary.failed_units) == (0, 0)
    assert summary.map_count == 3
    assert summary.markdown_path.is_file()
    assert len(_rows(summary.maps_path)) == 6
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert all(report["validations"].values())
    assert len(report["lowest_coverage_maps"]) == 2

    repeated = summarize_voxelwise(manifest_path, top_maps=2)
    assert len(repeated.backups) == 6
    assert all(path.is_file() for path in repeated.backups)


def test_voxelwise_summary_counts_expected_skips(tmp_path):
    manifest_path, _output_root = _case(tmp_path, tiny_second=True)
    run = run_voxelwise(
        load_voxelwise_plan(manifest_path),
        _test_protocol(),
        extractor_factory=_factory,
        software=FAKE_SOFTWARE,
    )
    assert run.skipped == 2

    summary = summarize_voxelwise(manifest_path)
    assert (summary.successful_units, summary.skipped_units) == (2, 2)
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert all(report["validations"].values())
