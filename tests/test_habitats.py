from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pytest
import yaml

from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.habitats import (
    HabitatStageError,
    load_habitat_plan,
    run_habitats,
    summarize_habitats,
)
from radiomicviz.habitats.reference import (
    METHOD_ADAPTIVE,
    METHOD_FIXED,
    METHOD_GEOMETRIC,
    prepare_lesion_feature_matrix,
    process_region,
)
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import load_protocol
from radiomicviz.voxelwise import load_voxelwise_plan, run_voxelwise

MAP_NAMES = (
    "original_firstorder_Mean",
    "original_firstorder_InterquartileRange",
    "original_firstorder_TotalEnergy",
    "original_glcm_JointAverage",
)
BASE_SOFTWARE = {
    "python": "3.10.10",
    "pyradiomics": "3.0.1",
    "numpy": "1.26.4",
    "simpleitk": "2.5.5",
    "pywavelets": "1.8.0",
}
HABITAT_SOFTWARE = {
    **BASE_SOFTWARE,
    "scipy": "1.11.4",
    "scikit-learn": "1.1.3",
}


def _write_image(path: Path, reverse: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(9**3, dtype=np.float32).reshape(9, 9, 9)
    if reverse:
        values = values.max() - values
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_mask(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((9, 9, 9), dtype=np.uint8)
    coordinates = np.argwhere(np.ones((4, 4, 4), dtype=bool))[:count] + 2
    array[tuple(coordinates.T)] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _case(tmp_path: Path, count: int = 64) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    subject = data_root / "sub-001"
    _write_image(subject / "anat" / "sub-001_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-001_FLAIR_preprocessed.nii.gz", reverse=True)
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", count)
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(tmp_path / "output")
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    written = write_manifest_outputs(
        build_manifest_preview(load_project_config(project_path))
    )
    return written.manifest_path, tmp_path / "output"


def _protocol():
    protocol = load_protocol()
    preset = protocol.voxelwise_preset("b32")
    test_preset = replace(
        preset,
        expected_map_count=len(MAP_NAMES),
        expected_map_names=MAP_NAMES,
        expected_map_schema_sha256=hashlib.sha256(
            "\n".join(MAP_NAMES).encode()
        ).hexdigest(),
    )
    return replace(
        protocol,
        expected_software=BASE_SOFTWARE,
        voxelwise_variants={"b32": test_preset},
    )


class FakeExtractor:
    def execute(self, image: str, mask: str, *, label: int, voxelBased: bool):
        assert voxelBased is True
        source = sitk.ReadImage(image)
        mask_image = sitk.ReadImage(mask)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_image)
        bounding_box = stats.GetBoundingBox(label)
        start, size = bounding_box[:3], bounding_box[3:]
        lower = tuple(max(0, start[axis] - 1) for axis in range(3))
        upper = tuple(
            min(source.GetSize()[axis], start[axis] + size[axis] + 1)
            for axis in range(3)
        )
        crop_size = tuple(upper[axis] - lower[axis] for axis in range(3))
        source_crop = sitk.RegionOfInterest(source, crop_size, lower)
        mask_crop = sitk.RegionOfInterest(mask_image, crop_size, lower)
        foreground = sitk.GetArrayFromImage(mask_crop) == label
        source_values = sitk.GetArrayFromImage(source_crop).astype(np.float64)

        def feature(values: np.ndarray) -> sitk.Image:
            result_values = np.full(values.shape, np.nan, dtype=np.float64)
            result_values[foreground] = values[foreground]
            result = sitk.GetImageFromArray(result_values)
            result.CopyInformation(source_crop)
            return result

        all_nan = np.full(source_values.shape, np.nan, dtype=np.float64)
        return OrderedDict(
            [
                (
                    "diagnostics_Configuration_Settings",
                    {
                        "binCount": 32,
                        "normalize": False,
                        "label": label,
                        "resampledPixelSpacing": None,
                    },
                ),
                (MAP_NAMES[0], feature(source_values)),
                (MAP_NAMES[1], feature(np.sqrt(source_values + 1.0))),
                (MAP_NAMES[2], feature(source_values**2)),
                (MAP_NAMES[3], feature(all_nan)),
            ]
        )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_voxelwise(manifest: Path, protocol) -> None:
    result = run_voxelwise(
        load_voxelwise_plan(manifest),
        protocol,
        extractor_factory=lambda _path: FakeExtractor(),
        software=BASE_SOFTWARE,
    )
    assert result.valid


def test_reference_core_is_lesion_local_and_writes_seven_rows() -> None:
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    x = np.arange(64, dtype=float)
    raw = np.column_stack((x, x, x[::-1], np.sqrt(x + 1)))
    names = ["T1_a", "T1_duplicate", "FLAIR_a", "FLAIR_b"]
    features, info = prepare_lesion_feature_matrix(raw, names, mask)
    assert features.shape[0] == 64
    assert info["n_features_selected_t1"] >= 1
    assert info["n_features_selected_flair"] >= 1
    result = process_region(raw, names, mask, (1.0, 1.0, 1.0))
    assert len(result.rows) == 7
    assert set(result.masks) == {METHOD_FIXED, METHOD_ADAPTIVE, METHOD_GEOMETRIC}
    for volume in result.masks.values():
        assert volume.dtype == np.uint8
        assert np.count_nonzero(volume) == 64


def test_habitats_run_all_methods_resume_and_write_cohort_tables(tmp_path: Path) -> None:
    manifest, output_root = _case(tmp_path)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    first = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert first.valid and first.succeeded == 1
    assert len(_rows(first.table_path)) == 7
    unit = first.units[0]
    assert unit.qc["representation_status"] == "computed"
    completion = json.loads(unit.completion_path.read_text(encoding="utf-8"))
    assert completion["reference_status"] == "reference-compatible"
    source_size = sitk.ReadImage(
        str(load_habitat_plan(manifest).units[0].mask_path)
    ).GetSize()
    expected_order = [
        f"{modality}_{name.split('original_', 1)[1]}"
        for modality in ("T1", "FLAIR")
        for name in sorted(MAP_NAMES)
        if "firstorder_TotalEnergy" not in name
    ]
    assert completion["feature_names"] == expected_order
    for method, path in {
        METHOD_FIXED: unit.output_dir / METHOD_FIXED / "b32" / "mask.nii.gz",
        METHOD_ADAPTIVE: unit.output_dir / METHOD_ADAPTIVE / "b32" / "mask.nii.gz",
        METHOD_GEOMETRIC: unit.output_dir / METHOD_GEOMETRIC / "mask.nii.gz",
    }.items():
        values = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        assert np.count_nonzero(values) == 64, method
        source_grid = path.with_name("mask_source_grid.nii.gz")
        assert source_grid.is_file()
        assert sitk.ReadImage(str(source_grid)).GetSize() == source_size
        assert np.count_nonzero(
            sitk.GetArrayFromImage(sitk.ReadImage(str(source_grid)))
        ) == 64

    second = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert second.valid and second.resumed == 1
    assert len(_rows(second.table_path)) == 7


def test_habitat_dry_run_blocks_before_missing_voxelwise_inputs(tmp_path: Path) -> None:
    manifest, _output_root = _case(tmp_path)
    result = run_habitats(
        load_habitat_plan(manifest),
        _protocol(),
        software=HABITAT_SOFTWARE,
        dry_run=True,
    )

    assert not result.valid
    assert result.blocked == 1
    assert result.units[0].status == "blocked"
    assert "run 'radiomicviz voxelwise'" in result.units[0].message
    assert result.failures[0].error_code == "habitat_voxelwise_input_missing"

def test_habitats_skip_when_both_voxelwise_units_were_too_small(tmp_path: Path) -> None:
    manifest, _output_root = _case(tmp_path, count=10)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    result = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert result.valid and result.skipped == 1
    assert _rows(result.table_path) == []
    assert result.warnings == 1

def test_habitats_reference_fallback_for_eleven_to_forty_nine_voxels(
    tmp_path: Path,
) -> None:
    manifest, _output_root = _case(tmp_path, count=20)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    result = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert result.valid and result.succeeded == 1
    assert len(result.units[0].rows) == 7
    assert result.units[0].qc["representation_status"] == (
        "not_computed_below_min_lesion_voxels"
    )
    radiomic_rows = [
        row
        for row in result.units[0].rows
        if row["method"] in {METHOD_FIXED, METHOD_ADAPTIVE}
    ]
    assert {row["chosen_k"] for row in radiomic_rows} == {1}


def test_habitat_corruption_blocks_until_force_archives_and_recomputes(
    tmp_path: Path,
) -> None:
    manifest, _output_root = _case(tmp_path)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    first = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    fixed = first.units[0].output_dir / METHOD_FIXED / "b32" / "mask.nii.gz"
    fixed.write_bytes(b"deliberate corruption")

    blocked = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert not blocked.valid and blocked.blocked == 1

    replaced = run_habitats(
        load_habitat_plan(manifest),
        protocol,
        software=HABITAT_SOFTWARE,
        force=True,
    )
    assert replaced.valid and replaced.succeeded == 1
    assert sitk.ReadImage(str(fixed)).GetPixelID() == sitk.sitkUInt8
    assert list((fixed.parent.parent).glob("b32.bak-*"))


def test_upstream_voxelwise_version_deviation_requires_explicit_override(
    tmp_path: Path,
) -> None:
    manifest, output_root = _case(tmp_path, count=20)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    completion = next(
        output_root.glob(
            "subjects/*/sessions/*/regions/*/radiomics/voxelwise/b32/T1/completion.json"
        )
    )
    data = json.loads(completion.read_text(encoding="utf-8"))
    data["software"]["numpy"] = "9.9.9"
    completion.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(HabitatStageError, match="upstream voxelwise unit"):
        run_habitats(
            load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
        )
    allowed = run_habitats(
        load_habitat_plan(manifest),
        protocol,
        software=HABITAT_SOFTWARE,
        allow_version_mismatch=True,
    )
    assert allowed.valid and not allowed.reference_compatible


def test_habitat_summary_writes_reports_and_archives_previous_reports(
    tmp_path: Path,
) -> None:
    manifest, _output_root = _case(tmp_path)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    run = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert run.valid

    summary = summarize_habitats(manifest, top_features=3)
    assert summary.total_regions == 1
    assert summary.completed_regions == 1
    assert summary.skipped_regions == 0
    assert summary.failed_units == 0
    assert summary.source_run_runtime_seconds is not None
    assert summary.markdown_path.is_file()
    assert summary.features_path.is_file()
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert report["cohort"]["habitat_index_rows"] == 7
    assert all(report["validations"].values())
    assert len(report["top_selected_features"]) <= 3

    repeated = summarize_habitats(manifest, top_features=3)
    assert len(repeated.backups) == 6
    assert all(path.is_file() for path in repeated.backups)


def test_habitat_summary_supports_an_all_skipped_run(tmp_path: Path) -> None:
    manifest, _output_root = _case(tmp_path, count=10)
    protocol = _protocol()
    _prepare_voxelwise(manifest, protocol)
    run = run_habitats(
        load_habitat_plan(manifest), protocol, software=HABITAT_SOFTWARE
    )
    assert run.valid and run.skipped == 1

    summary = summarize_habitats(manifest)
    assert summary.total_regions == 1
    assert summary.completed_regions == 0
    assert summary.skipped_regions == 1
    report = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert report["cohort"]["habitat_index_rows"] == 0
    assert all(report["validations"].values())
