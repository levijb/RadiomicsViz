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
from radiomicviz.whole import load_whole_plan, run_whole

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
    _write_mask(subject / "lesions" / "lesion_002.nii.gz", np.s_[4:6, 4:6, 4:6])
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
