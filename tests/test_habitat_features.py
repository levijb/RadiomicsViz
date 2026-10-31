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

from radiomicviz._integrity import file_fingerprint
from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.habitat_features import (
    load_habitat_feature_plan,
    run_habitat_features,
    summarize_habitat_features,
)
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import load_protocol

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
    array = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_binary_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:5, 2:5, 2:5] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _case(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    subject = data_root / "sub-001"
    _write_image(subject / "anat" / "sub-001_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-001_FLAIR_preprocessed.nii.gz")
    _write_binary_mask(subject / "lesions" / "lesion_001.nii.gz")
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(tmp_path / "output")
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    written = write_manifest_outputs(
        build_manifest_preview(load_project_config(project_path))
    )
    _write_habitat_outputs(written.manifest_path, tmp_path / "output")
    return written.manifest_path, tmp_path / "output"


def _write_habitat_outputs(manifest_path: Path, output_root: Path) -> None:
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    region_root = (
        output_root / "subjects" / "sub-001" / "sessions" / "ses-01"
        / "regions" / "lesion_001"
    )
    arrays: dict[str, np.ndarray] = {}
    fixed = np.zeros((8, 8, 8), dtype=np.uint8)
    fixed[2:4, 2:5, 2:5] = 1  # 18
    fixed[4:5, 2:5, 2:5] = 2  # 9, intentionally small but non-empty
    arrays["fixed_k2"] = fixed
    adaptive = np.zeros((8, 8, 8), dtype=np.uint8)
    coords = np.argwhere(np.ones((3, 3, 3), dtype=bool)) + 2
    for index, (z, y, x) in enumerate(coords):
        adaptive[z, y, x] = 1 if index < 8 else (2 if index < 16 else 3)
    arrays["adaptive_k1to3"] = adaptive
    geometric = np.zeros((8, 8, 8), dtype=np.uint8)
    for index, (z, y, x) in enumerate(coords):
        geometric[z, y, x] = 1 if index < 19 else 2
    arrays["geometric_1mm"] = geometric

    source_outputs = {}
    index_rows = []
    zone_names = {
        "fixed_k2": ("peripheral_habitat", "central_habitat"),
        "adaptive_k1to3": (
            "peripheral_habitat", "intermediate_habitat_1", "central_habitat"
        ),
        "geometric_1mm": ("geometric_shell", "geometric_core"),
    }
    for method, array in arrays.items():
        folder = region_root / "habitats" / method
        if method != "geometric_1mm":
            folder /= "b32"
        folder.mkdir(parents=True, exist_ok=True)
        mask_path = folder / "mask_source_grid.nii.gz"
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(mask_path))
        source_outputs[method] = file_fingerprint(mask_path)
        for zone_id, zone_name in enumerate(zone_names[method]):
            label = zone_id + 1
            count = int(np.count_nonzero(array == label))
            habitat_count = count - 1 if method == "fixed_k2" and zone_id == 0 else count
            index_rows.append(
                {
                    "subject_id": "sub-001", "session_id": "ses-01",
                    "region_id": "lesion_001", "region_type": "lesion",
                    "variant": "b32", "implementation_id": "reference",
                    "reference_status": "reference-compatible", "method": method,
                    "mask_path": "unused-cropped-mask", "source_grid_mask_path": str(mask_path),
                    "zone_id": zone_id, "zone_name": zone_name,
                    "n_voxels": habitat_count, "missing": int(habitat_count < 10),
                    "eligible_for_split": True,
                    "chosen_k": "" if method == "geometric_1mm" else len(zone_names[method]),
                    "silhouette_score": "", "habitat_valid": True,
                    "mask_label_value": label,
                }
            )

    completion_dir = region_root / "habitats" / "reference" / "b32"
    completion_dir.mkdir(parents=True, exist_ok=True)
    completion = {
        "schema_version": 2, "stage": "habitats", "status": "complete",
        "unit_key": "sub-001__ses-01__lesion_001",
        "variant": "b32", "manifest_sha256": manifest_sha,
        "source_grid_outputs": source_outputs,
    }
    (completion_dir / "completion.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    index_dir = output_root / "tables" / "habitats" / "b32"
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "habitat_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "habitats_b32_run_metadata.json").write_text(
        json.dumps(
            {
                "stage": "habitats", "status": "complete", "variant": "b32",
                "manifest_sha256": manifest_sha, "reference_compatible": True,
            }
        ),
        encoding="utf-8",
    )


def _test_protocol():
    protocol = load_protocol()
    preset = protocol.roi_preset("b32")
    preset = replace(
        preset,
        expected_feature_count=2,
        expected_shape_feature_count=1,
        expected_diagnostics_count=4,
        expected_feature_schema_sha256=hashlib.sha256(
            "\n".join(FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
    )
    return replace(
        protocol, expected_software=FAKE_SOFTWARE, variants={"b32": preset}
    )


class FakeExtractor:
    def __init__(
        self,
        fail_label: int | None = None,
        fail_below_voxels: int | None = None,
    ):
        self.fail_label = fail_label
        self.fail_below_voxels = fail_below_voxels

    def execute(self, image: str, mask: str, *, label: int):
        if label == self.fail_label:
            raise ValueError("deliberate habitat-zone failure")
        values = sitk.GetArrayFromImage(sitk.ReadImage(mask))
        count = int(np.count_nonzero(values == label))
        if self.fail_below_voxels is not None and count < self.fail_below_voxels:
            raise ValueError("PyRadiomics rejected a deliberately small habitat zone")
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
                (FEATURE_NAMES[0], label / 10),
                (FEATURE_NAMES[1], count + label / 100),
            )
        )


def _factory(_path: Path):
    return FakeExtractor()


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_extracts_every_nonempty_zone_and_writes_reference_tables(tmp_path):
    manifest, output_root = _case(tmp_path)
    plan = load_habitat_feature_plan(manifest)
    assert len(plan.units) == 6
    assert sum(len(unit.zones) for unit in plan.units) == 14

    result = run_habitat_features(
        plan, _test_protocol(), jobs=2,
        extractor_factory=_factory, software=FAKE_SOFTWARE,
    )

    assert result.valid
    assert (result.completed, result.resumed, result.zone_rows) == (6, 0, 14)
    assert result.successful_zones == 14
    assert len(_rows(result.table_path)) == 14
    fixed_t1 = next(
        row for row in _rows(result.table_path)
        if row["method"] == "fixed_k2" and row["zone_id"] == "0"
        and row["modality"] == "T1"
    )
    assert fixed_t1["source_zone_voxels"] == "18"
    assert fixed_t1["habitat_grid_voxels"] == "17"
    assert len(result.method_table_paths) == 3
    assert len(result.reference_table_paths) == 6
    for path in result.reference_table_paths.values():
        with path.open(encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        assert fields[:10] == [
            "ID", "Exam", "Label", "Rim", "Image", "Mask", "Cluster",
            "NumofVoxels", "Patient", "Reader",
        ]
        assert len(fields) == 16
    assert any(
        row["check_id"] == "habitat_feature_small_zone" and row["status"] == "WARN"
        for row in _rows(result.checks_path)
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_schema_sha256"] == _test_protocol().roi_preset("b32").expected_feature_schema_sha256
    assert metadata["zone_rows"] == 14
    for unit in result.units:
        assert (unit.output_dir / "features.csv").is_file()
        assert (unit.output_dir / "completion.json").is_file()
        assert unit.output_dir.is_relative_to(output_root)


def test_rerun_resumes_and_output_tampering_blocks_one_unit(tmp_path):
    manifest, _ = _case(tmp_path)
    plan = load_habitat_feature_plan(manifest)
    protocol = _test_protocol()
    first = run_habitat_features(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    second = run_habitat_features(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    assert second.resumed == 6
    assert second.valid
    assert all(unit.diagnostic_names == (
        "diagnostics_Versions_PyRadiomics",
        "diagnostics_Configuration_Settings",
        "diagnostics_Mask-original_VoxelNum",
        "diagnostics_Mask-interpolated_VoxelNum",
    ) for unit in second.units)
    for path in second.reference_table_paths.values():
        with path.open(encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        assert len(fields) == 16
        assert len(fields) == len(set(fields))

    path = first.units[0].output_dir / "features.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace("0.1", "0.9", 1), encoding="utf-8")
    blocked = run_habitat_features(
        plan, protocol, extractor_factory=_factory, software=FAKE_SOFTWARE
    )
    assert blocked.blocked == 1
    assert not blocked.valid


def test_failed_small_zone_is_retained_and_resumable(tmp_path):
    manifest, _ = _case(tmp_path)
    plan = load_habitat_feature_plan(manifest)
    protocol = _test_protocol()
    failing_factory = lambda _path: FakeExtractor(fail_label=3)

    first = run_habitat_features(
        plan, protocol, extractor_factory=failing_factory, software=FAKE_SOFTWARE
    )
    assert not first.valid
    assert first.failed == 2
    rows = _rows(first.table_path)
    assert len(rows) == 14
    assert sum(bool(row["error"]) for row in rows) == 2
    assert {row["modality"] for row in rows if row["error"]} == {"T1", "FLAIR"}

    second = run_habitat_features(
        plan, protocol, extractor_factory=failing_factory, software=FAKE_SOFTWARE
    )
    assert second.resumed == 6
    assert second.failed == 2
    assert len(_rows(second.table_path)) == 14


def test_expected_small_zone_rejection_is_warning_and_resumable(tmp_path):
    manifest, _ = _case(tmp_path)
    plan = load_habitat_feature_plan(manifest)
    protocol = _test_protocol()
    small_zone_factory = lambda _path: FakeExtractor(fail_below_voxels=10)

    first = run_habitat_features(
        plan, protocol, extractor_factory=small_zone_factory, software=FAKE_SOFTWARE
    )
    assert first.valid
    assert first.failed == 0
    assert first.skipped_zones == 8
    assert first.successful_zones == 6
    rows = _rows(first.table_path)
    assert len(rows) == 14
    assert sum(row["zone_status"] == "skipped" for row in rows) == 8
    assert not _rows(first.errors_path)
    assert any(
        row["check_id"] == "habitat_feature_expected_skip"
        and row["status"] == "WARN"
        for row in _rows(first.checks_path)
    )

    second = run_habitat_features(
        plan, protocol, extractor_factory=small_zone_factory, software=FAKE_SOFTWARE
    )
    assert second.resumed == 6
    assert second.valid
    assert second.skipped_zones == 8
    assert second.failed == 0


def test_summary_reports_status_and_archives_previous_summary(tmp_path):
    manifest, _ = _case(tmp_path)
    plan = load_habitat_feature_plan(manifest)
    run_habitat_features(
        plan, _test_protocol(), extractor_factory=_factory, software=FAKE_SOFTWARE
    )

    first = summarize_habitat_features(manifest, top_features=2)
    assert first.total_rows == 14
    assert first.successful_rows == 14
    assert first.skipped_rows == 0
    assert first.failed_rows == 0
    assert first.feature_count == 2
    assert first.markdown_path.is_file()
    assert "Primary results:" in first.markdown_path.read_text(encoding="utf-8")

    second = summarize_habitat_features(manifest, top_features=2)
    assert len(second.backups) == 6
    assert all(path.is_file() for path in second.backups)
