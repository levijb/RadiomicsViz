import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml
from click.testing import CliRunner

from radiomicviz.cli import main
from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.lesions import (
    MaterializationError,
    build_materialization_plan,
    run_materialization,
)
from radiomicviz.manifest import build_manifest_preview


def _write_image(path: Path, array: np.ndarray, *, spacing=(1.0, 1.0, 1.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((2.0, 3.0, 4.0))
    sitk.WriteImage(image, str(path))


def _project(tmp_path: Path, *, mode: str, connectivity=None):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    source = data_root / "sub-001" / "lesions" / "combined.nii.gz"

    data = yaml.safe_load(template_text("project"))
    data["data_root"] = str(data_root)
    data["output_root"] = str(output_root)
    data["regions"].update(
        mode=mode,
        mask_pattern="{subject}/lesions/combined.nii.gz",
        id_regex=None,
        label=None if mode == "labeled_map" else 1,
        connectivity=connectivity,
    )
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return project_path, source, data_root, output_root


def _labeled_array():
    array = np.zeros((8, 8, 8), dtype=np.uint16)
    array[1:3, 1:4, 1:3] = 1
    array[5:7, 4:7, 4:7] = 4
    return array


def test_labeled_map_dry_run_then_materialize_and_resume(tmp_path):
    project_path, source, _data_root, output_root = _project(
        tmp_path, mode="labeled_map"
    )
    _write_image(source, _labeled_array())
    project = load_project_config(project_path)

    plan = build_materialization_plan(project)
    assert plan.units[0].status == "run"
    assert [region.region_id for region in plan.units[0].regions] == [
        "lesion_001",
        "lesion_004",
    ]
    dry = run_materialization(plan, dry_run=True)
    assert dry.valid
    assert not output_root.exists()

    result = run_materialization(plan)
    assert result.valid and result.completed == 1
    root = output_root / "materialized_regions" / "sub-001" / "ses-01" / "lesion"
    mask1 = root / "lesion_001" / "mask.nii.gz"
    mask4 = root / "lesion_004" / "mask.nii.gz"
    assert mask1.is_file() and mask4.is_file()
    assert int(np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(str(mask1))))) == 12
    assert int(np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(str(mask4))))) == 18

    marker = json.loads(
        (root / ".materialization" / "complete.json").read_text(encoding="utf-8")
    )
    assert marker["mode"] == "labeled_map"
    assert [record["source_label"] for record in marker["regions"]] == [1, 4]
    assert all(len(record["mask_sha256"]) == 64 for record in marker["regions"])

    resume_plan = build_materialization_plan(project)
    assert resume_plan.units[0].status == "resume"
    resumed = run_materialization(resume_plan)
    assert resumed.valid and resumed.resumed == 1 and resumed.completed == 0


@pytest.mark.parametrize("connectivity, expected", [(6, 2), (26, 1)])
def test_connected_component_connectivity_is_explicit(tmp_path, connectivity, expected):
    project_path, source, _data_root, _output_root = _project(
        tmp_path, mode="connected_components", connectivity=connectivity
    )
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[1:3, 1:3, 1:4] = 1
    array[3:5, 3:5, 4:7] = 1
    _write_image(source, array)

    plan = build_materialization_plan(load_project_config(project_path))

    assert len(plan.units[0].regions) == expected
    result = run_materialization(plan)
    assert result.valid
    assert result.units[0].n_regions == expected


def test_connected_components_are_numbered_by_descending_voxel_count(tmp_path):
    project_path, source, _data_root, output_root = _project(
        tmp_path, mode="connected_components", connectivity=6
    )
    array = np.zeros((12, 12, 12), dtype=np.uint8)
    array[1, 1, 1] = 1
    array[4:6, 4:6, 4:7] = 1
    array[8, 3:5, 1:6] = 1
    _write_image(source, array)

    plan = build_materialization_plan(load_project_config(project_path))
    regions = plan.units[0].regions
    assert [region.region_id for region in regions] == ["lesion_001", "lesion_002"]
    assert [region.n_voxels for region in regions] == [12, 10]
    assert [region.component_id for region in regions] == [2, 3]
    assert [region.n_voxels for region in plan.units[0].excluded_regions] == [1]

    assert run_materialization(plan).valid
    root = output_root / "materialized_regions" / "sub-001" / "ses-01" / "lesion"
    largest = sitk.GetArrayFromImage(
        sitk.ReadImage(str(root / "lesion_001" / "mask.nii.gz"))
    )
    assert int(np.count_nonzero(largest)) == 12

    marker = json.loads(
        (root / ".materialization" / "complete.json").read_text(encoding="utf-8")
    )
    assert marker["schema_version"] == 3
    assert marker["minimum_region_voxels"] == 10
    assert [record["n_voxels"] for record in marker["regions"]] == [12, 10]
    assert [record["component_id"] for record in marker["regions"]] == [2, 3]
    assert marker["n_regions_excluded_below_minimum"] == 1
    assert marker["excluded_regions"][0]["n_voxels"] == 1


def test_materialization_includes_ten_voxels_and_excludes_nine(tmp_path):
    project_path, source, _data_root, output_root = _project(
        tmp_path, mode="labeled_map"
    )
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[1, 1:4, 1:4] = 1
    array[4:6, 4, 1:6] = 2
    _write_image(source, array)

    plan = build_materialization_plan(load_project_config(project_path))

    assert [region.source_label for region in plan.units[0].regions] == [2]
    assert [region.n_voxels for region in plan.units[0].regions] == [10]
    assert [region.source_label for region in plan.units[0].excluded_regions] == [1]
    assert [region.n_voxels for region in plan.units[0].excluded_regions] == [9]
    result = run_materialization(plan)
    assert result.valid
    assert result.units[0].n_regions == 1
    assert result.units[0].n_excluded_regions == 1
    root = output_root / "materialized_regions" / "sub-001" / "ses-01" / "lesion"
    assert not (root / "lesion_001").exists()
    assert (root / "lesion_002" / "mask.nii.gz").is_file()


def test_connected_components_rejects_nonbinary_source(tmp_path):
    project_path, source, _data_root, _output_root = _project(
        tmp_path, mode="connected_components", connectivity=26
    )
    array = np.zeros((5, 5, 5), dtype=np.uint8)
    array[1, 1, 1] = 2
    _write_image(source, array)

    with pytest.raises(MaterializationError, match="values other than background 0"):
        build_materialization_plan(load_project_config(project_path))


def test_changed_source_blocks_until_force_and_archives_previous_tree(tmp_path):
    project_path, source, _data_root, output_root = _project(
        tmp_path, mode="labeled_map"
    )
    _write_image(source, _labeled_array())
    project = load_project_config(project_path)
    assert run_materialization(build_materialization_plan(project)).valid

    changed = _labeled_array()
    changed[3, 3, 3] = 7
    _write_image(source, changed)
    blocked_plan = build_materialization_plan(project)
    assert blocked_plan.units[0].status == "blocked"
    blocked = run_materialization(blocked_plan)
    assert not blocked.valid and blocked.blocked == 1

    forced = run_materialization(blocked_plan, force=True)
    assert forced.valid and forced.completed == 1
    parent = output_root / "materialized_regions" / "sub-001" / "ses-01"
    assert len(list(parent.glob("lesion.bak-*"))) == 1
    assert not (parent / "lesion" / "lesion_007").exists()
    marker = json.loads(
        (parent / "lesion" / ".materialization" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["excluded_regions"] == [
        {
            "component_id": None,
            "n_voxels": 1,
            "region_id": "lesion_007",
            "source_label": 7,
        }
    ]


def test_manifest_requires_current_materialization_then_uses_binary_outputs(tmp_path):
    project_path, source, data_root, _output_root = _project(
        tmp_path, mode="labeled_map"
    )
    array = _labeled_array()
    _write_image(source, array)
    subject = data_root / "sub-001" / "anat"
    _write_image(subject / "sub-001_T1w_preprocessed.nii.gz", array.astype(np.float32))
    _write_image(subject / "sub-001_FLAIR_preprocessed.nii.gz", array.astype(np.float32))
    project = load_project_config(project_path)

    before = build_manifest_preview(project)
    assert not before.valid
    assert {error.error_code for error in before.errors} == {"lesion_materialization"}

    assert run_materialization(build_materialization_plan(project)).valid
    after = build_manifest_preview(project)
    assert after.valid
    assert after.regions_found == 2
    assert len(after.rows) == 4
    assert {row.region_id for row in after.rows} == {"lesion_001", "lesion_004"}
    assert all(row.label == 1 for row in after.rows)
    assert all("materialized_regions" in row.mask_path for row in after.rows)

    changed = _labeled_array()
    changed[3, 3, 3] = 7
    _write_image(source, changed)
    stale = build_manifest_preview(project)
    assert not stale.valid
    assert "lesion_materialization" in {error.error_code for error in stale.errors}


def test_cli_materialize_dry_run_and_execute(tmp_path):
    project_path, source, _data_root, output_root = _project(
        tmp_path, mode="labeled_map"
    )
    _write_image(source, _labeled_array())
    runner = CliRunner()

    dry = runner.invoke(
        main, ["lesions", "materialize", "--project", str(project_path), "--dry-run"]
    )
    assert dry.exit_code == 0, dry.output
    assert "Dry run: no lesion masks were written" in dry.output
    assert "Materializing lesion masks" not in dry.output
    assert not output_root.exists()

    run = runner.invoke(
        main, ["lesions", "materialize", "--project", str(project_path)]
    )
    assert run.exit_code == 0, run.output
    assert "completed: 1" in run.output
    assert "minimum lesion size: 10 voxels" in run.output
    # Progress goes to stderr, which CliRunner includes in output on click 8.1 and 8.2+.
    assert "Materializing lesion masks: 1 subject/session(s)" in run.output


def test_materialization_progress_reports_each_subject_session_in_order(tmp_path):
    project_path, source, data_root, _output_root = _project(
        tmp_path, mode="labeled_map"
    )
    _write_image(source, _labeled_array())
    _write_image(data_root / "sub-002" / "lesions" / "combined.nii.gz", _labeled_array())
    plan = build_materialization_plan(load_project_config(project_path))
    messages: list[str] = []

    assert run_materialization(plan, dry_run=True, progress=messages.append).valid
    assert messages == []

    assert run_materialization(plan, progress=messages.append).valid
    assert messages == [
        "Materializing lesion masks: 2 subject/session(s)",
        "[1/2] sub-001/ses-01",
        "[2/2] sub-002/ses-01",
    ]


def test_materialization_is_not_used_for_already_separated_masks(tmp_path):
    data = yaml.safe_load(template_text("project"))
    data["data_root"] = str(tmp_path / "data")
    data["output_root"] = str(tmp_path / "output")
    (tmp_path / "data").mkdir()
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializationError, match="already separated"):
        build_materialization_plan(load_project_config(path))
