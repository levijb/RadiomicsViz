import csv
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
import SimpleITK as sitk
import yaml
from click.testing import CliRunner

from radiomicviz.cli import main
from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs


def _write_image(path: Path, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


def _write_mask(
    path: Path,
    block,
    *,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    value=1,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((10, 10, 10), dtype=np.uint8)
    array[block] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


def _project(tmp_path: Path, *, spacing=(1.0, 1.0, 1.0)):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    subject = data_root / "sub-001"
    _write_image(subject / "anat" / "sub-001_T1w_preprocessed.nii.gz", spacing=spacing)
    _write_image(subject / "anat" / "sub-001_FLAIR_preprocessed.nii.gz", spacing=spacing)
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", np.s_[1:5, 1:5, 1:5], spacing=spacing)
    _write_mask(subject / "lesions" / "lesion_002.nii.gz", np.s_[5:9, 5:9, 5:9], spacing=spacing)

    data = yaml.safe_load(template_text("project"))
    data["data_root"] = str(data_root)
    data["output_root"] = str(output_root)
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return project_path, data_root, output_root


def test_valid_manifest_is_deterministic_and_uses_analysis_ready_images(tmp_path):
    project_path, _data_root, output_root = _project(tmp_path, spacing=(1.00000691, 1.0, 1.0))
    project = load_project_config(project_path)

    first = build_manifest_preview(project)
    second = build_manifest_preview(project)

    assert first.valid
    assert first.rows == second.rows
    assert first.subjects_found == 1
    assert first.sessions_found == 1
    assert first.regions_found == 2
    assert len(first.rows) == 4
    assert [row.region_id for row in first.rows] == [
        "lesion_001", "lesion_001", "lesion_002", "lesion_002"
    ]
    assert {row.modality for row in first.rows} == {"T1", "FLAIR"}
    assert all(Path(row.image_path).is_file() for row in first.rows)
    assert all(Path(row.image_path).is_relative_to(tmp_path / "data") for row in first.rows)
    assert not first.errors


def test_progress_reports_each_subject_session_in_order(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    subject = data_root / "sub-002"
    _write_image(subject / "anat" / "sub-002_T1w_preprocessed.nii.gz")
    _write_image(subject / "anat" / "sub-002_FLAIR_preprocessed.nii.gz")
    _write_mask(subject / "lesions" / "lesion_001.nii.gz", np.s_[1:5, 1:5, 1:5])
    messages: list[str] = []

    preview = build_manifest_preview(load_project_config(project_path), progress=messages.append)

    assert preview.valid
    assert messages == [
        "Building manifest: 2 subject/session(s)",
        "[1/2] sub-001/ses-01",
        "[2/2] sub-002/ses-01",
    ]


def test_nonreference_spacing_fails_by_default(tmp_path):
    project_path, _data_root, _output_root = _project(tmp_path, spacing=(1.2, 1.0, 1.0))
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert preview.reference_status == "reference-compatible"
    assert {error.error_code for error in preview.errors} == {"reference_spacing"}


def test_nonreference_spacing_override_warns_and_stamps_run(tmp_path):
    project_path, _data_root, _output_root = _project(tmp_path, spacing=(1.2, 1.0, 1.0))
    data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    data["allow_nonreference_spacing"] = True
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    preview = build_manifest_preview(load_project_config(project_path))

    assert preview.valid
    assert preview.reference_status == "non-reference-spacing"
    assert sum(c.check_id == "reference_spacing" and c.status == "WARN" for c in preview.checks) == 2


def test_missing_modality_is_a_blocking_failure(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    (data_root / "sub-001" / "anat" / "sub-001_FLAIR_preprocessed.nii.gz").unlink()
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert "image_missing" in {error.error_code for error in preview.errors}


def test_invalid_mask_values_are_rejected(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    _write_mask(
        data_root / "sub-001" / "lesions" / "lesion_001.nii.gz",
        np.s_[1:5, 1:5, 1:5],
        value=255,
    )
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert "mask_values" in {error.error_code for error in preview.errors}


def test_geometry_mismatch_is_rejected(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    _write_mask(
        data_root / "sub-001" / "lesions" / "lesion_001.nii.gz",
        np.s_[1:5, 1:5, 1:5],
        origin=(1.0, 0.0, 0.0),
    )
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert "mask_image_geometry" in {error.error_code for error in preview.errors}
    assert any("(origin)" in error.message for error in preview.errors)


def test_identifiers_extracted_from_paths_must_be_safe(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    source = data_root / "sub-001" / "lesions" / "lesion_001.nii.gz"
    source.rename(source.with_name("lesion bad.nii.gz"))
    data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    data["regions"]["id_regex"] = r"^(?P<region>.+)\.nii\.gz$"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert "region_id_unsafe" in {error.error_code for error in preview.errors}


def test_patterns_cannot_escape_data_root(tmp_path):
    project_path, _data_root, _output_root = _project(tmp_path)
    data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    data["modalities"]["T1"]["image_pattern"] = "{subject}/../*.nii.gz"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    preview = build_manifest_preview(load_project_config(project_path))
    assert not preview.valid
    assert "image_pattern" in {error.error_code for error in preview.errors}


def test_truncated_nifti_is_rejected_before_extraction(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    image_path = data_root / "sub-001" / "anat" / "sub-001_T1w_preprocessed.nii.gz"
    payload = image_path.read_bytes()
    image_path.write_bytes(payload[: len(payload) // 2])

    preview = build_manifest_preview(load_project_config(project_path))

    assert not preview.valid
    assert "image_unreadable" in {error.error_code for error in preview.errors}


def test_nonfinite_nifti_is_rejected_before_simpleitk_can_sanitise_it(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    image_path = data_root / "sub-001" / "anat" / "sub-001_T1w_preprocessed.nii.gz"
    source = nib.load(str(image_path))
    values = np.asanyarray(source.dataobj).copy()
    values.flat[0] = np.nan
    values.flat[1] = np.inf
    nib.save(nib.Nifti1Image(values, source.affine, source.header), str(image_path))

    preview = build_manifest_preview(load_project_config(project_path))

    assert not preview.valid
    assert "image_nonfinite" in {error.error_code for error in preview.errors}


def test_vector_image_is_rejected_with_a_concise_error(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    image_path = data_root / "sub-001" / "anat" / "sub-001_T1w_preprocessed.nii.gz"
    vector = sitk.GetImageFromArray(
        np.zeros((10, 10, 10, 2), dtype=np.float32), isVector=True
    )
    vector.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(vector, str(image_path))

    preview = build_manifest_preview(load_project_config(project_path))

    assert not preview.valid
    assert "image_components" in {error.error_code for error in preview.errors}


def test_output_root_inside_data_root_is_never_written(tmp_path):
    project_path, data_root, _output_root = _project(tmp_path)
    unsafe_output = data_root / "generated"
    data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    data["output_root"] = str(unsafe_output)
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        main, ["manifest", "--project", str(project_path), "--write"]
    )

    assert result.exit_code != 0
    assert "separate" in result.output or "inside data_root" in result.output
    assert not unsafe_output.exists()


def test_writer_creates_minimal_artifacts_and_hashes(tmp_path):
    project_path, _data_root, output_root = _project(tmp_path)
    preview = build_manifest_preview(load_project_config(project_path))

    result = write_manifest_outputs(preview)

    assert result.manifest_path == output_root / "manifest" / "regions_manifest.csv"
    assert result.manifest_path.is_file()
    assert result.checks_path.is_file()
    assert result.errors_path.is_file()
    assert result.metadata_path.is_file()
    assert result.log_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["manifest_rows"] == 4
    assert len(metadata["project_sha256"]) == 64
    assert len(metadata["manifest_sha256"]) == 64
    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "subject_id", "session_id", "region_id", "region_type", "modality",
            "image_path", "mask_path", "label", "space_id", "image_size_bytes",
            "image_mtime_utc", "image_sha256", "mask_size_bytes", "mask_mtime_utc",
            "mask_sha256",
        ]
        assert len(list(reader)) == 4


def test_failed_writer_keeps_diagnostics_but_not_manifest(tmp_path):
    project_path, _data_root, output_root = _project(tmp_path, spacing=(1.2, 1.0, 1.0))
    preview = build_manifest_preview(load_project_config(project_path))
    result = write_manifest_outputs(preview)
    assert result.manifest_path is None
    assert not (output_root / "manifest" / "regions_manifest.csv").exists()
    assert result.errors_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"


def test_rewrite_archives_previous_outputs_without_force(tmp_path):
    project_path, _data_root, output_root = _project(tmp_path)
    preview = build_manifest_preview(load_project_config(project_path))
    first = write_manifest_outputs(preview)
    first_manifest = first.manifest_path.read_bytes()

    second = write_manifest_outputs(preview)

    assert len(second.backups) == 5
    assert all(path.exists() and ".bak-" in path.name for path in second.backups)
    archived_manifest = [path for path in second.backups if path.name.startswith("regions_manifest.csv")]
    assert archived_manifest[0].read_bytes() == first_manifest
    assert second.manifest_path.read_bytes() == first_manifest
    assert not list((output_root / "manifest").glob(".*.tmp"))


def test_failed_rewrite_archives_the_previous_manifest_and_writes_none(tmp_path):
    project_path, data_root, output_root = _project(tmp_path)
    write_manifest_outputs(build_manifest_preview(load_project_config(project_path)))
    (data_root / "sub-001" / "anat" / "sub-001_T1w_preprocessed.nii.gz").rename(
        data_root / "sub-001" / "anat" / "sub-001_T1w_preprocessed.moved.nii.gz"
    )

    result = write_manifest_outputs(build_manifest_preview(load_project_config(project_path)))

    assert result.manifest_path is None
    assert not (output_root / "manifest" / "regions_manifest.csv").exists()
    assert any(path.name.startswith("regions_manifest.csv.bak-") for path in result.backups)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"


def test_cli_is_preview_only_until_write_is_requested(tmp_path):
    project_path, _data_root, output_root = _project(tmp_path)
    runner = CliRunner()

    preview = runner.invoke(main, ["manifest", "--project", str(project_path)])
    assert preview.exit_code == 0, preview.output
    assert "Preview only: no files written" in preview.output
    assert not output_root.exists()

    written = runner.invoke(main, ["manifest", "--project", str(project_path), "--write"])
    assert written.exit_code == 0, written.output
    assert "Wrote canonical manifest" in written.output
    assert (output_root / "manifest" / "regions_manifest.csv").is_file()

    rewritten = runner.invoke(main, ["manifest", "--project", str(project_path), "--write"])
    assert rewritten.exit_code == 0, rewritten.output
    assert "Archived 5 previous artifact(s)" in rewritten.output

    forced = runner.invoke(main, ["manifest", "--project", str(project_path), "--write", "--force"])
    assert forced.exit_code == 2
    assert "No such option" in forced.output


def test_discovered_sessions_are_supported(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    for session in ("ses-01", "ses-02"):
        base = data_root / "sub-001" / session
        _write_image(base / "anat" / "T1.nii.gz")
        _write_image(base / "anat" / "FLAIR.nii.gz")
        _write_mask(base / "lesions" / "lesion_001.nii.gz", np.s_[1:5, 1:5, 1:5])
    data = yaml.safe_load(template_text("project"))
    data["data_root"] = str(data_root)
    data["output_root"] = str(output_root)
    data["sessions"] = {
        "mode": "discovered",
        "default_session_id": None,
        "discovery_pattern": "ses-*",
        "id_regex": "^(?P<session>ses-[0-9]+)$",
    }
    data["modalities"]["T1"]["image_pattern"] = "{subject}/{session}/anat/T1.nii.gz"
    data["modalities"]["FLAIR"]["image_pattern"] = "{subject}/{session}/anat/FLAIR.nii.gz"
    data["regions"]["mask_pattern"] = "{subject}/{session}/lesions/*.nii.gz"
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    preview = build_manifest_preview(load_project_config(project_path))

    assert preview.valid
    assert preview.sessions_found == 2
    assert preview.regions_found == 2
    assert len(preview.rows) == 4
    assert {row.session_id for row in preview.rows} == {"ses-01", "ses-02"}


def test_single_session_mode_warns_before_ignoring_other_session_folders(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    for session in ("ses-01", "ses-02"):
        base = data_root / "sub-001" / session
        _write_image(base / "anat" / "T1.nii.gz")
        _write_image(base / "anat" / "FLAIR.nii.gz")
        _write_mask(base / "lesions" / "lesion_001.nii.gz", np.s_[1:5, 1:5, 1:5])
    data = yaml.safe_load(template_text("project"))
    data["data_root"] = str(data_root)
    data["output_root"] = str(output_root)
    data["modalities"]["T1"]["image_pattern"] = "{subject}/{session}/anat/T1.nii.gz"
    data["modalities"]["FLAIR"]["image_pattern"] = "{subject}/{session}/anat/FLAIR.nii.gz"
    data["regions"]["mask_pattern"] = "{subject}/{session}/lesions/*.nii.gz"
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    preview = build_manifest_preview(load_project_config(project_path))

    assert preview.valid
    assert preview.sessions_found == 1
    assert {row.session_id for row in preview.rows} == {"ses-01"}
    warning = next(
        check for check in preview.checks
        if check.check_id == "additional_sessions_ignored"
    )
    assert warning.status == "WARN"
    assert "ses-02" in warning.measured_value

    cli = CliRunner().invoke(main, ["manifest", "--project", str(project_path)])
    assert cli.exit_code == 0
    assert "WARN [sub-001/ses-01] additional_sessions_ignored" in cli.output
