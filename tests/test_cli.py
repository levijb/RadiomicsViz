import re

import pytest
from click.testing import CliRunner

from radiomicviz import __version__
from radiomicviz.cli import main
from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text

@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"radiomicviz {__version__}\n"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_lists_commands(runner, flag):
    result = runner.invoke(main, [flag])
    assert result.exit_code == 0
    assert "Usage: radiomicviz" in result.output
    assert re.search(r"^\s+project\s", result.output, re.MULTILINE)
    assert re.search(r"^\s+lesions\s", result.output, re.MULTILINE)
    assert re.search(r"^\s+manifest\s", result.output, re.MULTILINE)
    assert re.search(r"^\s+whole\s", result.output, re.MULTILINE)
    assert re.search(r"^\s+voxelwise\s", result.output, re.MULTILINE)
    assert not re.search(r"^\s+preprocess\s", result.output, re.MULTILINE)
    assert not re.search(r"^\s+site\s", result.output, re.MULTILINE)


def test_whole_help_exposes_reference_run_controls(runner):
    result = runner.invoke(main, ["whole", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--protocol" in result.output
    assert "--variant" in result.output
    assert "--jobs" in result.output
    assert "--dry-run" in result.output
    assert "--force" in result.output
    assert "--allow-version-mismatch" in result.output


def test_voxelwise_help_exposes_reference_run_controls(runner):
    result = runner.invoke(main, ["voxelwise", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--protocol" in result.output
    assert "--variant" in result.output
    assert "--jobs" in result.output
    assert "--dry-run" in result.output
    assert "--force" in result.output
    assert "--allow-version-mismatch" in result.output
    assert "minimum ROI size" in result.output


def test_manifest_help_explains_preview_and_write(runner):
    result = runner.invoke(main, ["manifest", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.output
    assert "--write" in result.output
    assert ".bak-<timestamp>" in result.output
    assert "--force" not in result.output
    assert re.search(r"read-only\s+preview", result.output)


def test_lesion_materialization_help_exposes_safe_controls(runner):
    result = runner.invoke(main, ["lesions", "materialize", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.output
    assert "--dry-run" in result.output
    assert "--force" in result.output
    assert "Touching foreground voxels remain one lesion" in result.output


def test_init_help(runner):
    result = runner.invoke(main, ["project", "init", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--force" in result.output


def test_init_writes_a_valid_template(runner, tmp_path):
    target = tmp_path / "project.yaml"
    result = runner.invoke(main, ["project", "init", "--output", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8") == template_text("project")
    assert target.read_bytes().count(b"\r") == 0
    load_project_config(target)
    assert "Moved" not in result.output


def test_init_defaults_to_current_directory(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["project", "init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "project.yaml").is_file()


def test_init_refuses_to_overwrite(runner, tmp_path):
    target = tmp_path / "project.yaml"
    target.write_text("keep me\n", encoding="utf-8")
    result = runner.invoke(main, ["project", "init", "-o", str(target)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "--force" in result.output
    assert target.read_text(encoding="utf-8") == "keep me\n"
    assert list(tmp_path.iterdir()) == [target]


def test_init_force_backs_up_the_existing_file(runner, tmp_path):
    target = tmp_path / "project.yaml"
    target.write_text("old contents\n", encoding="utf-8")
    result = runner.invoke(main, ["project", "init", "-o", str(target), "--force"])
    assert result.exit_code == 0, result.output

    backups = list(tmp_path.glob("project.yaml.bak-*"))
    assert len(backups) == 1
    assert re.fullmatch(r"project\.yaml\.bak-\d{8}T\d{6}", backups[0].name)
    assert backups[0].read_text(encoding="utf-8") == "old contents\n"
    assert backups[0].name in result.output
    assert target.read_text(encoding="utf-8") == template_text("project")


def test_repeated_force_never_loses_a_backup(runner, tmp_path):
    target = tmp_path / "project.yaml"
    target.write_text("first\n", encoding="utf-8")
    for _ in range(2):
        result = runner.invoke(main, ["project", "init", "-o", str(target), "--force"])
        assert result.exit_code == 0, result.output

    contents = sorted(p.read_text(encoding="utf-8") for p in tmp_path.glob("project.yaml.bak-*"))
    assert contents == sorted(["first\n", template_text("project")])


def test_init_force_without_existing_file_writes_no_backup(runner, tmp_path):
    target = tmp_path / "project.yaml"
    result = runner.invoke(main, ["project", "init", "-o", str(target), "--force"])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == [target]


def test_init_requires_an_existing_parent_directory(runner, tmp_path):
    target = tmp_path / "missing" / "project.yaml"
    result = runner.invoke(main, ["project", "init", "-o", str(target)])
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert not (tmp_path / "missing").exists()


def test_init_rejects_a_directory_as_output(runner, tmp_path):
    result = runner.invoke(main, ["project", "init", "-o", str(tmp_path)])
    assert result.exit_code != 0
    assert "is a directory" in result.output
