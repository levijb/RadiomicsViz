import pytest

from radiomicviz.config import (
    ConfigError,
    load_project_config,
    parse_project_config,
)
from radiomicviz.config.templates import template_text, write_template


def errors_of(parse, data):
    with pytest.raises(ConfigError) as info:
        parse(data)
    return info.value.errors


# Shipped templates -----------------------------------------------------------


def test_project_template_loads(tmp_path):
    written = write_template("project", tmp_path / "project.yaml")
    config = load_project_config(written.path)
    assert config.source == written.path
    assert config.schema_version == 1
    assert [m.name for m in config.modalities] == ["T1", "FLAIR"]
    assert config.sessions.mode == "single"
    assert config.sessions.default_session_id == "ses-01"
    assert config.regions.mode == "per_region_masks"
    assert config.regions.region_type == "lesion"
    assert config.regions.label == 1
    assert config.regions.connectivity is None
    assert config.allow_nonreference_spacing is False


def test_paths_are_not_checked_for_existence(project_data):
    project_data["data_root"] = "/no/such/folder"
    assert parse_project_config(project_data).data_root == "/no/such/folder"


def test_discovered_sessions(project_data):
    project_data["sessions"] = {
        "mode": "discovered",
        "default_session_id": None,
        "discovery_pattern": "ses-*",
        "id_regex": "^(?P<session>ses-[0-9]+)$",
    }
    for modality in project_data["modalities"].values():
        modality["image_pattern"] = "{subject}/{session}/anat/image.nii.gz"
    project_data["regions"]["mask_pattern"] = "{subject}/{session}/lesions/*.nii.gz"
    config = parse_project_config(project_data)
    assert config.sessions.mode == "discovered"
    assert config.sessions.default_session_id is None


def test_discovered_sessions_require_session_placeholder(project_data):
    project_data["sessions"] = {
        "mode": "discovered",
        "default_session_id": "ses-01",
        "discovery_pattern": "ses-*",
        "id_regex": "^(?P<session>ses-[0-9]+)$",
    }
    errors = errors_of(parse_project_config, project_data)
    assert "sessions.default_session_id: must be null when sessions.mode is 'discovered'" in errors
    assert "modalities.T1.image_pattern: must contain the placeholder {session}" in errors
    assert "modalities.FLAIR.image_pattern: must contain the placeholder {session}" in errors
    assert "regions.mask_pattern: must contain the placeholder {session}" in errors


def test_every_problem_is_reported_at_once(project_data):
    project_data["threads"] = 4
    project_data["regions"]["label"] = "1"
    with pytest.raises(ConfigError) as info:
        parse_project_config(project_data)
    assert len(info.value.errors) == 2
    assert "2 problems found" in str(info.value)


# Project schema violations ---------------------------------------------------


def _set(*keys_and_value):
    *keys, last, value = keys_and_value

    def mutate(data):
        for key in keys:
            data = data[key]
        data[last] = value

    return mutate


def _drop(key):
    return lambda data: data.pop(key)


PROJECT_CASES = [
    (_set("extra", 1), "extra: unknown key (allowed keys: schema_version, "),
    (_drop("space_id"), "space_id: required key is missing"),
    (_set("regions", "labels", 2), "regions.labels: unknown key (allowed keys: mode, region_type, "),
    (_set("modalities", "T1", "brain_mask_pattern", None), "modalities.T1.brain_mask_pattern: unknown key"),
    (_set("schema_version", 2), "schema_version: unsupported version 2 (this release reads version 1)"),
    (_set("schema_version", "1"), "schema_version: must be an integer, not a string"),
    (_set("modalities", "T1", "input_state", "original"), "modalities.T1.input_state: unknown key"),
    (
        _set("sessions", "default_session_id", 1),
        "sessions.default_session_id: must be a string, not an integer (put quotes around it)",
    ),
    (
        _set("space_id", "native space"),
        "space_id: 'native space' is not a safe identifier (use only letters, digits, '_', '.', '-')",
    ),
    (_set("modalities", {"T 1": {"image_pattern": "{subject}/t1.nii.gz"}}),
     "modalities.T 1: 'T 1' is not a safe identifier"),
    (_set("regions", "label", True), "regions.label: must be an integer, not a boolean"),
    (_set("regions", "label", 0), "regions.label: must be at least 1 (got 0)"),
    (
        _set("allow_nonreference_spacing", "no"),
        "allow_nonreference_spacing: must be true or false, not a string",
    ),
    (
        _set("subjects", "id_regex", "^sub-.*$"),
        "subjects.id_regex: regular expression must define the named group (?P<subject>...)",
    ),
    (_set("regions", "id_regex", "(?P<region>lesion_[0-9]+"), "regions.id_regex: invalid regular expression"),
    (
        _set("modalities", "T1", "image_pattern", "{subject}/{visit}/t1.nii.gz"),
        "modalities.T1.image_pattern: unknown placeholder {visit} (allowed: {subject}, {session})",
    ),
    (
        _set("regions", "mask_pattern", "lesions/*.nii.gz"),
        "regions.mask_pattern: must contain the placeholder {subject}",
    ),
    (
        _set("regions", "mode", "watershed"),
        "regions.mode: must be one of 'per_region_masks', 'labeled_map', 'connected_components'",
    ),
    (
        _set("sessions", "mode", "multiple"),
        "sessions.mode: must be one of 'single', 'discovered' (got 'multiple')",
    ),
    (
        _set("sessions", "discovery_pattern", "ses-*"),
        "sessions.discovery_pattern: must be null when sessions.mode is 'single'",
    ),
    (_set("modalities", {}), "modalities: at least one modality is required"),
    (_set("regions", ["lesion"]), "regions: must be a mapping, not a list"),
    (_set("output_root", "   "), "output_root: must not be empty"),
    (_set("data_root", None), "data_root: must be a string, not null"),
]


@pytest.mark.parametrize("mutate, expected", PROJECT_CASES)
def test_project_schema_violation(project_data, mutate, expected):
    mutate(project_data)
    errors = errors_of(parse_project_config, project_data)
    assert any(message.startswith(expected) for message in errors), errors


def test_labeled_map_configuration(project_data):
    project_data["regions"].update(
        mode="labeled_map",
        mask_pattern="{subject}/lesions/labeled.nii.gz",
        id_regex=None,
        label=None,
        connectivity=None,
    )
    regions = parse_project_config(project_data).regions
    assert regions.mode == "labeled_map"
    assert regions.id_regex is None
    assert regions.label is None
    assert regions.connectivity is None


def test_connected_components_configuration(project_data):
    project_data["regions"].update(
        mode="connected_components",
        mask_pattern="{subject}/lesions/combined.nii.gz",
        id_regex=None,
        label=1,
        connectivity=26,
    )
    regions = parse_project_config(project_data).regions
    assert regions.mode == "connected_components"
    assert regions.label == 1
    assert regions.connectivity == 26


@pytest.mark.parametrize("connectivity", [None, 0, 18])
def test_connected_components_requires_explicit_supported_connectivity(
    project_data, connectivity
):
    project_data["regions"].update(
        mode="connected_components",
        mask_pattern="{subject}/lesions/combined.nii.gz",
        id_regex=None,
        label=1,
        connectivity=connectivity,
    )
    errors = errors_of(parse_project_config, project_data)
    assert any(error.startswith("regions.connectivity:") for error in errors)


def test_labeled_map_rejects_binary_mode_fields(project_data):
    project_data["regions"].update(
        mode="labeled_map",
        mask_pattern="{subject}/lesions/labeled.nii.gz",
        connectivity=26,
    )
    errors = errors_of(parse_project_config, project_data)
    assert "regions.id_regex: must be null when regions.mode is 'labeled_map'" in errors
    assert "regions.label: must be null when regions.mode is 'labeled_map'" in errors
    assert "regions.connectivity: must be null when regions.mode is 'labeled_map'" in errors


# Reading files ---------------------------------------------------------------


def _write(tmp_path, text, name="config.yaml", encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def _file_errors(loader, path):
    with pytest.raises(ConfigError) as info:
        loader(path)
    assert info.value.source == str(path)
    return info.value.errors


def test_duplicate_top_level_key_is_rejected(tmp_path):
    text = template_text("project").replace("\nspace_id: native\n", "\nspace_id: native\nspace_id: other\n")
    errors = _file_errors(load_project_config, _write(tmp_path, text))
    assert len(errors) == 1
    assert errors[0].startswith("invalid YAML: found duplicate key 'space_id' (line ")


def test_duplicate_nested_key_is_rejected(tmp_path):
    text = template_text("project").replace("\n  FLAIR:\n", "\n  T1:\n")
    errors = _file_errors(load_project_config, _write(tmp_path, text))
    assert errors[0].startswith("invalid YAML: found duplicate key 'T1' (line ")


def test_yaml_syntax_error_reports_position(tmp_path):
    errors = _file_errors(load_project_config, _write(tmp_path, "schema_version: [1\n"))
    assert errors[0].startswith("invalid YAML: ")
    assert "(line " in errors[0] and "column " in errors[0]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", "file is empty"),
        ("# only a comment\n", "file is empty"),
        ("- a\n- b\n", "top level must be a mapping of keys to values, not a list"),
        ("just text\n", "top level must be a mapping of keys to values, not a string"),
    ],
)
def test_file_must_hold_a_mapping(tmp_path, text, expected):
    assert _file_errors(load_project_config, _write(tmp_path, text)) == [expected]


def test_missing_file(tmp_path):
    assert _file_errors(load_project_config, tmp_path / "absent.yaml") == ["file not found"]


def test_invalid_utf8(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_bytes(b"schema_version: \xff\n")
    errors = _file_errors(load_project_config, path)
    assert errors[0].startswith("file is not valid UTF-8")


def test_utf8_byte_order_mark_is_accepted(tmp_path):
    path = _write(tmp_path, template_text("project"), encoding="utf-8-sig")
    assert load_project_config(path).schema_version == 1
