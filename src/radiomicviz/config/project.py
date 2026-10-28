"""Project configuration (``project.yaml``): where analysis-ready inputs are laid out.

The project file describes input discovery only. It does not hold scientific settings
(those belong to the protocol). All configured images must already be preprocessed,
co-registered with their region masks and ready for radiomics.
Loading checks the schema strictly: unknown keys, missing keys, wrong types and
inconsistent combinations are errors. Paths are not checked for existence here; that
happens when the manifest is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radiomicviz.config._common import Checker, load_yaml_mapping

PROJECT_SCHEMA_VERSION = 1

SESSION_MODES = ("single", "discovered")
REGION_MODES = ("per_region_masks", "labeled_map", "connected_components")

_TOP_LEVEL_KEYS = (
    "schema_version",
    "data_root",
    "output_root",
    "sessions",
    "subjects",
    "modalities",
    "regions",
    "space_id",
    "allow_nonreference_spacing",
)
_PATH_PLACEHOLDERS = ("subject", "session")


@dataclass(frozen=True)
class SessionSpec:
    """How sessions are identified.

    ``single``: one session per subject, recorded as ``default_session_id``.
    ``discovered``: session directories are found inside each subject directory with
    ``discovery_pattern`` and named by the ``session`` group of ``id_regex``.
    """

    mode: str
    default_session_id: str | None
    discovery_pattern: str | None
    id_regex: str | None


@dataclass(frozen=True)
class SubjectSpec:
    """How subject directories are found under the data root and named."""

    discovery_pattern: str
    id_regex: str


@dataclass(frozen=True)
class ModalitySpec:
    """Where one analysis-ready modality image is found for each session."""

    name: str
    image_pattern: str


@dataclass(frozen=True)
class RegionSpec:
    """How region masks are supplied or materialized.

    ``per_region_masks`` reads one existing binary file per region. ``labeled_map``
    materializes every positive integer in one combined map. ``connected_components``
    materializes the configured foreground value of one binary mask using explicit
    6- or 26-connectivity.
    """

    mode: str
    region_type: str
    mask_pattern: str
    id_regex: str | None
    label: int | None
    connectivity: int | None


@dataclass(frozen=True)
class ProjectConfig:
    """A validated ``project.yaml``.

    Relative ``data_root`` and ``output_root`` values are interpreted relative to the
    directory that contains the project file (``source.parent``).
    """

    schema_version: int
    data_root: str
    output_root: str
    sessions: SessionSpec
    subjects: SubjectSpec
    modalities: tuple[ModalitySpec, ...]
    regions: RegionSpec
    space_id: str
    allow_nonreference_spacing: bool
    source: Path | None = None

def load_project_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project file.

    Raises:
        ConfigError: Listing every problem found in the file.
    """
    path = Path(path)
    return parse_project_config(load_yaml_mapping(path), source=path)


def parse_project_config(data: dict[str, Any], source: Path | None = None) -> ProjectConfig:
    """Validate an already-parsed project mapping.

    Raises:
        ConfigError: Listing every problem found in ``data``.
    """
    check = Checker()
    check.keys(data, "", required=_TOP_LEVEL_KEYS)

    schema_version = None
    if "schema_version" in data:
        schema_version = check.integer(data["schema_version"], "schema_version")
        if schema_version is not None and schema_version != PROJECT_SCHEMA_VERSION:
            check.add(
                "schema_version",
                f"unsupported version {schema_version} (this release reads version {PROJECT_SCHEMA_VERSION})",
            )
            schema_version = None

    data_root = check.string(data["data_root"], "data_root") if "data_root" in data else None
    output_root = check.string(data["output_root"], "output_root") if "output_root" in data else None
    space_id = check.identifier(data["space_id"], "space_id") if "space_id" in data else None
    allow_nonreference = (
        check.boolean(data["allow_nonreference_spacing"], "allow_nonreference_spacing")
        if "allow_nonreference_spacing" in data
        else None
    )

    sessions = _parse_sessions(check, data["sessions"]) if "sessions" in data else None
    subjects = _parse_subjects(check, data["subjects"]) if "subjects" in data else None
    session_required = sessions is not None and sessions.mode == "discovered"
    modalities = (
        _parse_modalities(check, data["modalities"], session_required) if "modalities" in data else None
    )
    regions = _parse_regions(check, data["regions"], session_required) if "regions" in data else None

    check.raise_if_errors(str(source) if source is not None else "project configuration")
    return ProjectConfig(
        schema_version=schema_version,
        data_root=data_root,
        output_root=output_root,
        sessions=sessions,
        subjects=subjects,
        modalities=modalities,
        regions=regions,
        space_id=space_id,
        allow_nonreference_spacing=allow_nonreference,
        source=source,
    )


def _parse_sessions(check: Checker, value: Any) -> SessionSpec | None:
    section = check.mapping(value, "sessions")
    if section is None:
        return None
    check.keys(
        section,
        "sessions",
        required=("mode",),
        optional=("default_session_id", "discovery_pattern", "id_regex"),
    )
    mode = check.choice(section.get("mode"), "sessions.mode", SESSION_MODES) if "mode" in section else None
    default_id = section.get("default_session_id")
    pattern = section.get("discovery_pattern")
    id_regex = section.get("id_regex")

    if mode == "single":
        default_id = check.identifier(default_id, "sessions.default_session_id")
        for key, current in (("discovery_pattern", pattern), ("id_regex", id_regex)):
            if current is not None:
                check.add(f"sessions.{key}", "must be null when sessions.mode is 'single'")
        pattern = id_regex = None
    elif mode == "discovered":
        if default_id is not None:
            check.add("sessions.default_session_id", "must be null when sessions.mode is 'discovered'")
        default_id = None
        pattern = check.string(pattern, "sessions.discovery_pattern")
        id_regex = check.regex(id_regex, "sessions.id_regex", group="session")
    else:
        return None
    return SessionSpec(mode=mode, default_session_id=default_id, discovery_pattern=pattern, id_regex=id_regex)


def _parse_subjects(check: Checker, value: Any) -> SubjectSpec | None:
    section = check.mapping(value, "subjects")
    if section is None:
        return None
    check.keys(section, "subjects", required=("discovery_pattern", "id_regex"))
    pattern = check.string(section.get("discovery_pattern"), "subjects.discovery_pattern")
    id_regex = check.regex(section.get("id_regex"), "subjects.id_regex", group="subject")
    if pattern is None or id_regex is None:
        return None
    return SubjectSpec(discovery_pattern=pattern, id_regex=id_regex)


def _parse_modalities(check: Checker, value: Any, session_required: bool) -> tuple[ModalitySpec, ...] | None:
    section = check.mapping(value, "modalities")
    if section is None:
        return None
    if not section:
        check.add("modalities", "at least one modality is required")
        return None
    required_placeholders = ("subject", "session") if session_required else ("subject",)

    parsed: list[ModalitySpec] = []
    for name, entry in section.items():
        where = f"modalities.{name}"
        modality = check.identifier(name, where)
        body = check.mapping(entry, where)
        if body is None or modality is None:
            continue
        check.keys(body, where, required=("image_pattern",))
        image = check.path_template(
            body.get("image_pattern"),
            f"{where}.image_pattern",
            allowed=_PATH_PLACEHOLDERS,
            required=required_placeholders,
        )
        if image is None:
            continue
        parsed.append(ModalitySpec(name=modality, image_pattern=image))
    if len(parsed) != len(section):
        return None
    return tuple(parsed)


def _parse_regions(check: Checker, value: Any, session_required: bool) -> RegionSpec | None:
    section = check.mapping(value, "regions")
    if section is None:
        return None
    check.keys(
        section,
        "regions",
        required=("mode", "region_type", "mask_pattern", "id_regex", "label"),
        optional=("connectivity",),
    )
    mode = (
        check.choice(section["mode"], "regions.mode", REGION_MODES)
        if "mode" in section
        else None
    )
    region_type = (
        check.identifier(section["region_type"], "regions.region_type") if "region_type" in section else None
    )
    mask = (
        check.path_template(
            section["mask_pattern"],
            "regions.mask_pattern",
            allowed=_PATH_PLACEHOLDERS,
            required=("subject", "session") if session_required else ("subject",),
        )
        if "mask_pattern" in section
        else None
    )
    raw_regex = section.get("id_regex")
    raw_label = section.get("label")
    raw_connectivity = section.get("connectivity")
    id_regex: str | None = None
    label: int | None = None
    connectivity: int | None = None

    if mode == "per_region_masks":
        id_regex = check.regex(raw_regex, "regions.id_regex", group="region")
        label = check.integer(raw_label, "regions.label", minimum=1)
        if raw_connectivity is not None:
            check.add(
                "regions.connectivity",
                "must be null or omitted when regions.mode is 'per_region_masks'",
            )
    elif mode == "labeled_map":
        if raw_regex is not None:
            check.add("regions.id_regex", "must be null when regions.mode is 'labeled_map'")
        if raw_label is not None:
            check.add("regions.label", "must be null when regions.mode is 'labeled_map'")
        if raw_connectivity is not None:
            check.add("regions.connectivity", "must be null when regions.mode is 'labeled_map'")
    elif mode == "connected_components":
        if raw_regex is not None:
            check.add(
                "regions.id_regex",
                "must be null when regions.mode is 'connected_components'",
            )
        label = check.integer(raw_label, "regions.label", minimum=1)
        connectivity = check.integer(raw_connectivity, "regions.connectivity")
        if connectivity is not None and connectivity not in (6, 26):
            check.add("regions.connectivity", "must be either 6 or 26")
            connectivity = None

    if None in (mode, region_type, mask):
        return None
    if mode == "per_region_masks" and None in (id_regex, label):
        return None
    if mode == "connected_components" and None in (label, connectivity):
        return None
    return RegionSpec(
        mode=mode,
        region_type=region_type,
        mask_pattern=mask,
        id_regex=id_regex,
        label=label,
        connectivity=connectivity,
    )
