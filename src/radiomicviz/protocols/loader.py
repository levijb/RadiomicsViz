"""Load the immutable built-in scientific protocol and radiomics presets.

Preset metadata keeps stage-specific scientific constants out of Python code and
authenticates the frozen PyRadiomics parameter files shipped with the package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from radiomicviz.config._common import Checker, ConfigError, load_yaml_mapping
from radiomicviz.policy import MINIMUM_REGION_VOXELS

REFERENCE_PROTOCOL_ID = "lesion-habitat-v1"


class ProtocolError(ValueError):
    """The requested scientific protocol or one of its frozen presets is invalid."""


@dataclass(frozen=True)
class RoiPreset:
    """One immutable PyRadiomics ROI parameter preset."""

    preset_id: str
    variant: str
    status: str
    parameter_path: Path
    parameter_sha256: str
    metadata_path: Path
    metadata_sha256: str
    requires_preprocessed_input: bool
    bin_count: int
    minimum_roi_size: int
    expected_feature_count: int
    expected_shape_feature_count: int
    expected_diagnostics_count: int
    expected_feature_schema_sha256: str


@dataclass(frozen=True)
class VoxelwisePreset:
    """One immutable PyRadiomics voxelwise parameter preset."""

    preset_id: str
    variant: str
    status: str
    parameter_path: Path
    parameter_sha256: str
    metadata_path: Path
    metadata_sha256: str
    requires_preprocessed_input: bool
    bin_count: int
    minimum_roi_size: int
    kernel_radius: int
    expected_map_count: int
    expected_map_schema_sha256: str
    expected_map_names: tuple[str, ...]


@dataclass(frozen=True)
class Protocol:
    """Validated scientific protocol used by a radiomics stage."""

    protocol_id: str
    protocol_version: str
    status: str
    source_path: Path
    sha256: str
    expected_software: Mapping[str, str]
    variants: Mapping[str, RoiPreset]
    voxelwise_variants: Mapping[str, VoxelwisePreset]

    def roi_preset(self, variant: str) -> RoiPreset:
        try:
            return self.variants[variant]
        except KeyError:
            available = ", ".join(sorted(self.variants))
            raise ProtocolError(
                f"protocol {self.protocol_id!r} has no variant {variant!r}; available: {available}"
            ) from None

    def voxelwise_preset(self, variant: str) -> VoxelwisePreset:
        try:
            return self.voxelwise_variants[variant]
        except KeyError:
            available = ", ".join(sorted(self.voxelwise_variants))
            raise ProtocolError(
                f"protocol {self.protocol_id!r} has no voxelwise variant {variant!r}; "
                f"available: {available}"
            ) from None


def load_protocol(identifier: str = REFERENCE_PROTOCOL_ID) -> Protocol:
    """Load a built-in protocol by identifier.

    User-authored protocols are deliberately deferred until the built-in 32-bin
    vertical slice has passed its validation tests.
    """
    if identifier != REFERENCE_PROTOCOL_ID:
        raise ProtocolError(
            f"unknown protocol {identifier!r}; this release provides only {REFERENCE_PROTOCOL_ID!r}"
        )
    package_root = files("radiomicviz")
    source = Path(str(package_root / "protocols" / f"{identifier}.yaml"))
    try:
        data = load_yaml_mapping(source)
    except ConfigError as exc:
        raise ProtocolError(str(exc)) from exc
    try:
        return _parse_protocol(data, source, Path(str(package_root)))
    except ConfigError as exc:
        raise ProtocolError(str(exc)) from exc


def _parse_protocol(data: dict, source: Path, package_root: Path) -> Protocol:
    checker = Checker()
    checker.keys(
        data,
        "",
        required=(
            "schema_version",
            "protocol_id",
            "protocol_version",
            "status",
            "software",
            "variants",
        ),
    )
    schema = checker.integer(data.get("schema_version"), "schema_version", minimum=1)
    if schema is not None and schema != 1:
        checker.add("schema_version", f"unsupported value {schema}; expected 1")
    protocol_id = checker.identifier(data.get("protocol_id"), "protocol_id")
    protocol_version = checker.string(data.get("protocol_version"), "protocol_version")
    status = checker.choice(
        data.get("status"), "status", ("reference-verified", "extension")
    )
    software = _parse_software(data.get("software"), checker)
    variants_data = checker.mapping(data.get("variants"), "variants")
    variants: dict[str, RoiPreset] = {}
    voxelwise_variants: dict[str, VoxelwisePreset] = {}
    if variants_data is not None:
        if not variants_data:
            checker.add("variants", "must contain at least one variant")
        for variant, value in variants_data.items():
            variant_id = checker.identifier(variant, f"variants.{variant}")
            mapping = checker.mapping(value, f"variants.{variant}")
            if variant_id is None or mapping is None:
                continue
            checker.keys(
                mapping,
                f"variants.{variant}",
                required=(
                    "status",
                    "roi_preset_metadata",
                    "voxelwise_preset_metadata",
                ),
            )
            variant_status = checker.choice(
                mapping.get("status"),
                f"variants.{variant}.status",
                ("reference-verified", "extension"),
            )
            metadata_name = checker.string(
                mapping.get("roi_preset_metadata"),
                f"variants.{variant}.roi_preset_metadata",
            )
            voxelwise_metadata_name = checker.string(
                mapping.get("voxelwise_preset_metadata"),
                f"variants.{variant}.voxelwise_preset_metadata",
            )
            if (
                variant_status is None
                or metadata_name is None
                or voxelwise_metadata_name is None
            ):
                continue
            metadata_path = package_root / "presets" / metadata_name
            try:
                preset = _load_preset(metadata_path, variant_id, variant_status)
            except ProtocolError as exc:
                checker.add(f"variants.{variant}.roi_preset_metadata", str(exc))
            else:
                variants[variant_id] = preset
            voxelwise_metadata_path = package_root / "presets" / voxelwise_metadata_name
            try:
                voxelwise_preset = _load_voxelwise_preset(
                    voxelwise_metadata_path, variant_id, variant_status
                )
            except ProtocolError as exc:
                checker.add(
                    f"variants.{variant}.voxelwise_preset_metadata", str(exc)
                )
            else:
                voxelwise_variants[variant_id] = voxelwise_preset
    checker.raise_if_errors(str(source))
    assert protocol_id is not None and protocol_version is not None and status is not None
    return Protocol(
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        status=status,
        source_path=source.resolve(),
        sha256=_sha256(source),
        expected_software=software,
        variants=variants,
        voxelwise_variants=voxelwise_variants,
    )


def _parse_software(value: object, checker: Checker) -> dict[str, str]:
    path = "software"
    mapping = checker.mapping(value, path)
    names = ("python", "pyradiomics", "numpy", "simpleitk", "pywavelets")
    result: dict[str, str] = {}
    if mapping is None:
        return result
    checker.keys(mapping, path, required=names)
    for name in names:
        parsed = checker.string(mapping.get(name), f"{path}.{name}")
        if parsed is not None:
            result[name] = parsed
    return result


def _load_preset(metadata_path: Path, variant: str, status: str) -> RoiPreset:
    if not metadata_path.is_file():
        raise ProtocolError(f"preset metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read preset metadata {metadata_path}: {exc}") from exc
    required = {
        "schema_version",
        "preset_id",
        "variant",
        "status",
        "parameter_file",
        "parameter_sha256",
        "requires_preprocessed_input",
        "bin_count",
        "minimum_roi_size",
        "expected_feature_count",
        "expected_shape_feature_count",
        "expected_diagnostics_count",
        "expected_feature_schema_sha256",
    }
    missing = sorted(required - set(metadata))
    unknown = sorted(set(metadata) - required - {"resampling_note", "source"})
    problems: list[str] = []
    if missing:
        problems.append("missing key(s): " + ", ".join(missing))
    if unknown:
        problems.append("unknown key(s): " + ", ".join(unknown))
    if metadata.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if metadata.get("variant") != variant:
        problems.append(f"variant must be {variant!r}")
    if metadata.get("status") != status:
        problems.append(f"status must be {status!r}")
    if metadata.get("requires_preprocessed_input") is not True:
        problems.append("requires_preprocessed_input must be true")
    parameter_name = metadata.get("parameter_file")
    if not isinstance(parameter_name, str) or not parameter_name:
        problems.append("parameter_file must be a non-empty string")
        parameter_path = metadata_path.with_name("invalid")
    else:
        parameter_path = metadata_path.with_name(parameter_name)
    if not parameter_path.is_file():
        problems.append(f"parameter file is missing: {parameter_path}")
    else:
        actual_hash = _sha256(parameter_path)
        if actual_hash != metadata.get("parameter_sha256"):
            problems.append(
                "parameter SHA-256 mismatch: "
                f"expected {metadata.get('parameter_sha256')}, got {actual_hash}"
            )
    for name in (
        "bin_count",
        "minimum_roi_size",
        "expected_feature_count",
        "expected_shape_feature_count",
        "expected_diagnostics_count",
    ):
        if isinstance(metadata.get(name), bool) or not isinstance(metadata.get(name), int):
            problems.append(f"{name} must be an integer")
    if metadata.get("minimum_roi_size") != MINIMUM_REGION_VOXELS:
        problems.append(
            f"minimum_roi_size must be {MINIMUM_REGION_VOXELS} for this release"
        )
    schema_hash = metadata.get("expected_feature_schema_sha256")
    if not isinstance(schema_hash, str) or len(schema_hash) != 64:
        problems.append("expected_feature_schema_sha256 must be a 64-character digest")
    if problems:
        raise ProtocolError(f"{metadata_path}: " + "; ".join(problems))
    return RoiPreset(
        preset_id=str(metadata["preset_id"]),
        variant=variant,
        status=status,
        parameter_path=parameter_path.resolve(),
        parameter_sha256=str(metadata["parameter_sha256"]),
        metadata_path=metadata_path.resolve(),
        metadata_sha256=_sha256(metadata_path),
        requires_preprocessed_input=True,
        bin_count=int(metadata["bin_count"]),
        minimum_roi_size=int(metadata["minimum_roi_size"]),
        expected_feature_count=int(metadata["expected_feature_count"]),
        expected_shape_feature_count=int(metadata["expected_shape_feature_count"]),
        expected_diagnostics_count=int(metadata["expected_diagnostics_count"]),
        expected_feature_schema_sha256=str(schema_hash),
    )


def _load_voxelwise_preset(
    metadata_path: Path, variant: str, status: str
) -> VoxelwisePreset:
    if not metadata_path.is_file():
        raise ProtocolError(f"preset metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read preset metadata {metadata_path}: {exc}") from exc
    required = {
        "schema_version",
        "preset_id",
        "variant",
        "status",
        "parameter_file",
        "parameter_sha256",
        "requires_preprocessed_input",
        "bin_count",
        "minimum_roi_size",
        "kernel_radius",
        "expected_map_count",
        "expected_map_schema_sha256",
        "expected_map_names",
    }
    missing = sorted(required - set(metadata))
    unknown = sorted(set(metadata) - required - {"resampling_note", "source"})
    problems: list[str] = []
    if missing:
        problems.append("missing key(s): " + ", ".join(missing))
    if unknown:
        problems.append("unknown key(s): " + ", ".join(unknown))
    if metadata.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if metadata.get("variant") != variant:
        problems.append(f"variant must be {variant!r}")
    if metadata.get("status") != status:
        problems.append(f"status must be {status!r}")
    if metadata.get("requires_preprocessed_input") is not True:
        problems.append("requires_preprocessed_input must be true")
    parameter_name = metadata.get("parameter_file")
    if not isinstance(parameter_name, str) or not parameter_name:
        problems.append("parameter_file must be a non-empty string")
        parameter_path = metadata_path.with_name("invalid")
    else:
        parameter_path = metadata_path.with_name(parameter_name)
    if not parameter_path.is_file():
        problems.append(f"parameter file is missing: {parameter_path}")
    else:
        actual_hash = _sha256(parameter_path)
        if actual_hash != metadata.get("parameter_sha256"):
            problems.append(
                "parameter SHA-256 mismatch: "
                f"expected {metadata.get('parameter_sha256')}, got {actual_hash}"
            )
    for name in ("bin_count", "minimum_roi_size", "kernel_radius", "expected_map_count"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            problems.append(f"{name} must be a positive integer")
    if metadata.get("minimum_roi_size") != MINIMUM_REGION_VOXELS:
        problems.append(
            f"minimum_roi_size must be {MINIMUM_REGION_VOXELS} for this release"
        )
    names_value = metadata.get("expected_map_names")
    if not isinstance(names_value, list) or not names_value or not all(
        isinstance(name, str) and name.startswith("original_") for name in names_value
    ):
        problems.append("expected_map_names must be a non-empty list of original_* names")
        names: tuple[str, ...] = ()
    else:
        names = tuple(names_value)
        if len(set(names)) != len(names):
            problems.append("expected_map_names contains duplicates")
    if isinstance(metadata.get("expected_map_count"), int) and len(names) != metadata.get(
        "expected_map_count"
    ):
        problems.append("expected_map_count does not match expected_map_names")
    schema_hash = metadata.get("expected_map_schema_sha256")
    if not isinstance(schema_hash, str) or len(schema_hash) != 64:
        problems.append("expected_map_schema_sha256 must be a 64-character digest")
    elif _schema_hash(names) != schema_hash:
        problems.append("expected_map_schema_sha256 does not match expected_map_names")
    if problems:
        raise ProtocolError(f"{metadata_path}: " + "; ".join(problems))
    return VoxelwisePreset(
        preset_id=str(metadata["preset_id"]),
        variant=variant,
        status=status,
        parameter_path=parameter_path.resolve(),
        parameter_sha256=str(metadata["parameter_sha256"]),
        metadata_path=metadata_path.resolve(),
        metadata_sha256=_sha256(metadata_path),
        requires_preprocessed_input=True,
        bin_count=int(metadata["bin_count"]),
        minimum_roi_size=int(metadata["minimum_roi_size"]),
        kernel_radius=int(metadata["kernel_radius"]),
        expected_map_count=int(metadata["expected_map_count"]),
        expected_map_schema_sha256=str(schema_hash),
        expected_map_names=names,
    )


def _schema_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
