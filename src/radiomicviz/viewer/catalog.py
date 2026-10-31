"""Read-only discovery of one RadiomicsViz output tree for the viewer.

This is the only viewer module that knows where the pipeline stages write their outputs.
It never writes, never opens staging (``.tmp``), archive (``.bak-*``) or lock files, and
tolerates files that appear or disappear while it scans. Output paths are derived from
the canonical manifest exactly as the stages derive them; tests compare them with the
stages' own path helpers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat as stat_module
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from radiomicviz._summary import canonical_output_root

STAGES = ("whole", "voxelwise", "habitats", "habitat_features")
STAGE_COMMANDS = {
    "whole": "whole",
    "voxelwise": "voxelwise",
    "habitats": "habitats",
    "habitat_features": "habitat-features",
}
HABITAT_METHODS = ("fixed_k2", "adaptive_k1to3", "geometric_1mm")
HABITAT_MODALITIES = ("T1", "FLAIR")

COMPLETE = "complete"
SKIPPED = "skipped"
FAILED = "failed"
BLOCKED = "blocked"
IN_PROGRESS = "in-progress"
MISSING = "missing"
NOT_APPLICABLE = "not-applicable"
# Worst first: a region is reported with the worst status of its units.
STATUS_ORDER = (FAILED, BLOCKED, IN_PROGRESS, MISSING, SKIPPED, COMPLETE, NOT_APPLICABLE)

_TABLE_LIMIT_BYTES = 512 * 1024 * 1024
_JSON_LIMIT_BYTES = 16 * 1024 * 1024


class CatalogError(Exception):
    """The output tree cannot be opened; the message never contains an absolute path."""


# ----------------------------------------------------------------------------- layout
def region_dir(output_root: Path, subject: str, session: str, region: str) -> Path:
    return (
        output_root / "subjects" / _part(subject) / "sessions" / _part(session)
        / "regions" / _part(region)
    )


def materialized_session_dir(output_root: Path, subject: str, session: str, region_type: str) -> Path:
    """Materialized masks of one session; provenance lives in its ``.materialization`` folder."""
    return output_root / "materialized_regions" / _part(subject) / _part(session) / _part(region_type)


def whole_unit_dir(output_root: Path, key: tuple[str, str, str], modality: str, variant: str) -> Path:
    return region_dir(output_root, *key) / "radiomics" / "whole" / _part(variant) / _part(modality)


def voxelwise_unit_dir(
    output_root: Path, key: tuple[str, str, str], modality: str, variant: str
) -> Path:
    return region_dir(output_root, *key) / "radiomics" / "voxelwise" / _part(variant) / _part(modality)


def habitat_mask_path(
    output_root: Path, key: tuple[str, str, str], method: str, variant: str, *, source_grid: bool
) -> Path:
    base = region_dir(output_root, *key) / "habitats" / _part(method)
    if method != "geometric_1mm":
        base = base / _part(variant)
    return base / ("mask_source_grid.nii.gz" if source_grid else "mask.nii.gz")


def habitat_completion_path(output_root: Path, key: tuple[str, str, str], variant: str) -> Path:
    return region_dir(output_root, *key) / "habitats" / "reference" / _part(variant) / "completion.json"


def habitat_feature_unit_dir(
    output_root: Path, key: tuple[str, str, str], method: str, modality: str, variant: str
) -> Path:
    return (
        region_dir(output_root, *key) / "habitat_features" / _part(method) / _part(variant)
        / _part(modality)
    )


def stage_report_paths(output_root: Path, stage: str, variant: str) -> dict[str, Path]:
    """QC, errors, metadata and log of one stage invocation (never created here)."""
    qc = output_root / "qc"
    return {
        "qc": qc / f"{stage}_{variant}_qc.csv",
        "errors": qc / f"{stage}_{variant}_errors.csv",
        "run_metadata": qc / f"{stage}_{variant}_run_metadata.json",
        "log": qc / "logs" / f"{stage}_{variant}.log",
    }


def stage_table_paths(output_root: Path, stage: str, variant: str) -> dict[str, Path]:
    """Cohort tables written by a stage (never created here)."""
    if stage == "whole":
        base = output_root / "tables" / "whole" / variant
        return {
            "whole_region": base / "whole_region.csv",
            "whole_region_reference_format": base / "whole_region_reference_format.csv",
        }
    if stage == "habitats":
        base = output_root / "tables" / "habitats" / variant
        return {"habitat_index": base / "habitat_index.csv", "lesion_qc": base / "lesion_qc.csv"}
    if stage == "habitat_features":
        base = output_root / "tables" / "habitat_features" / variant
        tables = {"habitat_features": base / "habitat_features.csv"}
        for method in HABITAT_METHODS:
            tables[f"habitat_features_{method}"] = base / f"habitat_features_{method}.csv"
        for method in HABITAT_METHODS:
            for modality in HABITAT_MODALITIES:
                name = f"habitat_features_{method}_{modality}_reference_format"
                tables[name] = base / f"{name}.csv"
        return tables
    return {}


def summary_dir(output_root: Path, stage: str, variant: str) -> Path:
    return output_root / "tables" / stage / variant / "summary"


def unit_key(*parts: str) -> str:
    return "__".join(parts)


def staging_names(stage: str, key: str) -> tuple[str, ...]:
    """Name prefixes the stage uses for a unit's staging folder under ``.tmp/<stage>``."""
    if stage == "habitats":
        return (f"{key}.",)
    return (f"u-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}-",)


def _part(value: str) -> str:
    """Reject identifiers that are not a single, plain path component."""
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or any(separator in text for separator in ("/", "\\", "\x00"))
        or (len(text) >= 2 and text[1] == ":")
    ):
        raise CatalogError("identifier is not a plain folder name")
    return text


def _inside(path: Path, root: Path, *, root_resolved: bool = False) -> bool:
    """True when ``path`` resolves (following links and junctions) inside ``root``."""
    try:
        target = path.resolve(strict=False)
        base = root if root_resolved else root.resolve(strict=False)
        target.relative_to(base)
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def _is_link(info: os.stat_result) -> bool:
    reparse = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat_module.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & reparse)


def stable_id(*parts: object, length: int = 12) -> str:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


# ----------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class ManifestEntry:
    modality: str
    image_path: str
    mask_path: str
    label: int | None
    image_sha256: str
    mask_sha256: str
    space_id: str


@dataclass
class UnitState:
    stage: str
    modality: str | None
    method: str | None
    unit_key: str
    status: str
    reason: str = ""
    reference_status: str = ""
    completion: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    completion_path: Path | None = None


@dataclass
class Region:
    rid: str
    subject: str
    session: str
    region: str
    region_type: str
    entries: dict[str, ManifestEntry] = field(default_factory=dict)
    units: dict[str, list[UnitState]] = field(default_factory=dict)
    voxels: int | None = None
    checks: list[dict[str, str]] = field(default_factory=list)
    unsafe: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.session, self.region)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def stage_status(self, stage: str) -> tuple[str, str]:
        units = self.units.get(stage, [])
        if not units:
            return NOT_APPLICABLE, ""
        worst = min(units, key=lambda unit: STATUS_ORDER.index(unit.status))
        return worst.status, worst.reason

    def reference_statuses(self) -> set[str]:
        return {
            unit.reference_status for units in self.units.values() for unit in units
            if unit.reference_status
        }


@dataclass
class Snapshot:
    manifest_path: Path
    output_root: Path
    variant: str
    data_root: Path | None
    manifest_metadata: Mapping[str, Any]
    regions: dict[str, Region]
    order: list[str]
    stage_metadata: dict[str, Mapping[str, Any] | None]
    stage_errors: dict[str, list[dict[str, str]]]
    stage_checks: dict[str, list[dict[str, str]]]
    input_checks: list[dict[str, str]]
    manifest_errors: list[dict[str, str]]
    staging_active: dict[str, bool]
    signature: tuple[tuple[str, int, int], ...]
    problems: list[str]


# ----------------------------------------------------------------------------- catalog
class Catalog:
    """Thread-safe, rescannable view of one output tree. Never writes."""

    def __init__(self, manifest_path: str | Path, *, variant: str = "b32") -> None:
        try:
            manifest, output_root = canonical_output_root(manifest_path)
        except ValueError as exc:
            raise CatalogError(_safe_message(str(exc))) from None
        self.manifest_path = manifest
        self.output_root = output_root
        self.variant = _part(variant)
        self._root_resolved = output_root.resolve(strict=False)
        self._component_ok: dict[Path, bool] = {}
        self._index_root = str(output_root / "subjects") + os.sep
        self._index_files: dict[str, tuple[int, int]] = {}
        self._index_dirs: set[str] = set()
        self._lock = threading.Lock()
        self._json_cache: dict[Path, tuple[tuple[int, int], Any]] = {}
        self._snapshot = self._scan()

    # -- public
    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    def rescan(self) -> Snapshot:
        snapshot = self._scan()
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def current_signature(self) -> tuple[tuple[str, int, int], ...]:
        return self._signature(self._signature_paths())

    def changed_since_scan(self) -> bool:
        return self.current_signature() != self._snapshot.signature

    def region(self, rid: str) -> Region | None:
        return self._snapshot.regions.get(rid)

    def inside_roots(self, path: Path) -> bool:
        """Whether a file may be served: inside output_root, or inside data_root."""
        snapshot = self._snapshot
        if _inside(path, self._root_resolved, root_resolved=True):
            return True
        return snapshot.data_root is not None and _inside(path, snapshot.data_root)

    def contained(self, path: Path) -> bool:
        """Whether ``path`` exists inside output_root without leaving it through a link.

        Lexically derived paths are checked component by component with ``lstat``; only a
        component that is a symbolic link or junction is fully resolved. Verified folders
        are cached for the current scan.
        """
        text = str(path)
        if text.startswith(self._index_root):
            return text in self._index_files or text in self._index_dirs
        try:
            relative = path.relative_to(self.output_root)
        except ValueError:
            return path.exists() and _inside(path, self._root_resolved, root_resolved=True)
        if any(part in {".", ".."} for part in relative.parts):
            return path.exists() and _inside(path, self._root_resolved, root_resolved=True)
        current = self.output_root
        for part in relative.parts:
            current = current / part
            verdict = self._component_ok.get(current)
            if verdict is not None:
                if not verdict:
                    return False
                continue
            try:
                info = os.lstat(current)
            except OSError:
                return False
            if _is_link(info):
                verdict = _inside(current, self._root_resolved, root_resolved=True)
                self._component_ok[current] = verdict
                if not verdict:
                    return False
            elif current != path:
                self._component_ok[current] = True
        return True

    def read_json(self, path: Path) -> Any:
        """Read a small JSON file inside output_root, cached by size and mtime."""
        if _ignored_name(path) or not self.contained(path):
            return None
        signature = self._index_files.get(str(path))
        if signature is None:
            try:
                stat = path.stat()
            except OSError:
                return None
            signature = (stat.st_mtime_ns, stat.st_size)
        if signature[1] > _JSON_LIMIT_BYTES:
            return None
        cached = self._json_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            value = None
        self._json_cache[path] = (signature, value)
        return value

    def read_csv_rows(self, path: Path) -> tuple[list[dict[str, str]] | None, tuple[str, ...]]:
        """Read a CSV only when its resolved path remains inside ``output_root``."""
        if not self.contained(path):
            return None, ()
        return read_csv_rows(path)

    # -- scanning
    def _signature_paths(self) -> list[Path]:
        paths = [self.manifest_path, self.manifest_path.with_name("run_metadata.json")]
        for stage in STAGES:
            reports = stage_report_paths(self.output_root, stage, self.variant)
            paths.extend((reports["run_metadata"], reports["errors"], reports["qc"]))
        return paths

    @staticmethod
    def _signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
        result = []
        for path in paths:
            try:
                stat = path.stat()
                result.append((path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                result.append((path.name, 0, -1))
        return tuple(result)

    def _build_index(self) -> None:
        """One scandir pass over ``subjects/``: existing files and folders, never through links.

        Directory entries carry stat data on Windows and file types on POSIX, so the pass
        avoids one system call per checked path. Staging, archive and lock names are skipped.
        """
        files: dict[str, tuple[int, int]] = {}
        folders: set[str] = set()
        stack = [str(self.output_root / "subjects")]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        name = entry.name
                        if name.startswith(".") or ".bak-" in name or name.endswith((".tmp", ".lock")):
                            continue
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if _is_link(info):
                            continue
                        if stat_module.S_ISDIR(info.st_mode):
                            folders.add(entry.path)
                            stack.append(entry.path)
                        else:
                            files[entry.path] = (info.st_mtime_ns, info.st_size)
            except OSError:
                continue
        self._index_files, self._index_dirs = files, folders

    def _exists(self, path: Path) -> bool:
        text = str(path)
        if text.startswith(self._index_root):
            return text in self._index_files or text in self._index_dirs
        return path.exists()

    def _scan(self) -> Snapshot:
        self._component_ok = {}
        self._build_index()
        problems: list[str] = []
        signature = self._signature(self._signature_paths())
        rows, fields = read_csv_rows(self.manifest_path)
        if rows is None:
            raise CatalogError("the manifest cannot be read as CSV")
        required = {"subject_id", "session_id", "region_id", "modality", "image_path", "mask_path"}
        missing = sorted(required - set(fields))
        if missing:
            raise CatalogError("the manifest is missing column(s): " + ", ".join(missing))
        manifest_metadata = self.read_json(self.manifest_path.with_name("run_metadata.json")) or {}
        data_root = None
        if isinstance(manifest_metadata.get("data_root"), str) and manifest_metadata["data_root"]:
            data_root = Path(manifest_metadata["data_root"])

        regions: dict[str, Region] = {}
        by_key: dict[tuple[str, str, str], Region] = {}
        for row in rows:
            key = (row.get("subject_id", ""), row.get("session_id", ""), row.get("region_id", ""))
            region = by_key.get(key)
            if region is None:
                rid = stable_id(*key)
                length = 12
                while rid in regions and regions[rid].key != key:
                    length += 4
                    rid = stable_id(*key, length=length)
                region = Region(rid, *key, region_type=row.get("region_type", "") or "region")
                region.unsafe = not all(_is_plain(part) for part in key)
                regions[rid] = region
                by_key[key] = region
            modality = row.get("modality", "")
            region.entries[modality] = ManifestEntry(
                modality=modality,
                image_path=row.get("image_path", ""),
                mask_path=row.get("mask_path", ""),
                label=_int(row.get("label")),
                image_sha256=row.get("image_sha256", ""),
                mask_sha256=row.get("mask_sha256", ""),
                space_id=row.get("space_id", ""),
            )
        order = sorted(regions, key=lambda rid: regions[rid].key)

        input_checks, _ = self.read_csv_rows(self.manifest_path.with_name("input_checks.csv"))
        manifest_errors, _ = self.read_csv_rows(self.manifest_path.with_name("errors.csv"))
        for check in input_checks or []:
            region = by_key.get(
                (check.get("subject_id", ""), check.get("session_id", ""), check.get("region_id", ""))
            )
            if region is None:
                continue
            region.checks.append(check)
            if check.get("check_id") == "region_voxel_count":
                region.voxels = _int(check.get("measured_value"))

        stage_metadata: dict[str, Mapping[str, Any] | None] = {}
        stage_errors: dict[str, list[dict[str, str]]] = {}
        stage_checks: dict[str, list[dict[str, str]]] = {}
        staging_active: dict[str, bool] = {}
        for stage in STAGES:
            reports = stage_report_paths(self.output_root, stage, self.variant)
            metadata = self.read_json(reports["run_metadata"])
            stage_metadata[stage] = metadata if isinstance(metadata, Mapping) else None
            stage_errors[stage] = self.read_csv_rows(reports["errors"])[0] or []
            stage_checks[stage] = self.read_csv_rows(reports["qc"])[0] or []
            staging = self._staging_entries(stage)
            staging_active[stage] = bool(staging)
            errors_by_unit = _rows_by_unit(stage_errors[stage])
            for rid in order:
                region = regions[rid]
                if region.unsafe:
                    region.units[stage] = [
                        UnitState(stage, None, None, "", BLOCKED, "unsafe identifier in the manifest")
                    ]
                    continue
                region.units[stage] = self._stage_units(region, stage, errors_by_unit, staging)
        return Snapshot(
            manifest_path=self.manifest_path,
            output_root=self.output_root,
            variant=self.variant,
            data_root=data_root,
            manifest_metadata=manifest_metadata,
            regions=regions,
            order=order,
            stage_metadata=stage_metadata,
            stage_errors=stage_errors,
            stage_checks=stage_checks,
            input_checks=input_checks or [],
            manifest_errors=manifest_errors or [],
            staging_active=staging_active,
            signature=signature,
            problems=problems,
        )

    def _staging_entries(self, stage: str) -> tuple[str, ...]:
        """Names (only) of staging folders; nothing under .tmp is ever opened."""
        root = self.output_root / ".tmp" / stage
        try:
            with os.scandir(root) as entries:
                return tuple(entry.name for entry in entries)
        except OSError:
            return ()

    def _stage_units(
        self,
        region: Region,
        stage: str,
        errors_by_unit: Mapping[str, list[dict[str, str]]],
        staging: tuple[str, ...],
    ) -> list[UnitState]:
        key = region.key
        variant = self.variant
        units: list[UnitState] = []
        if stage in {"whole", "voxelwise"}:
            for modality in region.modalities:
                directory = (
                    whole_unit_dir if stage == "whole" else voxelwise_unit_dir
                )(self.output_root, key, modality, variant)
                units.append(
                    self._unit_state(
                        stage, unit_key(*key, modality), directory / "completion.json",
                        directory, errors_by_unit, staging, modality=modality,
                    )
                )
            return units
        if not all(modality in region.entries for modality in HABITAT_MODALITIES):
            return [
                UnitState(
                    stage, None, None, unit_key(*key), NOT_APPLICABLE,
                    "habitats need both T1 and FLAIR modalities",
                )
            ]
        if stage == "habitats":
            completion = habitat_completion_path(self.output_root, key, variant)
            directory = region_dir(self.output_root, *key) / "habitats"
            return [
                self._unit_state(
                    stage, unit_key(*key), completion, directory, errors_by_unit, staging,
                    output_marker=completion.parent,
                )
            ]
        for method in HABITAT_METHODS:
            for modality in HABITAT_MODALITIES:
                directory = habitat_feature_unit_dir(
                    self.output_root, key, method, modality, variant
                )
                units.append(
                    self._unit_state(
                        stage, unit_key(*key, method, modality), directory / "completion.json",
                        directory, errors_by_unit, staging, modality=modality, method=method,
                    )
                )
        return units

    def _unit_state(
        self,
        stage: str,
        key: str,
        completion_path: Path,
        output_dir: Path,
        errors_by_unit: Mapping[str, list[dict[str, str]]],
        staging: tuple[str, ...],
        *,
        modality: str | None = None,
        method: str | None = None,
        output_marker: Path | None = None,
    ) -> UnitState:
        state = UnitState(
            stage, modality, method, key, MISSING,
            output_dir=output_dir, completion_path=completion_path,
        )
        completion = self.read_json(completion_path)
        if isinstance(completion, Mapping):
            state.completion = completion
            state.reference_status = str(completion.get("reference_status", ""))
            status = str(completion.get("status", ""))
            if status == "skipped":
                state.status = SKIPPED
                state.reason = str(
                    completion.get("skip_reason") or completion.get("reason")
                    or completion.get("skip_code") or "skipped"
                )
            elif status in {"complete", "complete_with_failures"}:
                state.status = COMPLETE
                if status == "complete_with_failures":
                    state.reason = "completed with failed habitat zones"
            else:
                state.status = BLOCKED
                state.reason = f"unrecognised completion status {status!r}"
        errors = errors_by_unit.get(key, [])
        if errors:
            codes = {row.get("error_code", "") for row in errors}
            state.status = BLOCKED if any(code.endswith("_blocked") for code in codes) else FAILED
            state.reason = "; ".join(
                sorted({f"{row.get('error_code', '')}: {row.get('message', '')}" for row in errors})
            )
            return state
        if state.completion is not None:
            return state
        prefixes = staging_names(stage, key)
        if any(name.startswith(prefixes) for name in staging):
            state.status = IN_PROGRESS
            state.reason = "a staging folder for this unit exists"
            return state
        marker = output_marker or output_dir
        if self._exists(marker):
            state.status = BLOCKED
            state.reason = "outputs exist without a completion record"
        return state


# ----------------------------------------------------------------------------- helpers
def read_csv_rows(path: Path) -> tuple[list[dict[str, str]] | None, tuple[str, ...]]:
    """Read a CSV as dicts; ``(None, ())`` when it is missing, too large or unreadable."""
    if _ignored_name(path):
        return None, ()
    try:
        if path.stat().st_size > _TABLE_LIMIT_BYTES:
            return None, ()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            return rows, tuple(reader.fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error):
        return None, ()


def _rows_by_unit(rows: Iterable[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = row.get("unit_key", "")
        if key:
            result.setdefault(key, []).append(dict(row))
    return result


def _ignored_name(path: Path) -> bool:
    name = path.name
    return (
        name.startswith(".") and name != ".materialization"
        or ".bak-" in name
        or name.endswith((".tmp", ".lock"))
        or any(part == ".tmp" or ".bak-" in part for part in path.parts)
    )


def _is_plain(value: str) -> bool:
    try:
        _part(value)
    except CatalogError:
        return False
    return True


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _safe_message(message: str) -> str:
    """Drop anything after a colon that may hold an absolute path."""
    head = message.split(":", 1)[0]
    return head.strip() or "invalid manifest"
