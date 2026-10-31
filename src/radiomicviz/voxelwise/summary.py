"""Exploratory summaries of voxelwise radiomics extraction outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from radiomicviz._summary import (
    aggregate_fingerprint,
    append_counts,
    archive_existing,
    as_float,
    as_int,
    atomic_csv,
    atomic_text,
    canonical_output_root,
    fingerprint,
    format_duration,
    format_number,
    numeric_row,
    read_csv,
    require_columns,
    runtime_seconds,
)

FEATURE_FAMILIES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


class VoxelwiseSummaryError(RuntimeError):
    """Existing voxelwise outputs cannot produce a trustworthy summary."""


@dataclass(frozen=True)
class VoxelwiseSummaryResult:
    output_root: Path
    variant: str
    summary_dir: Path
    json_path: Path
    markdown_path: Path
    counts_path: Path
    numeric_path: Path
    subjects_path: Path
    maps_path: Path
    backups: tuple[Path, ...]
    total_units: int
    successful_units: int
    skipped_units: int
    failed_units: int
    map_count: int
    source_run_runtime_seconds: float | None


def summarize_voxelwise(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    summary_dir: str | Path | None = None,
    top_maps: int = 20,
) -> VoxelwiseSummaryResult:
    """Summarize existing voxelwise completions without reading NRRD payloads."""
    if top_maps < 1:
        raise VoxelwiseSummaryError("summary_top_maps must be at least 1")
    try:
        manifest, output_root = canonical_output_root(manifest_path)
        sources = {
            "checks": output_root / "qc" / f"voxelwise_{variant}_qc.csv",
            "errors": output_root / "qc" / f"voxelwise_{variant}_errors.csv",
            "run_metadata": output_root / "qc" / f"voxelwise_{variant}_run_metadata.json",
        }
        _require_sources(sources)
        checks, check_fields = read_csv(sources["checks"])
        errors, error_fields = read_csv(sources["errors"])
        metadata = json.loads(sources["run_metadata"].read_text(encoding="utf-8"))
        identity_fields = (
            "unit_key",
            "subject_id",
            "session_id",
            "region_id",
            "modality",
        )
        require_columns(check_fields, (*identity_fields, "status", "check_id"), "voxelwise QC")
        require_columns(error_fields, (*identity_fields, "error_code"), "voxelwise errors")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VoxelwiseSummaryError(str(exc)) from exc
    if metadata.get("stage") != "voxelwise" or metadata.get("variant") != variant:
        raise VoxelwiseSummaryError(
            "voxelwise run metadata does not match the requested stage and variant"
        )
    raw_units = metadata.get("unit_results")
    if not isinstance(raw_units, list) or not all(isinstance(unit, dict) for unit in raw_units):
        raise VoxelwiseSummaryError("voxelwise run metadata has no valid unit_results list")
    units: list[dict[str, object]] = [dict(unit) for unit in raw_units]

    identities = _identity_rows(checks, errors)
    completion_records: dict[str, dict[str, object]] = {}
    completion_fingerprints: list[dict[str, object]] = []
    invalid_completions: list[str] = []
    statuses: dict[str, str] = {}
    for unit in units:
        unit_key = str(unit.get("unit_key", ""))
        statuses[unit_key] = _unit_status(unit.get("status"))
        completion_value = unit.get("completion_path")
        if not completion_value:
            continue
        completion_path = Path(str(completion_value)).expanduser()
        if not completion_path.is_file():
            invalid_completions.append(unit_key)
            continue
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_completions.append(unit_key)
            continue
        if (
            completion.get("stage") != "voxelwise"
            or completion.get("variant") != variant
            or completion.get("unit_key") != unit_key
        ):
            invalid_completions.append(unit_key)
            continue
        completion_records[unit_key] = completion
        completion_fingerprints.append(fingerprint(completion_path))

    status_counts = Counter(statuses.values())
    modality_by_unit = {
        key: row.get("modality", "unknown") or "unknown"
        for key, row in identities.items()
    }
    modality_counts = Counter(modality_by_unit.get(key, "unknown") for key in statuses)
    count_rows: list[dict[str, object]] = []
    append_counts(count_rows, "unit_status", status_counts, len(units))
    append_counts(count_rows, "units_by_modality", modality_counts, len(units))
    modality_status = Counter(
        f"{modality_by_unit.get(key, 'unknown')}__{status}"
        for key, status in statuses.items()
    )
    append_counts(count_rows, "modality_status", modality_status, len(units))
    qc_counts = Counter(row.get("status", "") or "blank" for row in checks)
    append_counts(count_rows, "invocation_checks", qc_counts, len(checks))
    error_counts = Counter(row.get("error_code", "") or "blank" for row in errors)
    append_counts(count_rows, "error_code", error_counts, len(errors))

    successful_keys = {
        key for key, status in statuses.items() if status == "complete_or_resumed"
    }
    schema_names = [
        tuple(str(name) for name in completion.get("map_names", []))
        for key, completion in completion_records.items()
        if key in successful_keys and isinstance(completion.get("map_names"), list)
    ]
    map_names = tuple(sorted({name for schema in schema_names for name in schema}))
    family_counts = Counter(_map_parts(name)[1] for name in map_names)
    append_counts(count_rows, "map_schema_family", family_counts, len(map_names))

    observations: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    per_unit_min_fraction: list[float] = []
    per_unit_zero_maps: list[float] = []
    all_fractions: list[float] = []
    mask_by_status: dict[str, list[float]] = defaultdict(list)
    map_counts: list[float] = []
    for key, status in statuses.items():
        completion = completion_records.get(key)
        mask_voxels = as_float(
            completion.get("mask_voxels") if completion else _unit_value(units, key, "mask_voxels")
        )
        if mask_voxels is not None:
            mask_by_status[status].append(mask_voxels)
        if key not in successful_keys or completion is None:
            continue
        modality = modality_by_unit.get(key, "unknown")
        names = completion.get("map_names", [])
        finite_map = completion.get("map_finite_voxels", {})
        if not isinstance(names, list):
            names = []
        if not isinstance(finite_map, dict):
            finite_map = {}
        map_counts.append(float(len(names)))
        unit_fractions: list[float] = []
        zero_count = 0
        for name_value in names:
            name = str(name_value)
            finite = as_int(finite_map.get(name))
            fraction = (
                finite / mask_voxels
                if finite is not None and mask_voxels is not None and mask_voxels > 0
                else None
            )
            observations[(modality, name)].append(
                {
                    "unit_key": key,
                    "finite_voxels": finite,
                    "finite_fraction": fraction,
                }
            )
            if fraction is not None:
                unit_fractions.append(fraction)
                all_fractions.append(fraction)
            if finite == 0:
                zero_count += 1
        if unit_fractions:
            per_unit_min_fraction.append(min(unit_fractions))
        per_unit_zero_maps.append(float(zero_count))

    map_rows: list[dict[str, object]] = []
    successful_by_modality = Counter(
        modality_by_unit.get(key, "unknown") for key in successful_keys
    )
    for (modality, name), records in sorted(observations.items()):
        finite_values = [
            float(value)
            for record in records
            if (value := as_float(record.get("finite_voxels"))) is not None
        ]
        fractions = [
            value
            for record in records
            if (value := as_float(record.get("finite_fraction"))) is not None
        ]
        finite_stats = numeric_row(name, finite_values)
        fraction_stats = numeric_row(name, fractions)
        image_type, family = _map_parts(name)
        expected = successful_by_modality.get(modality, 0)
        coverage_records = len(fractions)
        map_rows.append(
            {
                "modality": modality,
                "image_type": image_type,
                "family": family,
                "map": name,
                "successful_units": expected,
                "coverage_record_units": coverage_records,
                "missing_coverage_units": max(0, expected - coverage_records),
                "zero_finite_units": sum(
                    as_int(record.get("finite_voxels")) == 0 for record in records
                ),
                "mean_finite_voxels": finite_stats["mean"],
                "min_finite_voxels": finite_stats["min"],
                "median_finite_voxels": finite_stats["median"],
                "max_finite_voxels": finite_stats["max"],
                "mean_finite_fraction": fraction_stats["mean"],
                "min_finite_fraction": fraction_stats["min"],
                "median_finite_fraction": fraction_stats["median"],
                "max_finite_fraction": fraction_stats["max"],
            }
        )

    top_low_coverage = sorted(
        map_rows,
        key=lambda row: (
            -int(row["missing_coverage_units"]),
            float(row["mean_finite_fraction"])
            if row["mean_finite_fraction"] is not None
            else float("inf"),
            str(row["modality"]),
            str(row["map"]),
        ),
    )[:top_maps]
    numeric_rows = [
        numeric_row("mask_voxels_all_reported_units", [v for values in mask_by_status.values() for v in values]),
        *(
            numeric_row(f"mask_voxels__{status}", values)
            for status, values in sorted(mask_by_status.items())
        ),
        numeric_row("maps_per_successful_unit", map_counts),
        numeric_row("finite_fraction_all_map_unit_pairs", all_fractions),
        numeric_row("minimum_map_finite_fraction_per_unit", per_unit_min_fraction),
        numeric_row("zero_finite_maps_per_unit", per_unit_zero_maps),
    ]
    subject_rows = _subject_rows(statuses, identities)
    run_seconds = runtime_seconds(metadata)
    successful_completion_count = sum(key in completion_records for key in successful_keys)
    successful_coverage_count = sum(
        _has_complete_finite_map_record(completion_records.get(key))
        for key in successful_keys
    )
    unique_schemas = set(schema_names)
    validations = {
        "metadata_unit_results_equal_source_run_units": len(units) == metadata.get("units"),
        "error_rows_equal_source_run_failed": len(errors) == metadata.get("failed"),
        "all_nonfailed_units_have_readable_completion_records": (
            len(completion_records)
            == status_counts.get("complete_or_resumed", 0) + status_counts.get("skipped", 0)
        ),
        "successful_map_schema_is_consistent": len(unique_schemas) <= 1,
        "successful_units_have_readable_completion_records": successful_completion_count
        == len(successful_keys),
        "successful_units_have_finite_map_coverage_records": successful_coverage_count
        == len(successful_keys),
    }
    summary = {
        "schema_version": 1,
        "stage": "voxelwise_summary",
        "variant": variant,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "source_files": {name: fingerprint(path) for name, path in sources.items()},
        "completion_records": {
            "count": len(completion_fingerprints),
            "aggregate_sha256": aggregate_fingerprint(completion_fingerprints),
            "invalid_or_missing_unit_keys": sorted(invalid_completions),
        },
        "source_run": _source_run(metadata),
        "source_run_runtime_seconds": run_seconds,
        "cohort": {
            "units": len(units),
            "successful_units": status_counts.get("complete_or_resumed", 0),
            "skipped_units": status_counts.get("skipped", 0),
            "failed_or_blocked_units": status_counts.get("failed_or_blocked", 0),
            "subjects": len({row.get("subject_id", "") for row in identities.values()}),
            "sessions": len(
                {
                    (row.get("subject_id", ""), row.get("session_id", ""))
                    for row in identities.values()
                }
            ),
            "regions": len(
                {
                    (
                        row.get("subject_id", ""),
                        row.get("session_id", ""),
                        row.get("region_id", ""),
                    )
                    for row in identities.values()
                }
            ),
            "modalities": sorted(set(modality_by_unit.values())),
            "map_count": len(map_names),
        },
        "validations": validations,
        "counts": count_rows,
        "numeric_summaries": numeric_rows,
        "subjects": subject_rows,
        "map_coverage": map_rows,
        "lowest_coverage_maps": top_low_coverage,
    }

    table_dir = output_root / "tables" / "voxelwise" / variant
    destination = (
        Path(summary_dir).expanduser().resolve()
        if summary_dir is not None
        else (table_dir / "summary").resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "json": destination / "voxelwise_summary.json",
        "markdown": destination / "voxelwise_summary.md",
        "counts": destination / "voxelwise_summary_counts.csv",
        "numeric": destination / "voxelwise_summary_numeric.csv",
        "subjects": destination / "voxelwise_summary_by_subject.csv",
        "maps": destination / "voxelwise_summary_map_coverage.csv",
    }
    backups = archive_existing(targets.values())
    atomic_text(targets["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(targets["markdown"], _markdown(summary, top_maps))
    atomic_csv(
        targets["counts"], count_rows, ("group", "category", "count", "denominator", "percent")
    )
    atomic_csv(
        targets["numeric"],
        numeric_rows,
        ("metric", "n", "mean", "std", "min", "q25", "median", "q75", "max"),
    )
    atomic_csv(
        targets["subjects"],
        subject_rows,
        (
            "subject_id",
            "session_id",
            "regions",
            "total_units",
            "successful_units",
            "skipped_units",
            "failed_or_blocked_units",
            "successful_T1_units",
            "successful_FLAIR_units",
        ),
    )
    atomic_csv(
        targets["maps"],
        map_rows,
        (
            "modality",
            "image_type",
            "family",
            "map",
            "successful_units",
            "coverage_record_units",
            "missing_coverage_units",
            "zero_finite_units",
            "mean_finite_voxels",
            "min_finite_voxels",
            "median_finite_voxels",
            "max_finite_voxels",
            "mean_finite_fraction",
            "min_finite_fraction",
            "median_finite_fraction",
            "max_finite_fraction",
        ),
    )
    return VoxelwiseSummaryResult(
        output_root=output_root,
        variant=variant,
        summary_dir=destination,
        json_path=targets["json"],
        markdown_path=targets["markdown"],
        counts_path=targets["counts"],
        numeric_path=targets["numeric"],
        subjects_path=targets["subjects"],
        maps_path=targets["maps"],
        backups=backups,
        total_units=len(units),
        successful_units=status_counts.get("complete_or_resumed", 0),
        skipped_units=status_counts.get("skipped", 0),
        failed_units=status_counts.get("failed_or_blocked", 0),
        map_count=len(map_names),
        source_run_runtime_seconds=run_seconds,
    )


def _require_sources(sources: Mapping[str, Path]) -> None:
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "required voxelwise output(s) are missing:\n  - " + "\n  - ".join(missing)
        )


def _identity_rows(
    checks: list[dict[str, str]], errors: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for row in (*checks, *errors):
        key = row.get("unit_key", "")
        if key and key not in identities:
            identities[key] = {
                name: row.get(name, "")
                for name in ("subject_id", "session_id", "region_id", "modality")
            }
    return identities


def _unit_status(value: object) -> str:
    normalized = str(value or "").lower()
    if normalized in {"complete", "resumed"}:
        return "complete_or_resumed"
    if normalized in {"skipped", "resumed-skipped"}:
        return "skipped"
    return "failed_or_blocked"


def _unit_value(units: list[dict[str, object]], key: str, name: str) -> object:
    return next((unit.get(name) for unit in units if unit.get("unit_key") == key), None)


def _map_parts(name: str) -> tuple[str, str]:
    for family in FEATURE_FAMILIES:
        marker = f"_{family}_"
        if marker in name:
            return name.split(marker, 1)[0], family
    return "unknown", "unknown"


def _has_complete_finite_map_record(completion: Mapping[str, object] | None) -> bool:
    if completion is None:
        return False
    names = completion.get("map_names")
    finite = completion.get("map_finite_voxels")
    return (
        isinstance(names, list)
        and isinstance(finite, dict)
        and all(name in finite for name in names)
    )


def _subject_rows(
    statuses: Mapping[str, str], identities: Mapping[str, Mapping[str, str]]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, status in statuses.items():
        identity = identities.get(key, {})
        groups[(identity.get("subject_id", "unknown"), identity.get("session_id", "unknown"))].append(
            (key, status)
        )
    rows: list[dict[str, object]] = []
    for (subject, session), units in sorted(groups.items()):
        status_counts = Counter(status for _key, status in units)
        rows.append(
            {
                "subject_id": subject,
                "session_id": session,
                "regions": len(
                    {
                        identities.get(key, {}).get("region_id", "unknown")
                        for key, _status in units
                    }
                ),
                "total_units": len(units),
                "successful_units": status_counts.get("complete_or_resumed", 0),
                "skipped_units": status_counts.get("skipped", 0),
                "failed_or_blocked_units": status_counts.get("failed_or_blocked", 0),
                "successful_T1_units": sum(
                    identities.get(key, {}).get("modality") == "T1"
                    and status == "complete_or_resumed"
                    for key, status in units
                ),
                "successful_FLAIR_units": sum(
                    identities.get(key, {}).get("modality") == "FLAIR"
                    and status == "complete_or_resumed"
                    for key, status in units
                ),
            }
        )
    return rows


def _source_run(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "status",
        "started_utc",
        "finished_utc",
        "protocol_id",
        "protocol_version",
        "protocol_sha256",
        "preset_id",
        "preset_sha256",
        "reference_compatible",
        "reference_deviations",
        "jobs",
        "units",
        "completed",
        "resumed",
        "skipped",
        "failed",
        "warnings",
    )
    return {key: metadata.get(key) for key in keys}


def _markdown(summary: Mapping[str, object], top_maps: int) -> str:
    cohort = summary["cohort"]
    source_run = summary["source_run"]
    validations = summary["validations"]
    counts = summary["counts"]
    numeric = summary["numeric_summaries"]
    maps = summary["lowest_coverage_maps"]
    subjects = summary["subjects"]
    lines = [
        f"# Voxelwise radiomics summary — {summary['variant']}",
        "",
        f"Generated: `{summary['generated_utc']}`  ",
        f"Source run finished: `{source_run.get('finished_utc')}`  ",
        f"Reference compatible: `{source_run.get('reference_compatible')}`  ",
        f"Runtime: `{format_duration(summary.get('source_run_runtime_seconds'))}`",
        "",
        "## Cohort",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        *[f"| {key.replace('_', ' ')} | {value} |" for key, value in cohort.items()],
        "",
        "## Integrity checks",
        "",
        "| Check | Passed |",
        "| --- | --- |",
        *[f"| {key.replace('_', ' ')} | {value} |" for key, value in validations.items()],
        "",
        "## Unit status",
        "",
        "| Category | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['category']} | {row['count']} | {row['percent']:.2f}% |"
        for row in counts
        if row["group"] == "unit_status"
    )
    lines.extend(
        [
            "",
            "## Numeric summaries",
            "",
            "| Metric | N | Mean | SD | Min | Q25 | Median | Q75 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        "| {metric} | {n} | {mean} | {std} | {min} | {q25} | {median} | {q75} | {max} |".format(
            **{key: format_number(value) for key, value in row.items()}
        )
        for row in numeric
    )
    lines.extend(
        [
            "",
            f"## Lowest-coverage {top_maps} maps",
            "",
            "Coverage is the recorded finite-map voxel count divided by lesion voxels.",
            "",
            "| Modality | Family | Map | Missing records | Zero finite | Mean fraction | Minimum fraction |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['modality']} | {row['family']} | {row['map']} | "
        f"{row['missing_coverage_units']} | {row['zero_finite_units']} | "
        f"{format_number(row['mean_finite_fraction'])} | "
        f"{format_number(row['min_finite_fraction'])} |"
        for row in maps
    )
    lines.extend(
        [
            "",
            "## By subject/session",
            "",
            "| Subject | Session | Regions | Units | Successful | Skipped | Failed/blocked | T1 successful | FLAIR successful |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['subject_id']} | {row['session_id']} | {row['regions']} | "
        f"{row['total_units']} | {row['successful_units']} | {row['skipped_units']} | "
        f"{row['failed_or_blocked_units']} | {row['successful_T1_units']} | "
        f"{row['successful_FLAIR_units']} |"
        for row in subjects
    )
    return "\n".join(lines).rstrip() + "\n"
