"""Exploratory summaries of completed habitat-clustering outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping
from uuid import uuid4

import numpy as np

EXPECTED_METHOD_ROWS = {
    "fixed_k2": 2,
    "adaptive_k1to3": 3,
    "geometric_1mm": 2,
}
KEY_FIELDS = ("subject_id", "session_id", "region_id")


class HabitatSummaryError(RuntimeError):
    """Existing habitat outputs cannot produce a trustworthy summary."""


@dataclass(frozen=True)
class HabitatSummaryResult:
    output_root: Path
    variant: str
    summary_dir: Path
    json_path: Path
    markdown_path: Path
    counts_path: Path
    numeric_path: Path
    subjects_path: Path
    features_path: Path
    backups: tuple[Path, ...]
    total_regions: int
    completed_regions: int
    skipped_regions: int
    failed_units: int
    fixed_valid_regions: int
    adaptive_split_regions: int
    source_run_runtime_seconds: float | None


def summarize_habitats(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    summary_dir: str | Path | None = None,
    top_features: int = 20,
) -> HabitatSummaryResult:
    """Summarize the latest habitat cohort tables without changing stage outputs."""
    if top_features < 1:
        raise HabitatSummaryError("summary_top_features must be at least 1")
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.name != "regions_manifest.csv" or manifest.parent.name != "manifest":
        raise HabitatSummaryError(
            "expected the canonical <output_root>/manifest/regions_manifest.csv path"
        )
    if not manifest.is_file():
        raise HabitatSummaryError(f"manifest does not exist: {manifest}")
    output_root = manifest.parent.parent.resolve()
    table_dir = output_root / "tables" / "habitats" / variant
    qc_dir = output_root / "qc"
    sources = {
        "habitat_index": table_dir / "habitat_index.csv",
        "lesion_qc": table_dir / "lesion_qc.csv",
        "checks": qc_dir / f"habitats_{variant}_qc.csv",
        "errors": qc_dir / f"habitats_{variant}_errors.csv",
        "run_metadata": qc_dir / f"habitats_{variant}_run_metadata.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise HabitatSummaryError(
            "required habitat output(s) are missing:\n  - " + "\n  - ".join(missing)
        )

    index = _read_csv(sources["habitat_index"])
    lesion_qc = _read_csv(sources["lesion_qc"])
    checks = _read_csv(sources["checks"])
    errors = _read_csv(sources["errors"])
    try:
        run_metadata = json.loads(sources["run_metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HabitatSummaryError(f"cannot read habitat run metadata: {exc}") from exc
    if run_metadata.get("stage") != "habitats" or run_metadata.get("variant") != variant:
        raise HabitatSummaryError(
            "habitat run metadata does not match the requested stage and variant"
        )

    _require_columns(
        index,
        {
            *KEY_FIELDS,
            "method",
            "zone_id",
            "n_voxels",
            "missing",
            "eligible_for_split",
            "chosen_k",
            "habitat_valid",
        },
        "habitat_index.csv",
        sources["habitat_index"],
    )
    _require_columns(
        lesion_qc,
        {*KEY_FIELDS, "n_voxels", "representation_status"},
        "lesion_qc.csv",
        sources["lesion_qc"],
    )

    index_by_region = _group_rows(index, KEY_FIELDS)
    qc_by_region = _unique_rows(lesion_qc, KEY_FIELDS, "lesion_qc.csv")
    completed_keys = set(index_by_region)
    qc_keys = set(qc_by_region)
    total_regions = len(qc_keys)
    completed_regions = len(completed_keys)
    skipped_keys = {
        key
        for key, row in qc_by_region.items()
        if row.get("representation_status") == "not_available_voxelwise_stage_skipped"
    }
    skipped_regions = len(skipped_keys)

    rows_per_region = {key: len(rows) for key, rows in index_by_region.items()}
    method_layout_ok = all(
        Counter(row["method"] for row in rows) == EXPECTED_METHOD_ROWS
        for rows in index_by_region.values()
    )
    validations = {
        "every_completed_region_has_seven_rows": all(
            count == 7 for count in rows_per_region.values()
        ),
        "every_completed_region_has_expected_method_rows": method_layout_ok,
        "habitat_index_regions_are_present_in_lesion_qc": completed_keys <= qc_keys,
        "completed_plus_skipped_equals_total_qc_regions": (
            completed_regions + skipped_regions == total_regions
        ),
        "error_table_is_empty": len(errors) == 0,
    }

    count_rows: list[dict[str, object]] = []
    representation_counts = Counter(
        row.get("representation_status", "") or "blank" for row in lesion_qc
    )
    _append_counts(
        count_rows, "representation_status", representation_counts, total_regions
    )
    fixed_counts = Counter(row.get("fixed_k2_status", "") or "blank" for row in lesion_qc)
    _append_counts(count_rows, "fixed_k2_status", fixed_counts, total_regions)
    method_counts = Counter(row["method"] for row in index)
    _append_counts(count_rows, "habitat_index_rows_by_method", method_counts, len(index))
    check_counts = Counter(row.get("status", "") or "blank" for row in checks)
    _append_counts(count_rows, "invocation_checks", check_counts, len(checks))

    adaptive_region_rows = _method_zone_zero(index_by_region, "adaptive_k1to3")
    adaptive_k_counts = Counter(_display_k(row.get("chosen_k", "")) for row in adaptive_region_rows)
    _append_counts(
        count_rows, "adaptive_chosen_k_all_completed", adaptive_k_counts, completed_regions
    )
    adaptive_eligible = [
        row for row in adaptive_region_rows if _as_bool(row.get("eligible_for_split"))
    ]
    adaptive_eligible_counts = Counter(
        _display_k(row.get("chosen_k", "")) for row in adaptive_eligible
    )
    _append_counts(
        count_rows,
        "adaptive_chosen_k_eligible_regions",
        adaptive_eligible_counts,
        len(adaptive_eligible),
    )

    for method in EXPECTED_METHOD_ROWS:
        method_rows = [row for row in index if row["method"] == method]
        missing_counts = Counter(
            "missing" if _as_bool(row.get("missing")) else "present"
            for row in method_rows
        )
        _append_counts(
            count_rows, f"{method}_zone_rows", missing_counts, len(method_rows)
        )
        region_rows = _method_zone_zero(index_by_region, method)
        validity = Counter(_display_bool(row.get("habitat_valid")) for row in region_rows)
        _append_counts(
            count_rows, f"{method}_habitat_valid", validity, len(region_rows)
        )

    numeric_rows: list[dict[str, object]] = []
    _append_numeric(numeric_rows, "region_voxels_all", _column_floats(lesion_qc, "n_voxels"))
    for status in sorted(representation_counts):
        _append_numeric(
            numeric_rows,
            f"region_voxels__{status}",
            _column_floats(
                [row for row in lesion_qc if (row.get("representation_status") or "blank") == status],
                "n_voxels",
            ),
        )
    for name in (
        "n_features_input",
        "n_features_usable",
        "n_features_selected",
        "n_features_selected_t1",
        "n_features_selected_flair",
        "n_nonfinite_selected_scaled_values",
        "max_abs_spearman_among_selected",
    ):
        _append_numeric(numeric_rows, name, _column_floats(lesion_qc, name))

    fixed_min_sizes: list[float] = []
    fixed_min_fractions: list[float] = []
    for row in lesion_qc:
        sizes = _json_list(row.get("fixed_k2_raw_cluster_sizes", ""))
        n_voxels = _as_float(row.get("n_voxels"))
        if sizes and n_voxels and n_voxels > 0:
            minimum = float(min(sizes))
            fixed_min_sizes.append(minimum)
            fixed_min_fractions.append(minimum / n_voxels)
    _append_numeric(numeric_rows, "fixed_k2_min_raw_cluster_voxels", fixed_min_sizes)
    _append_numeric(numeric_rows, "fixed_k2_min_raw_cluster_fraction", fixed_min_fractions)

    for method in EXPECTED_METHOD_ROWS:
        for zone_id in range(EXPECTED_METHOD_ROWS[method]):
            values = [
                _as_float(row.get("n_voxels"))
                for row in index
                if row["method"] == method and _as_int(row.get("zone_id")) == zone_id
            ]
            _append_numeric(
                numeric_rows,
                f"zone_voxels__{method}__zone_{zone_id}",
                [value for value in values if value is not None],
            )
    _append_numeric(
        numeric_rows,
        "adaptive_silhouette_eligible",
        [
            value
            for row in adaptive_eligible
            if (value := _as_float(row.get("silhouette_score"))) is not None
        ],
    )

    computed_regions = representation_counts.get("computed", 0)
    feature_counts: Counter[str] = Counter()
    insufficient_counts: Counter[str] = Counter()
    constant_counts: Counter[str] = Counter()
    redundant_counts: Counter[str] = Counter()
    for row in lesion_qc:
        if row.get("representation_status") != "computed":
            continue
        feature_counts.update(_json_list(row.get("selected_feature_names", "")))
        insufficient_counts.update(
            _json_list(row.get("dropped_insufficient_support_feature_names", ""))
        )
        constant_counts.update(
            _json_list(row.get("dropped_near_constant_feature_names", ""))
        )
        redundant_counts.update(
            _json_list(row.get("dropped_redundant_feature_names", ""))
        )
    feature_rows = _feature_rows(feature_counts, computed_regions)
    top_selected = feature_rows[:top_features]

    subject_rows = _subject_rows(qc_by_region, index_by_region)
    runtime_seconds = _runtime_seconds(run_metadata)
    fixed_valid = fixed_counts.get("valid_two_cluster", 0)
    adaptive_split = sum(
        count for key, count in adaptive_eligible_counts.items() if key in {"2", "3"}
    )
    summary = {
        "schema_version": 1,
        "stage": "habitats_summary",
        "variant": variant,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "source_files": {name: _fingerprint(path) for name, path in sources.items()},
        "source_run": {
            key: run_metadata.get(key)
            for key in (
                "status",
                "started_utc",
                "finished_utc",
                "protocol_id",
                "protocol_version",
                "protocol_sha256",
                "implementation_id",
                "scientific_config_sha256",
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
        },
        "source_run_runtime_seconds": runtime_seconds,
        "cohort": {
            "total_qc_regions": total_regions,
            "completed_regions": completed_regions,
            "skipped_regions": skipped_regions,
            "failed_units": len(errors),
            "subjects": len({row["subject_id"] for row in lesion_qc}),
            "sessions": len({(row["subject_id"], row["session_id"]) for row in lesion_qc}),
            "habitat_index_rows": len(index),
            "fixed_valid_regions": fixed_valid,
            "adaptive_split_eligible_regions": adaptive_split,
        },
        "validations": validations,
        "counts": count_rows,
        "numeric_summaries": numeric_rows,
        "subjects": subject_rows,
        "selected_features": feature_rows,
        "top_selected_features": top_selected,
        "top_dropped_features": {
            "insufficient_support": _feature_rows(insufficient_counts, computed_regions)[:top_features],
            "near_constant": _feature_rows(constant_counts, computed_regions)[:top_features],
            "redundant": _feature_rows(redundant_counts, computed_regions)[:top_features],
        },
    }

    destination = (
        Path(summary_dir).expanduser().resolve()
        if summary_dir is not None
        else (table_dir / "summary").resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "json": destination / "habitat_summary.json",
        "markdown": destination / "habitat_summary.md",
        "counts": destination / "habitat_summary_counts.csv",
        "numeric": destination / "habitat_summary_numeric.csv",
        "subjects": destination / "habitat_summary_by_subject.csv",
        "features": destination / "habitat_selected_features.csv",
    }
    backups = _archive_existing(targets.values())
    _atomic_text(targets["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(targets["markdown"], _markdown(summary, top_features))
    _atomic_csv(
        targets["counts"],
        count_rows,
        ("group", "category", "count", "denominator", "percent"),
    )
    _atomic_csv(
        targets["numeric"],
        numeric_rows,
        ("metric", "n", "mean", "std", "min", "q25", "median", "q75", "max"),
    )
    _atomic_csv(
        targets["subjects"],
        subject_rows,
        (
            "subject_id",
            "session_id",
            "total_regions",
            "completed_regions",
            "skipped_regions",
            "computed_regions",
            "below_50_voxel_fallback_regions",
            "fixed_valid_regions",
            "fixed_invalid_two_cluster_regions",
            "adaptive_k1_regions",
            "adaptive_k2_regions",
            "adaptive_k3_regions",
        ),
    )
    _atomic_csv(
        targets["features"],
        feature_rows,
        ("rank", "feature", "region_count", "computed_regions", "percent"),
    )
    return HabitatSummaryResult(
        output_root=output_root,
        variant=variant,
        summary_dir=destination,
        json_path=targets["json"],
        markdown_path=targets["markdown"],
        counts_path=targets["counts"],
        numeric_path=targets["numeric"],
        subjects_path=targets["subjects"],
        features_path=targets["features"],
        backups=backups,
        total_regions=total_regions,
        completed_regions=completed_regions,
        skipped_regions=skipped_regions,
        failed_units=len(errors),
        fixed_valid_regions=fixed_valid,
        adaptive_split_regions=adaptive_split,
        source_run_runtime_seconds=runtime_seconds,
    )


def _markdown(summary: Mapping[str, object], top_features: int) -> str:
    cohort = summary["cohort"]
    source_run = summary["source_run"]
    validations = summary["validations"]
    counts = summary["counts"]
    numeric = summary["numeric_summaries"]
    subjects = summary["subjects"]
    features = summary["top_selected_features"]
    lines = [
        f"# Habitat summary — {summary['variant']}",
        "",
        f"Generated: `{summary['generated_utc']}`  ",
        f"Source run finished: `{source_run.get('finished_utc')}`  ",
        f"Reference compatible: `{source_run.get('reference_compatible')}`  ",
        f"Runtime: `{_format_duration(summary.get('source_run_runtime_seconds'))}`",
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
        "## Key categorical results",
        "",
        "| Group | Category | Count | Percent |",
        "| --- | --- | ---: | ---: |",
    ]
    important_groups = {
        "representation_status",
        "fixed_k2_status",
        "adaptive_chosen_k_all_completed",
        "adaptive_chosen_k_eligible_regions",
    }
    lines.extend(
        f"| {row['group']} | {row['category']} | {row['count']} | {row['percent']:.2f}% |"
        for row in counts
        if row["group"] in important_groups
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
            **{key: _format_number(value) for key, value in row.items()}
        )
        for row in numeric
    )
    lines.extend(
        [
            "",
            f"## Top {top_features} selected features",
            "",
            "| Rank | Feature | Regions | Percent of computed regions |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['rank']} | {row['feature']} | {row['region_count']} | {row['percent']:.2f}% |"
        for row in features
    )
    lines.extend(
        [
            "",
            "## By subject/session",
            "",
            "| Subject | Session | Total | Completed | Skipped | Computed | Fallback | Fixed valid | Adaptive k1 | k2 | k3 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['subject_id']} | {row['session_id']} | {row['total_regions']} | "
        f"{row['completed_regions']} | {row['skipped_regions']} | {row['computed_regions']} | "
        f"{row['below_50_voxel_fallback_regions']} | {row['fixed_valid_regions']} | "
        f"{row['adaptive_k1_regions']} | {row['adaptive_k2_regions']} | {row['adaptive_k3_regions']} |"
        for row in subjects
    )
    return "\n".join(lines).rstrip() + "\n"


def _subject_rows(
    qc_by_region: Mapping[tuple[str, ...], Mapping[str, str]],
    index_by_region: Mapping[tuple[str, ...], list[dict[str, str]]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for key in qc_by_region:
        groups[(key[0], key[1])].append(key)
    rows: list[dict[str, object]] = []
    for (subject, session), keys in sorted(groups.items()):
        qc_rows = [qc_by_region[key] for key in keys]
        adaptive_rows = [
            row
            for key in keys
            for row in index_by_region.get(key, [])
            if row["method"] == "adaptive_k1to3" and _as_int(row.get("zone_id")) == 0
        ]
        adaptive = Counter(_display_k(row.get("chosen_k", "")) for row in adaptive_rows)
        fixed = Counter(row.get("fixed_k2_status", "") for row in qc_rows)
        representation = Counter(row.get("representation_status", "") for row in qc_rows)
        rows.append(
            {
                "subject_id": subject,
                "session_id": session,
                "total_regions": len(keys),
                "completed_regions": sum(key in index_by_region for key in keys),
                "skipped_regions": representation.get("not_available_voxelwise_stage_skipped", 0),
                "computed_regions": representation.get("computed", 0),
                "below_50_voxel_fallback_regions": representation.get(
                    "not_computed_below_min_lesion_voxels", 0
                ),
                "fixed_valid_regions": fixed.get("valid_two_cluster", 0),
                "fixed_invalid_two_cluster_regions": fixed.get(
                    "two_cluster_below_validity_threshold", 0
                ),
                "adaptive_k1_regions": adaptive.get("1", 0),
                "adaptive_k2_regions": adaptive.get("2", 0),
                "adaptive_k3_regions": adaptive.get("3", 0),
            }
        )
    return rows


def _feature_rows(counts: Counter[str], denominator: int) -> list[dict[str, object]]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "rank": rank,
            "feature": feature,
            "region_count": count,
            "computed_regions": denominator,
            "percent": 100.0 * count / denominator if denominator else 0.0,
        }
        for rank, (feature, count) in enumerate(ordered, start=1)
    ]


def _append_counts(
    rows: list[dict[str, object]],
    group: str,
    counts: Mapping[str, int],
    denominator: int,
) -> None:
    for category in sorted(counts, key=str):
        count = int(counts[category])
        rows.append(
            {
                "group": group,
                "category": category,
                "count": count,
                "denominator": denominator,
                "percent": 100.0 * count / denominator if denominator else 0.0,
            }
        )


def _append_numeric(
    rows: list[dict[str, object]], metric: str, values: Iterable[float]
) -> None:
    clean = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        rows.append(
            {"metric": metric, "n": 0, "mean": None, "std": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
        )
        return
    q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
    rows.append(
        {
            "metric": metric,
            "n": int(clean.size),
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
            "min": float(clean.min()),
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "max": float(clean.max()),
        }
    )


def _method_zone_zero(
    grouped: Mapping[tuple[str, ...], list[dict[str, str]]], method: str
) -> list[dict[str, str]]:
    return [
        row
        for rows in grouped.values()
        for row in rows
        if row["method"] == method and _as_int(row.get("zone_id")) == 0
    ]


def _group_rows(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return dict(grouped)


def _unique_rows(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...], name: str
) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in output:
            raise HabitatSummaryError(f"{name} contains duplicate region key {'/'.join(key)}")
        output[key] = row
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise HabitatSummaryError(f"cannot read {path}: {exc}") from exc


def _require_columns(
    rows: list[dict[str, str]], required: set[str], name: str, path: Path
) -> None:
    fields = set(rows[0]) if rows else _csv_header(path)
    missing = sorted(required - fields)
    if missing:
        raise HabitatSummaryError(f"{name} is missing column(s): {', '.join(missing)}")


def _csv_header(path: Path) -> set[str]:
    """Read a CSV header even when the table has no data rows."""
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return set(csv.DictReader(handle).fieldnames or ())
    except (OSError, csv.Error) as exc:
        raise HabitatSummaryError(f"cannot read {path}: {exc}") from exc


def _column_floats(rows: Iterable[Mapping[str, str]], name: str) -> list[float]:
    return [value for row in rows if (value := _as_float(row.get(name))) is not None]


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    return parsed if np.isfinite(parsed) else None


def _as_int(value: object) -> int | None:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _display_bool(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return "true"
    if normalized in {"false", "0", "no"}:
        return "false"
    return "not_applicable"


def _display_k(value: object) -> str:
    parsed = _as_int(value)
    return str(parsed) if parsed is not None else "not_available"


def _json_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _runtime_seconds(metadata: Mapping[str, object]) -> float | None:
    try:
        started = datetime.fromisoformat(str(metadata["started_utc"]))
        finished = datetime.fromisoformat(str(metadata["finished_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (finished - started).total_seconds())


def _format_duration(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return "not available"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": digest.hexdigest(),
    }


def _archive_existing(paths: Iterable[Path]) -> tuple[Path, ...]:
    backups: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        counter = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
            counter += 1
        path.replace(backup)
        backups.append(backup)
    return tuple(backups)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(
    path: Path, rows: list[Mapping[str, object]], fields: tuple[str, ...]
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            restval="",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
