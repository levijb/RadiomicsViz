"""Exploratory summaries of habitat-zone radiomics outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from radiomicviz._summary import (
    append_counts,
    archive_existing,
    as_float,
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

KEY_FIELDS = (
    "subject_id",
    "session_id",
    "region_id",
    "method",
    "zone_id",
    "modality",
)
FEATURE_FAMILIES = (
    "shape",
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


class HabitatFeatureSummaryError(RuntimeError):
    """Existing habitat-feature outputs cannot produce a trustworthy summary."""


@dataclass(frozen=True)
class HabitatFeatureSummaryResult:
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
    total_rows: int
    successful_rows: int
    skipped_rows: int
    failed_rows: int
    feature_count: int
    source_run_runtime_seconds: float | None


def summarize_habitat_features(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    summary_dir: str | Path | None = None,
    top_features: int = 20,
) -> HabitatFeatureSummaryResult:
    """Summarize existing habitat-zone feature tables without rerunning extraction."""
    if top_features < 1:
        raise HabitatFeatureSummaryError("summary_top_features must be at least 1")
    try:
        manifest, output_root = canonical_output_root(manifest_path)
        table_dir = output_root / "tables" / "habitat_features" / variant
        sources = {
            "habitat_features": table_dir / "habitat_features.csv",
            "checks": output_root / "qc" / f"habitat_features_{variant}_qc.csv",
            "errors": output_root / "qc" / f"habitat_features_{variant}_errors.csv",
            "run_metadata": (
                output_root / "qc" / f"habitat_features_{variant}_run_metadata.json"
            ),
        }
        _require_sources(sources)
        rows, fields = read_csv(sources["habitat_features"])
        checks, check_fields = read_csv(sources["checks"])
        errors, error_fields = read_csv(sources["errors"])
        metadata = json.loads(sources["run_metadata"].read_text(encoding="utf-8"))
        require_columns(
            fields,
            (*KEY_FIELDS, "source_zone_voxels", "habitat_grid_voxels", "error"),
            "habitat_features.csv",
        )
        require_columns(check_fields, ("status", "check_id"), "habitat-feature QC")
        require_columns(error_fields, ("error_code",), "habitat-feature errors")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HabitatFeatureSummaryError(str(exc)) from exc
    if metadata.get("stage") != "habitat-features" or metadata.get("variant") != variant:
        raise HabitatFeatureSummaryError(
            "habitat-feature run metadata does not match the requested stage and variant"
        )

    feature_names = tuple(name for name in fields if _feature_parts(name) is not None)
    statuses = [_row_status(row) for row in rows]
    status_counts = Counter(statuses)
    successful = [row for row, status in zip(rows, statuses) if status == "complete"]
    methods = sorted({row["method"] for row in rows})
    modalities = sorted({row["modality"] for row in rows})

    count_rows: list[dict[str, object]] = []
    append_counts(count_rows, "zone_status", status_counts, len(rows))
    append_counts(
        count_rows,
        "rows_by_method",
        Counter(row["method"] for row in rows),
        len(rows),
    )
    append_counts(
        count_rows,
        "rows_by_modality",
        Counter(row["modality"] for row in rows),
        len(rows),
    )
    append_counts(
        count_rows,
        "method_modality_status",
        Counter(
            f"{row['method']}__{row['modality']}__{status}"
            for row, status in zip(rows, statuses)
        ),
        len(rows),
    )
    append_counts(
        count_rows,
        "reference_status",
        Counter(row.get("reference_status", "") or "blank" for row in rows),
        len(rows),
    )
    append_counts(
        count_rows,
        "invocation_checks",
        Counter(row.get("status", "") or "blank" for row in checks),
        len(checks),
    )
    append_counts(
        count_rows,
        "error_code",
        Counter(row.get("error_code", "") or "blank" for row in errors),
        len(errors),
    )

    numeric_rows: list[dict[str, object]] = []
    for field in (
        "source_zone_voxels",
        "habitat_grid_voxels",
        "diagnostics_original_voxels",
        "diagnostics_interpolated_voxels",
    ):
        numeric_rows.append(
            numeric_row(
                f"{field}_all_rows",
                [
                    value
                    for row in rows
                    if (value := as_float(row.get(field))) is not None
                ],
            )
        )
    for status in sorted(status_counts):
        numeric_rows.append(
            numeric_row(
                f"source_zone_voxels__{status}",
                [
                    value
                    for row, row_status in zip(rows, statuses)
                    if row_status == status
                    and (value := as_float(row.get("source_zone_voxels"))) is not None
                ],
            )
        )

    feature_rows: list[dict[str, object]] = []
    for method in methods:
        for modality in modalities:
            selected = [
                row
                for row in successful
                if row["method"] == method and row["modality"] == modality
            ]
            for name in feature_names:
                image_type, family = _feature_parts(name) or ("unknown", "unknown")
                values = [
                    value
                    for row in selected
                    if (value := as_float(row.get(name))) is not None
                ]
                stats = numeric_row(name, values)
                missing = len(selected) - len(values)
                feature_rows.append(
                    {
                        "method": method,
                        "modality": modality,
                        "image_type": image_type,
                        "family": family,
                        "feature": name,
                        "successful_rows": len(selected),
                        "finite_values": len(values),
                        "missing_or_nonfinite": missing,
                        "missing_percent": 100.0 * missing / len(selected) if selected else 0.0,
                        **{key: value for key, value in stats.items() if key != "metric"},
                    }
                )
    feature_rows.sort(
        key=lambda row: (str(row["method"]), str(row["modality"]), str(row["feature"]))
    )
    nonzero_missing = [row for row in feature_rows if row["missing_or_nonfinite"]]
    top_missing = sorted(
        nonzero_missing,
        key=lambda row: (
            -float(row["missing_percent"]),
            str(row["method"]),
            str(row["modality"]),
            str(row["feature"]),
        ),
    )[:top_features]

    subject_rows = _subject_rows(rows, statuses)
    validations = {
        "cohort_rows_equal_source_run_zone_rows": len(rows) == metadata.get("zone_rows"),
        "error_rows_equal_source_run_failed_zones": len(errors) == metadata.get("failed_zones"),
        "feature_schema_matches_source_run": len(feature_names) == metadata.get("feature_count"),
        "identity_keys_are_unique": len(rows)
        == len({tuple(row[field] for field in KEY_FIELDS) for row in rows}),
    }
    run_seconds = runtime_seconds(metadata)
    summary = {
        "schema_version": 1,
        "stage": "habitat_feature_summary",
        "variant": variant,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "primary_results_table": str(sources["habitat_features"]),
        "source_files": {name: fingerprint(path) for name, path in sources.items()},
        "source_run": _source_run(metadata),
        "source_run_runtime_seconds": run_seconds,
        "cohort": {
            "zone_modality_rows": len(rows),
            "successful_rows": status_counts.get("complete", 0),
            "expected_skipped_rows": status_counts.get("skipped", 0),
            "failed_or_blocked_rows": (
                status_counts.get("failed", 0) + status_counts.get("blocked", 0)
            ),
            "subjects": len({row["subject_id"] for row in rows}),
            "sessions": len({(row["subject_id"], row["session_id"]) for row in rows}),
            "regions": len(
                {(row["subject_id"], row["session_id"], row["region_id"]) for row in rows}
            ),
            "methods": methods,
            "modalities": modalities,
            "feature_count": len(feature_names),
        },
        "validations": validations,
        "counts": count_rows,
        "numeric_summaries": numeric_rows,
        "subjects": subject_rows,
        "feature_summaries": feature_rows,
        "top_missing_features": top_missing,
    }

    destination = (
        Path(summary_dir).expanduser().resolve()
        if summary_dir is not None
        else (table_dir / "summary").resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "json": destination / "habitat_feature_summary.json",
        "markdown": destination / "habitat_feature_summary.md",
        "counts": destination / "habitat_feature_summary_counts.csv",
        "numeric": destination / "habitat_feature_summary_numeric.csv",
        "subjects": destination / "habitat_feature_summary_by_subject.csv",
        "features": destination / "habitat_feature_summary_features.csv",
    }
    backups = archive_existing(targets.values())
    atomic_text(targets["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(targets["markdown"], _markdown(summary, top_features))
    atomic_csv(
        targets["counts"], count_rows,
        ("group", "category", "count", "denominator", "percent"),
    )
    atomic_csv(
        targets["numeric"], numeric_rows,
        ("metric", "n", "mean", "std", "min", "q25", "median", "q75", "max"),
    )
    atomic_csv(
        targets["subjects"], subject_rows,
        (
            "subject_id", "session_id", "regions", "total_rows", "successful_rows",
            "skipped_rows", "failed_or_blocked_rows",
        ),
    )
    atomic_csv(
        targets["features"], feature_rows,
        (
            "method", "modality", "image_type", "family", "feature",
            "successful_rows", "finite_values", "missing_or_nonfinite",
            "missing_percent", "n", "mean", "std", "min", "q25", "median",
            "q75", "max",
        ),
    )
    return HabitatFeatureSummaryResult(
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
        total_rows=len(rows),
        successful_rows=status_counts.get("complete", 0),
        skipped_rows=status_counts.get("skipped", 0),
        failed_rows=status_counts.get("failed", 0) + status_counts.get("blocked", 0),
        feature_count=len(feature_names),
        source_run_runtime_seconds=run_seconds,
    )


def _require_sources(sources: Mapping[str, Path]) -> None:
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "required habitat-feature output(s) are missing; run "
            "'radiomicviz habitat-features' first:\n  - " + "\n  - ".join(missing)
        )


def _row_status(row: Mapping[str, str]) -> str:
    explicit = (row.get("zone_status") or "").strip()
    if explicit in {"complete", "skipped", "failed", "blocked"}:
        return explicit
    return "failed" if (row.get("error") or "").strip() else "complete"


def _feature_parts(name: str) -> tuple[str, str] | None:
    for family in FEATURE_FAMILIES:
        marker = f"_{family}_"
        if marker in name:
            return name.split(marker, 1)[0], family
    return None


def _subject_rows(
    rows: list[dict[str, str]], statuses: list[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, str], str]]] = defaultdict(list)
    for row, status in zip(rows, statuses):
        grouped[(row["subject_id"], row["session_id"])].append((row, status))
    output: list[dict[str, object]] = []
    for (subject, session), items in sorted(grouped.items()):
        counts = Counter(status for _row, status in items)
        output.append(
            {
                "subject_id": subject,
                "session_id": session,
                "regions": len({row["region_id"] for row, _status in items}),
                "total_rows": len(items),
                "successful_rows": counts.get("complete", 0),
                "skipped_rows": counts.get("skipped", 0),
                "failed_or_blocked_rows": counts.get("failed", 0) + counts.get("blocked", 0),
            }
        )
    return output


def _source_run(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "status", "started_utc", "finished_utc", "protocol_id", "protocol_version",
        "protocol_sha256", "preset_id", "preset_sha256", "reference_compatible",
        "reference_deviations", "jobs", "units", "completed", "resumed", "blocked",
        "zone_rows", "successful_zones", "skipped_zones", "failed_zones", "warnings",
        "diagnostics_count", "feature_count", "feature_schema_sha256",
    )
    return {key: metadata.get(key) for key in keys}


def _markdown(summary: Mapping[str, object], top_features: int) -> str:
    cohort = summary["cohort"]
    source_run = summary["source_run"]
    validations = summary["validations"]
    counts = summary["counts"]
    numeric = summary["numeric_summaries"]
    subjects = summary["subjects"]
    top_missing = summary["top_missing_features"]
    lines = [
        f"# Habitat-zone radiomics summary - {summary['variant']}",
        "",
        f"Generated: `{summary['generated_utc']}`  ",
        f"Primary results: `{summary['primary_results_table']}`  ",
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
        "## Zone-row status",
        "",
        "| Category | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['category']} | {row['count']} | {row['percent']:.2f}% |"
        for row in counts if row["group"] == "zone_status"
    )
    lines.extend(
        [
            "", "## Numeric summaries", "",
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
    lines.extend(["", f"## Up to {top_features} features with missing/non-finite values", ""])
    if top_missing:
        lines.extend(
            [
                "This is a data-completeness check, not feature selection.", "",
                "| Method | Modality | Family | Feature | Finite | Missing | Missing percent |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {row['method']} | {row['modality']} | {row['family']} | "
            f"{row['feature']} | {row['finite_values']} | {row['missing_or_nonfinite']} | "
            f"{row['missing_percent']:.2f}% |"
            for row in top_missing
        )
    else:
        lines.append("No missing or non-finite feature values were found in successful rows.")
    lines.extend(
        [
            "", "## By subject/session", "",
            "| Subject | Session | Regions | Rows | Successful | Skipped | Failed/blocked |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['subject_id']} | {row['session_id']} | {row['regions']} | "
        f"{row['total_rows']} | {row['successful_rows']} | {row['skipped_rows']} | "
        f"{row['failed_or_blocked_rows']} |"
        for row in subjects
    )
    return "\n".join(lines).rstrip() + "\n"
