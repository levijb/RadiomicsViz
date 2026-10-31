"""Exploratory summaries of whole-region radiomics cohort outputs."""

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

KEY_FIELDS = ("subject_id", "session_id", "region_id", "modality")
FEATURE_FAMILIES = (
    "shape",
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


class WholeSummaryError(RuntimeError):
    """Existing whole-region outputs cannot produce a trustworthy summary."""


@dataclass(frozen=True)
class WholeSummaryResult:
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
    total_units: int
    successful_units: int
    skipped_units: int
    failed_units: int
    feature_count: int
    source_run_runtime_seconds: float | None


def summarize_whole(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    summary_dir: str | Path | None = None,
    top_features: int = 20,
) -> WholeSummaryResult:
    """Summarize existing ROI feature tables without rerunning extraction."""
    if top_features < 1:
        raise WholeSummaryError("summary_top_features must be at least 1")
    try:
        manifest, output_root = canonical_output_root(manifest_path)
        table_dir = output_root / "tables" / "whole" / variant
        sources = {
            "whole_region": table_dir / "whole_region.csv",
            "checks": output_root / "qc" / f"whole_{variant}_qc.csv",
            "errors": output_root / "qc" / f"whole_{variant}_errors.csv",
            "run_metadata": output_root / "qc" / f"whole_{variant}_run_metadata.json",
        }
        _require_sources(sources)
        rows, fields = read_csv(sources["whole_region"])
        checks, check_fields = read_csv(sources["checks"])
        errors, error_fields = read_csv(sources["errors"])
        metadata = json.loads(sources["run_metadata"].read_text(encoding="utf-8"))
        require_columns(fields, (*KEY_FIELDS, "mask_voxels", "error"), "whole_region.csv")
        require_columns(check_fields, ("status", "check_id"), "whole QC")
        require_columns(error_fields, ("error_code",), "whole errors")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WholeSummaryError(str(exc)) from exc
    if metadata.get("stage") != "whole" or metadata.get("variant") != variant:
        raise WholeSummaryError(
            "whole run metadata does not match the requested stage and variant"
        )

    feature_names = tuple(name for name in fields if _feature_parts(name) is not None)
    status_by_row = [_row_status(row) for row in rows]
    status_counts = Counter(status_by_row)
    successful_rows = [
        row for row, status in zip(rows, status_by_row) if status == "complete_or_resumed"
    ]
    modalities = sorted({row["modality"] for row in rows})

    count_rows: list[dict[str, object]] = []
    append_counts(count_rows, "unit_status", status_counts, len(rows))
    modality_counts = Counter(row["modality"] for row in rows)
    append_counts(count_rows, "units_by_modality", modality_counts, len(rows))
    modality_status = Counter(
        f"{row['modality']}__{status}"
        for row, status in zip(rows, status_by_row)
    )
    append_counts(count_rows, "modality_status", modality_status, len(rows))
    qc_counts = Counter(row.get("status", "") or "blank" for row in checks)
    append_counts(count_rows, "invocation_checks", qc_counts, len(checks))
    error_counts = Counter(row.get("error_code", "") or "blank" for row in errors)
    append_counts(count_rows, "error_code", error_counts, len(errors))
    family_counts = Counter(_feature_parts(name)[1] for name in feature_names)
    append_counts(count_rows, "feature_schema_family", family_counts, len(feature_names))
    image_type_counts = Counter(_feature_parts(name)[0] for name in feature_names)
    append_counts(
        count_rows, "feature_schema_image_type", image_type_counts, len(feature_names)
    )

    numeric_rows: list[dict[str, object]] = []
    numeric_rows.append(
        numeric_row(
            "mask_voxels_all_units",
            [value for row in rows if (value := as_float(row.get("mask_voxels"))) is not None],
        )
    )
    for status in sorted(status_counts):
        numeric_rows.append(
            numeric_row(
                f"mask_voxels__{status}",
                [
                    value
                    for row, row_status in zip(rows, status_by_row)
                    if row_status == status
                    and (value := as_float(row.get("mask_voxels"))) is not None
                ],
            )
        )
    finite_per_unit = [
        sum(as_float(row.get(name)) is not None for name in feature_names)
        for row in successful_rows
    ]
    numeric_rows.append(
        numeric_row("finite_feature_values_per_successful_unit", finite_per_unit)
    )

    feature_rows: list[dict[str, object]] = []
    for modality in modalities:
        modality_rows = [row for row in successful_rows if row["modality"] == modality]
        for name in feature_names:
            image_type, family = _feature_parts(name) or ("unknown", "unknown")
            values = [
                value
                for row in modality_rows
                if (value := as_float(row.get(name))) is not None
            ]
            stats = numeric_row(name, values)
            missing = len(modality_rows) - len(values)
            feature_rows.append(
                {
                    "modality": modality,
                    "image_type": image_type,
                    "family": family,
                    "feature": name,
                    "total_successful_units": len(modality_rows),
                    "finite_values": len(values),
                    "missing_or_nonfinite": missing,
                    "missing_percent": (
                        100.0 * missing / len(modality_rows) if modality_rows else 0.0
                    ),
                    **{key: value for key, value in stats.items() if key != "metric"},
                }
            )
    feature_rows.sort(key=lambda row: (str(row["modality"]), str(row["feature"])))
    top_missing = sorted(
        [row for row in feature_rows if row["missing_or_nonfinite"]],
        key=lambda row: (
            -float(row["missing_percent"]),
            str(row["modality"]),
            str(row["feature"]),
        ),
    )[:top_features]

    subject_rows = _subject_rows(rows, status_by_row)
    run_seconds = runtime_seconds(metadata)
    validations = {
        "cohort_rows_equal_source_run_units": len(rows) == metadata.get("units"),
        "error_rows_equal_source_run_failed": len(errors) == metadata.get("failed"),
        "feature_schema_matches_source_run": (
            len(feature_names) == metadata.get("feature_count")
        ),
        "identity_keys_are_unique": len(rows)
        == len({tuple(row[field] for field in KEY_FIELDS) for row in rows}),
    }
    summary = {
        "schema_version": 1,
        "stage": "whole_summary",
        "variant": variant,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "source_files": {name: fingerprint(path) for name, path in sources.items()},
        "source_run": _source_run(metadata),
        "source_run_runtime_seconds": run_seconds,
        "cohort": {
            "units": len(rows),
            "successful_units": status_counts.get("complete_or_resumed", 0),
            "skipped_units": status_counts.get("skipped", 0),
            "failed_or_blocked_units": status_counts.get("failed_or_blocked", 0),
            "subjects": len({row["subject_id"] for row in rows}),
            "sessions": len({(row["subject_id"], row["session_id"]) for row in rows}),
            "regions": len(
                {(row["subject_id"], row["session_id"], row["region_id"]) for row in rows}
            ),
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
        "json": destination / "whole_summary.json",
        "markdown": destination / "whole_summary.md",
        "counts": destination / "whole_summary_counts.csv",
        "numeric": destination / "whole_summary_numeric.csv",
        "subjects": destination / "whole_summary_by_subject.csv",
        "features": destination / "whole_summary_features.csv",
    }
    backups = archive_existing(targets.values())
    atomic_text(targets["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(targets["markdown"], _markdown(summary, top_features))
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
        targets["features"],
        feature_rows,
        (
            "modality",
            "image_type",
            "family",
            "feature",
            "total_successful_units",
            "finite_values",
            "missing_or_nonfinite",
            "missing_percent",
            "n",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
        ),
    )
    return WholeSummaryResult(
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
        total_units=len(rows),
        successful_units=status_counts.get("complete_or_resumed", 0),
        skipped_units=status_counts.get("skipped", 0),
        failed_units=status_counts.get("failed_or_blocked", 0),
        feature_count=len(feature_names),
        source_run_runtime_seconds=run_seconds,
    )


def _require_sources(sources: Mapping[str, Path]) -> None:
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError("required whole output(s) are missing:\n  - " + "\n  - ".join(missing))


def _row_status(row: Mapping[str, str]) -> str:
    error = (row.get("error") or "").strip()
    if not error:
        return "complete_or_resumed"
    if error.startswith("skipped:"):
        return "skipped"
    return "failed_or_blocked"


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
    for (subject, session), units in sorted(grouped.items()):
        status_counts = Counter(status for _row, status in units)
        output.append(
            {
                "subject_id": subject,
                "session_id": session,
                "regions": len({row["region_id"] for row, _status in units}),
                "total_units": len(units),
                "successful_units": status_counts.get("complete_or_resumed", 0),
                "skipped_units": status_counts.get("skipped", 0),
                "failed_or_blocked_units": status_counts.get("failed_or_blocked", 0),
                "successful_T1_units": sum(
                    row["modality"] == "T1" and status == "complete_or_resumed"
                    for row, status in units
                ),
                "successful_FLAIR_units": sum(
                    row["modality"] == "FLAIR" and status == "complete_or_resumed"
                    for row, status in units
                ),
            }
        )
    return output


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
        "diagnostics_count",
        "feature_count",
        "feature_schema_sha256",
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
        f"# Whole-region radiomics summary — {summary['variant']}",
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
            f"## Up to {top_features} features with missing/non-finite values",
            "",
        ]
    )
    if top_missing:
        lines.extend(
            [
                "This ranking is a data-completeness check, not feature selection.",
                "",
                "| Modality | Family | Feature | Finite | Missing | Missing percent |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {row['modality']} | {row['family']} | {row['feature']} | "
            f"{row['finite_values']} | {row['missing_or_nonfinite']} | "
            f"{row['missing_percent']:.2f}% |"
            for row in top_missing
        )
    else:
        lines.append(
            "No missing or non-finite feature values were found in successful units."
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
