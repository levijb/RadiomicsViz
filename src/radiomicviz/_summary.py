"""Shared, dependency-light helpers for derived stage summaries."""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping
from uuid import uuid4

import numpy as np


def canonical_output_root(manifest_path: str | Path) -> tuple[Path, Path]:
    """Return the canonical manifest and output root or raise ``ValueError``."""
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.name != "regions_manifest.csv" or manifest.parent.name != "manifest":
        raise ValueError(
            "expected the canonical <output_root>/manifest/regions_manifest.csv path"
        )
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    return manifest, manifest.parent.parent.resolve()


def read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, tuple(reader.fieldnames or ())


def require_columns(fields: Iterable[str], required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise ValueError(f"{name} is missing column(s): {', '.join(missing)}")


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def as_int(value: object) -> int | None:
    parsed = as_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def append_counts(
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


def numeric_row(metric: str, values: Iterable[float]) -> dict[str, object]:
    clean = np.asarray(
        [float(value) for value in values if np.isfinite(value)], dtype=float
    )
    if clean.size == 0:
        return {
            "metric": metric,
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
    return {
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


def runtime_seconds(metadata: Mapping[str, object]) -> float | None:
    try:
        started = datetime.fromisoformat(str(metadata["started_utc"]))
        finished = datetime.fromisoformat(str(metadata["finished_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (finished - started).total_seconds())


def format_duration(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return "not available"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def fingerprint(path: Path) -> dict[str, object]:
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


def aggregate_fingerprint(fingerprints: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(
        (str(item.get("path", "")), str(item.get("sha256", "")))
        for item in fingerprints
    )
    for path, sha256 in ordered:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def archive_existing(paths: Iterable[Path]) -> tuple[Path, ...]:
    backups: list[Path] = []
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for path in paths:
        if not path.exists():
            continue
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        counter = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
            counter += 1
        path.replace(backup)
        backups.append(backup)
    return tuple(backups)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(
    path: Path, rows: list[Mapping[str, object]], fields: Iterable[str]
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
