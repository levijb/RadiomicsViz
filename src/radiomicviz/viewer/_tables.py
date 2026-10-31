"""Feature tables and cohort distributions for the viewer (csv plus NumPy, read-only).

Values are taken exactly as written by the stages. Percentiles, histograms and summary
statistics computed here are viewer-derived and labelled as such by the API.
"""

from __future__ import annotations

import csv
import math
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FEATURE_CLASSES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
_IMAGE_TYPE_PREFIXES = (
    "original", "wavelet", "log-sigma", "square", "squareroot", "logarithm",
    "exponential", "gradient", "lbp-2D", "lbp-3D",
)
_TABLE_LIMIT_BYTES = 1024 * 1024 * 1024


def feature_parts(name: str) -> tuple[str, str, str] | None:
    """Split ``<image type>_<class>_<feature>``; ``None`` for non-feature columns."""
    if name.startswith("diagnostics_"):
        return None
    pieces = name.split("_")
    for index in range(1, len(pieces) - 1):
        if pieces[index] in FEATURE_CLASSES:
            image_type = "_".join(pieces[:index])
            if image_type.startswith(_IMAGE_TYPE_PREFIXES):
                return image_type, pieces[index], "_".join(pieces[index + 1:])
    return None


def parse_value(text: str) -> tuple[float | None, str]:
    """Return ``(number or None, kind)``; kind is number, nonfinite, empty or text."""
    if text is None or text == "":
        return None, "empty"
    try:
        value = float(text)
    except ValueError:
        return None, "text"
    if math.isfinite(value):
        return value, "number"
    return value, "nonfinite"


def group_row(row: Mapping[str, str], fields: Iterable[str]) -> dict[str, Any]:
    """Split one table row into identity, diagnostics and features grouped by class."""
    identity: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in fields:
        text = row.get(name, "")
        if name.startswith("diagnostics_"):
            value, kind = parse_value(text)
            diagnostics.append({"name": name, "text": text, "kind": kind})
            continue
        parts = feature_parts(name)
        if parts is None:
            identity.append({"name": name, "text": text})
            continue
        image_type, feature_class, _ = parts
        value, kind = parse_value(text)
        groups.setdefault(feature_class, {}).setdefault(image_type, []).append(
            {"name": name, "text": text, "kind": kind,
             "value": value if kind == "number" else None}
        )
    return {"identity": identity, "diagnostics": diagnostics, "features": groups}


class ColumnTable:
    """A CSV held column-wise: text columns for identity, float arrays for features."""

    def __init__(self, path: Path) -> None:
        if path.stat().st_size > _TABLE_LIMIT_BYTES:
            raise ValueError("table exceeds the viewer size limit")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            rows = list(reader)
        self.fields = tuple(header)
        self.index = {name: position for position, name in enumerate(header)}
        self.n_rows = len(rows)
        self._rows = rows
        self._numeric: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def text(self, name: str) -> list[str]:
        position = self.index.get(name)
        if position is None:
            return [""] * self.n_rows
        return [row[position] if position < len(row) else "" for row in self._rows]

    def numeric(self, name: str) -> np.ndarray:
        with self._lock:
            cached = self._numeric.get(name)
            if cached is not None:
                return cached
        position = self.index[name]
        values = np.full(self.n_rows, np.nan)
        for row_number, row in enumerate(self._rows):
            text = row[position] if position < len(row) else ""
            try:
                values[row_number] = float(text) if text != "" else np.nan
            except ValueError:
                values[row_number] = np.nan
        with self._lock:
            self._numeric[name] = values
        return values

    def matrix(self, names: tuple[str, ...], rows: list[int]) -> np.ndarray:
        """Selected rows by named numeric columns (NaN for empty or text cells)."""
        result = np.empty((len(rows), len(names)))
        for column, name in enumerate(names):
            result[:, column] = self.numeric(name)[rows]
        return result

    def row(self, row_number: int) -> dict[str, str]:
        values = self._rows[row_number]
        return {name: (values[position] if position < len(values) else "")
                for name, position in self.index.items()}


class TableStore:
    """Load cohort tables lazily, keyed by (path, mtime, size)."""

    def __init__(self) -> None:
        self._tables: dict[str, tuple[tuple[int, int], ColumnTable]] = {}
        self._lock = threading.Lock()

    def get(self, path: Path) -> ColumnTable | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._tables.get(str(path))
            if cached is not None and cached[0] == key:
                return cached[1]
        try:
            table = ColumnTable(path)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError):
            return None
        with self._lock:
            self._tables[str(path)] = (key, table)
        return table


def percentile_ranks(matrix: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column rank of ``values`` among the finite cells: 100 * (below + equal / 2) / n.

    Returns ``(percentiles, n)``; a percentile is NaN where the value is not finite or the
    column has no finite cells. Matches ``distribution(...)["selected_percentile"]``.
    """
    finite = np.isfinite(matrix)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        below = (finite & (matrix < values)).sum(axis=0)
        equal = (finite & (matrix == values)).sum(axis=0)
        ranks = 100.0 * (below + 0.5 * equal) / np.where(n > 0, n, 1)
    ranks[(n == 0) | ~np.isfinite(values)] = np.nan
    return ranks, n


def distribution(values: np.ndarray, selected: float | None, *, bins: int = 30) -> dict[str, Any]:
    """Histogram, summary statistics and the selected value's percentile rank."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "n": int(finite.size),
        "n_nonfinite": int(values.size - finite.size),
        "viewer_derived": True,
        "selected_value": selected,
        "selected_percentile": None,
    }
    if finite.size == 0:
        result.update(histogram={"edges": [], "counts": []}, summary=None)
        return result
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        edges = np.array([low - 0.5, low + 0.5])
        counts = np.array([finite.size])
    else:
        counts, edges = np.histogram(finite, bins=bins, range=(low, high))
    q25, median, q75 = (float(value) for value in np.percentile(finite, [25, 50, 75]))
    result["histogram"] = {
        "edges": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
    }
    result["summary"] = {
        "min": low, "q25": q25, "median": median, "q75": q75, "max": high,
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
    }
    if selected is not None and math.isfinite(selected):
        below = int(np.count_nonzero(finite < selected))
        equal = int(np.count_nonzero(finite == selected))
        result["selected_percentile"] = 100.0 * (below + 0.5 * equal) / finite.size
    return result
