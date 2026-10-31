"""Volume access for the viewer: header checks, conversion, display windows and caching.

Nothing here writes to disk. NIfTI sources are served byte for byte; other formats
(the voxelwise NRRD maps) are converted in memory to float32 NIfTI whose affine is built
from the SimpleITK geometry with an explicit LPS-to-RAS flip, so every cropped map keeps
its own origin. Display windows are computed from finite voxels and are viewer-derived.
"""

from __future__ import annotations

import gzip
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0
_NIFTI_SUFFIXES = (".nii", ".nii.gz")


class VolumeError(Exception):
    """A volume cannot be served; ``code`` is stable and the message holds no paths."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Geometry:
    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    pixel_type: str
    oblique: bool
    sheared: bool

    @property
    def voxels(self) -> int:
        return int(np.prod(self.size, dtype=np.int64))

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "spacing": [float(value) for value in self.spacing],
            "origin": [float(value) for value in self.origin],
            "direction": [float(value) for value in self.direction],
            "pixel_type": self.pixel_type,
            "oblique": self.oblique,
            "sheared": self.sheared,
        }


def is_nifti(path: Path) -> bool:
    return path.name.casefold().endswith(_NIFTI_SUFFIXES)


def max_input_voxels() -> int:
    """The manifest stage's safety limit on header-declared volume size."""
    from radiomicviz.manifest.builder import MAX_INPUT_VOXELS

    return int(MAX_INPUT_VOXELS)


def read_geometry(path: Path) -> Geometry:
    """Read only the header; refuse non-scalar, non-3-D or oversized volumes."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        raise VolumeError("unreadable", "the volume header cannot be read") from None
    if reader.GetDimension() != 3:
        raise VolumeError("unsupported", "only three-dimensional volumes can be displayed")
    if reader.GetNumberOfComponents() != 1:
        raise VolumeError("unsupported", "only scalar volumes can be displayed")
    size = tuple(int(value) for value in reader.GetSize())
    if int(np.prod(size, dtype=np.int64)) > max_input_voxels():
        raise VolumeError("too_large", "the volume header exceeds the package safety limit")
    direction = tuple(float(value) for value in reader.GetDirection())
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    sheared = not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-4)
    oblique = not sheared and not np.allclose(np.abs(matrix), np.round(np.abs(matrix)), atol=1e-6)
    return Geometry(
        size=size,  # type: ignore[arg-type]
        spacing=tuple(float(value) for value in reader.GetSpacing()),  # type: ignore[arg-type]
        origin=tuple(float(value) for value in reader.GetOrigin()),  # type: ignore[arg-type]
        direction=direction,
        pixel_type=sitk.GetPixelIDValueAsString(reader.GetPixelID()),
        oblique=bool(oblique),
        sheared=bool(sheared),
    )


def ras_affine(geometry: Geometry) -> np.ndarray:
    """NIfTI (RAS) voxel-to-world affine from ITK (LPS) spacing, origin and direction."""
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    lps = np.eye(4)
    lps[:3, :3] = direction @ np.diag(geometry.spacing)
    lps[:3, 3] = geometry.origin
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def read_array(path: Path) -> tuple[np.ndarray, Geometry]:
    """Return the voxel array in (x, y, z) order as float64 plus the geometry."""
    import SimpleITK as sitk

    geometry = read_geometry(path)
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError:
        raise VolumeError("unreadable", "the volume data cannot be read") from None
    array = sitk.GetArrayFromImage(image)
    if array.shape[::-1] != geometry.size:
        raise VolumeError("unreadable", "the volume data do not match the header")
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0))), geometry


def nifti_bytes_from_array(array_xyz: np.ndarray, geometry: Geometry, dtype=np.float32) -> bytes:
    """Encode an (x, y, z) array as a single-file NIfTI-1 in memory."""
    import nibabel as nib

    affine = ras_affine(geometry)
    data = np.asarray(array_xyz, dtype=dtype)
    image = nib.Nifti1Image(data, affine)
    image.header.set_qform(affine, code=1)
    image.header.set_sform(affine, code=1)
    image.header.set_xyzt_units(xyz="mm")
    return image.to_bytes()


def check_nifti_source(path: Path) -> Geometry:
    """Validate a NIfTI that will be served unchanged: header limit plus full decode."""
    geometry = read_geometry(path)
    if path.name.casefold().endswith(".gz"):
        try:
            with gzip.open(path, "rb") as handle:
                while handle.read(8 * 1024 * 1024):
                    pass
        except (OSError, EOFError):
            raise VolumeError("unreadable", "the compressed volume is truncated or corrupt") from None
    read_array(path)
    return geometry


def display_window(array: np.ndarray, *, kind: str) -> dict[str, Any]:
    """Robust, viewer-derived display window from finite voxels.

    Images use finite non-zero voxels (background is often exactly 0); feature maps use
    every finite voxel (they are NaN outside the region). Constant, all-non-finite and
    infinite values are handled explicitly.
    """
    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    n_finite = int(finite.sum())
    result: dict[str, Any] = {
        "n_voxels": int(values.size),
        "n_finite": n_finite,
        "n_nonfinite": int(values.size - n_finite),
        "n_positive_infinite": int(np.isposinf(values).sum()),
        "n_negative_infinite": int(np.isneginf(values).sum()),
        "percentiles": [LOWER_PERCENTILE, UPPER_PERCENTILE],
        "basis": "finite non-zero voxels" if kind == "image" else "finite voxels",
        "viewer_derived": True,
    }
    selected = values[finite]
    if kind == "image":
        nonzero = selected[selected != 0]
        if nonzero.size:
            selected = nonzero
    if selected.size == 0:
        result.update(
            min=None, max=None, p_lo=None, p_hi=None, constant=False, all_nonfinite=True,
            cal_min=0.0, cal_max=1.0,
        )
        return result
    vmin, vmax = float(selected.min()), float(selected.max())
    p_lo, p_hi = (float(value) for value in np.percentile(selected, [LOWER_PERCENTILE, UPPER_PERCENTILE]))
    constant = vmax <= vmin
    cal_min, cal_max = p_lo, p_hi
    if constant:
        half = max(abs(vmin) * 0.05, 0.5)
        cal_min, cal_max = vmin - half, vmin + half
    elif cal_max <= cal_min:
        cal_min, cal_max = vmin, vmax
    result.update(
        min=vmin, max=vmax, p_lo=p_lo, p_hi=p_hi, constant=constant, all_nonfinite=False,
        cal_min=cal_min, cal_max=cal_max,
    )
    return result


def label_counts(array: np.ndarray) -> dict[str, int]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values.reshape(-1)
    labels, counts = np.unique(finite, return_counts=True)
    return {str(int(label)) if float(label).is_integer() else str(float(label)): int(count)
            for label, count in zip(labels, counts)}


class ByteLRU:
    """Thread-safe least-recently-used cache capped by total bytes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            self._items.move_to_end(key)
            return item[0]

    def put(self, key: Any, value: Any, size: int) -> None:
        if size > self.max_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._items[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes and self._items:
                _, (_, evicted) = self._items.popitem(last=False)
                self._bytes -= evicted

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes


def signature(path: Path) -> tuple[str, int, int]:
    """Cache key that changes when the file is replaced in place."""
    try:
        stat = path.stat()
    except OSError:
        raise VolumeError("missing", "the file does not exist") from None
    return (str(path), stat.st_mtime_ns, stat.st_size)


class VolumeStore:
    """Serve layer bytes and display statistics with a byte-capped memory cache."""

    def __init__(self, cache_mb: int = 512) -> None:
        self._cache = ByteLRU(max(16, int(cache_mb)) * 1024 * 1024)
        self._validated: dict[tuple[str, int, int], Geometry] = {}
        self._lock = threading.Lock()

    def layer(self, path: Path) -> tuple[str, bytes | Path, Geometry]:
        """Return ``("file", path, geometry)`` for NIfTI or ``("bytes", data, geometry)``."""
        key = signature(path)
        if is_nifti(path):
            with self._lock:
                geometry = self._validated.get(key)
            if geometry is None:
                geometry = check_nifti_source(path)
                with self._lock:
                    self._validated[key] = geometry
            return "file", path, geometry
        cached = self._cache.get(("nifti", key))
        if cached is not None:
            return "bytes", cached[0], cached[1]
        array, geometry = read_array(path)
        data = nifti_bytes_from_array(array, geometry)
        self._cache.put(("nifti", key), (data, geometry), len(data))
        return "bytes", data, geometry

    def centroid(self, path: Path, label: int | None) -> list[float] | None:
        """World (RAS, mm) centroid of the labelled voxels; viewer-derived."""
        key = ("centroid", label, signature(path))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        array, geometry = read_array(path)
        selected = array == label if label is not None else array > 0
        if not selected.any():
            return None
        index = np.argwhere(selected).mean(axis=0)
        world = ras_affine(geometry) @ np.append(index, 1.0)
        result = [float(value) for value in world[:3]]
        self._cache.put(key, result, 256)
        return result

    def stats(self, path: Path, kind: str) -> dict[str, Any]:
        key = ("stats", kind, signature(path))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        array, geometry = read_array(path)
        if kind in {"region_mask", "habitat_mask_cropped", "habitat_mask_source_grid"}:
            result: dict[str, Any] = {"labels": label_counts(array), "viewer_derived": True}
        else:
            result = display_window(array, kind="image" if kind == "image" else "map")
        result["geometry"] = geometry.to_dict()
        self._cache.put(key, result, 4096)
        return result
