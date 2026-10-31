"""Small, deterministic ROI checks performed before PyRadiomics extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class RoiInspection:
    voxel_count: int
    dimensions: int
    intensity_min: float | None
    intensity_max: float | None
    finite_intensities: bool


def inspect_roi(image_path: Path, mask_path: Path, label: int) -> RoiInspection:
    """Measure ROI size, dimensionality and intensity range on the source grid."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image_values = np.asarray(sitk.GetArrayViewFromImage(image))
    mask_values = np.asarray(sitk.GetArrayViewFromImage(mask))
    foreground = mask_values == label
    count = int(np.count_nonzero(foreground))
    if count == 0:
        return RoiInspection(0, 0, None, None, False)

    coordinates = np.argwhere(foreground)
    extents = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    dimensions = int(np.count_nonzero(extents > 1))
    values = np.asarray(image_values[foreground], dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    if not finite or values.size == 0:
        return RoiInspection(count, dimensions, None, None, finite)
    return RoiInspection(
        count,
        dimensions,
        float(values.min()),
        float(values.max()),
        True,
    )
