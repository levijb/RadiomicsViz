"""Load canonical manifest units for voxelwise extraction."""

from __future__ import annotations

from pathlib import Path

from radiomicviz.whole.plan import (
    WholePlan,
    WholePlanError,
    WholeUnit,
    load_whole_plan,
)

VoxelwisePlan = WholePlan
VoxelwiseUnit = WholeUnit


class VoxelwisePlanError(ValueError):
    """The canonical manifest cannot define a safe voxelwise plan."""


def load_voxelwise_plan(path: str | Path) -> VoxelwisePlan:
    """Load the same authenticated region-modality units used by ``whole``."""
    try:
        return load_whole_plan(path)
    except WholePlanError as exc:
        raise VoxelwisePlanError(str(exc)) from exc
