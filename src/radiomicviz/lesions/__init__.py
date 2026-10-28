"""Materialize combined lesion masks into one binary mask per lesion."""

from radiomicviz.lesions.materialize import (
    MaterializationError,
    MaterializationPlan,
    MaterializationResult,
    build_materialization_plan,
    materialized_masks_for_session,
    run_materialization,
)

__all__ = [
    "MaterializationError",
    "MaterializationPlan",
    "MaterializationResult",
    "build_materialization_plan",
    "materialized_masks_for_session",
    "run_materialization",
]
