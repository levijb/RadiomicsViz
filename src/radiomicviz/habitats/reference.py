"""Scientific core derived from the validated habitat-clustering script.

Only package I/O was removed.  The lesion-local representation, fixed/adaptive
KMeans rules, radial canonicalization, and geometric comparator retain the
reference implementation's constants and operation order.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EXCLUDED_FEATURES = frozenset({"glcm_MCC", "firstorder_TotalEnergy"})
WITHIN_LESION_CORR_THRESHOLD = 0.90
PER_LESION_IQR_RTOL = 1e-8
PER_LESION_IQR_ATOL = 1e-12
MIN_FEATURES_PER_MODALITY = 1
K_FIXED = 2
K_ADAPTIVE_CANDIDATES = (2, 3)
K_ADAPTIVE_MAX = 3
SILHOUETTE_MIN = 0.20
FIXED_K2_MIN_CLUSTER_VOXELS = 10
FIXED_K2_MIN_CLUSTER_FRACTION = 0.05
ADAPTIVE_MIN_CLUSTER_VOXELS = 10
N_KMEANS_INIT = 10
KMEANS_SEED = 42
MIN_LESION_VOXELS = 50
ZONE_TOO_SMALL_VOXELS = 10
SHELL_THICKNESS_MM = 1.0
GEOMETRY_RTOL = 1e-4
GEOMETRY_ATOL = 1e-4

METHOD_FIXED = "fixed_k2"
METHOD_ADAPTIVE = "adaptive_k1to3"
METHOD_GEOMETRIC = "geometric_1mm"
METHODS = (METHOD_FIXED, METHOD_ADAPTIVE, METHOD_GEOMETRIC)
REFERENCE_SOURCE_SHA256 = "c0184f4f24f163aebaf58ea8dade70ef50545013b84230e300fea000381ee20a"

SCIENTIFIC_CONFIG: Mapping[str, object] = {
    "implementation_id": "reference",
    "reference_source_sha256": REFERENCE_SOURCE_SHA256,
    "excluded_features": sorted(EXCLUDED_FEATURES),
    "representation_scope": "independent_per_lesion",
    "scaling_method": "per-lesion median/IQR; near-constant features excluded",
    "missing_value_handling": "median imputation as zero after scaling",
    "per_lesion_iqr_rtol": PER_LESION_IQR_RTOL,
    "per_lesion_iqr_atol": PER_LESION_IQR_ATOL,
    "within_lesion_corr_threshold": WITHIN_LESION_CORR_THRESHOLD,
    "modality_weighting": "each retained block weighted by 1/sqrt(n_features_in_modality)",
    "min_features_per_modality": MIN_FEATURES_PER_MODALITY,
    "k_fixed": K_FIXED,
    "k_adaptive_candidates": list(K_ADAPTIVE_CANDIDATES),
    "k_adaptive_max": K_ADAPTIVE_MAX,
    "silhouette_min": SILHOUETTE_MIN,
    "fixed_k2_min_cluster_voxels": FIXED_K2_MIN_CLUSTER_VOXELS,
    "fixed_k2_min_cluster_fraction": FIXED_K2_MIN_CLUSTER_FRACTION,
    "adaptive_min_cluster_voxels": ADAPTIVE_MIN_CLUSTER_VOXELS,
    "n_kmeans_init": N_KMEANS_INIT,
    "kmeans_seed": KMEANS_SEED,
    "min_lesion_voxels": MIN_LESION_VOXELS,
    "zone_too_small_voxels": ZONE_TOO_SMALL_VOXELS,
    "shell_thickness_mm": SHELL_THICKNESS_MM,
    "geometry_rtol": GEOMETRY_RTOL,
    "geometry_atol": GEOMETRY_ATOL,
}
SCIENTIFIC_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(SCIENTIFIC_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True)
class HabitatComputation:
    masks: Mapping[str, np.ndarray]
    rows: tuple[dict[str, object], ...]
    qc: Mapping[str, object]


def _feature_modality(name: str) -> str:
    if name.startswith("T1_"):
        return "T1"
    if name.startswith("FLAIR_"):
        return "FLAIR"
    raise ValueError(f"feature name has no T1_/FLAIR_ modality prefix: {name!r}")


def _spearman_matrix(matrix: np.ndarray) -> np.ndarray:
    """Column-wise Spearman matrix with scipy's two-column special case."""
    n_features = matrix.shape[1]
    if n_features == 1:
        return np.ones((1, 1), dtype=float)
    result = stats.spearmanr(matrix, axis=0, nan_policy="omit")
    corr = np.asarray(result.correlation, dtype=float)
    if corr.ndim == 0:
        value = float(corr)
        corr = np.array([[1.0, value], [value, 1.0]], dtype=float)
    corr = corr.reshape(n_features, n_features)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def prepare_lesion_feature_matrix(
    raw: np.ndarray,
    feature_names: list[str],
    mask_3d: np.ndarray,
    corr_threshold: float = WITHIN_LESION_CORR_THRESHOLD,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create the shared lesion-local T1+FLAIR matrix used by both KMeans methods."""
    if raw.ndim != 2 or raw.shape[1] != len(feature_names):
        raise ValueError(
            f"raw/feature-name shape mismatch: raw={raw.shape}, names={len(feature_names)}"
        )
    flat_mask = mask_3d.reshape(-1) > 0
    lesion_raw = np.asarray(raw[flat_mask], dtype=float)
    if lesion_raw.shape[0] == 0:
        raise ValueError("lesion mask contains zero voxels on the feature grid")

    finite = np.isfinite(lesion_raw)
    finite_counts = finite.sum(axis=0)
    clean = np.where(finite, lesion_raw, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        centers = np.nanmedian(clean, axis=0)
        q75, q25 = np.nanpercentile(clean, [75, 25], axis=0)
    scales = q75 - q25
    scale_floor = PER_LESION_IQR_ATOL + PER_LESION_IQR_RTOL * np.maximum(
        np.abs(centers), 1.0
    )
    insufficient = (finite_counts < 2) | ~np.isfinite(centers) | ~np.isfinite(scales)
    near_constant = ~insufficient & (scales <= scale_floor)
    usable_mask = ~(insufficient | near_constant)
    usable_indices = np.flatnonzero(usable_mask)
    insufficient_names = [feature_names[i] for i in np.flatnonzero(insufficient)]
    near_constant_names = [feature_names[i] for i in np.flatnonzero(near_constant)]
    if len(usable_indices) == 0:
        raise ValueError("zero usable features after lesion-local finite-support/IQR filtering")

    usable_names = [feature_names[i] for i in usable_indices]
    usable_modalities = [_feature_modality(name) for name in usable_names]
    for modality in ("T1", "FLAIR"):
        if usable_modalities.count(modality) < MIN_FEATURES_PER_MODALITY:
            raise ValueError(
                f"zero usable {modality} features after lesion-local finite-support/IQR filtering"
            )

    scaled = (lesion_raw[:, usable_indices] - centers[usable_indices]) / scales[usable_indices]
    nonfinite_per_usable = (~np.isfinite(scaled)).sum(axis=0)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    if progress is not None:
        progress("computing joint Spearman redundancy matrix")
    corr = _spearman_matrix(scaled)
    redundant = np.abs(corr) >= corr_threshold
    np.fill_diagonal(redundant, False)
    degree = redundant.sum(axis=1)
    priority = sorted(
        range(len(usable_names)), key=lambda i: (-int(degree[i]), usable_names[i])
    )

    selected_local: list[int] = []
    redundant_with: dict[str, dict[str, object]] = {}
    for index in priority:
        conflicts = [kept for kept in selected_local if redundant[index, kept]]
        if not conflicts:
            selected_local.append(index)
            continue
        blocker = sorted(
            conflicts,
            key=lambda kept: (-abs(corr[index, kept]), usable_names[kept]),
        )[0]
        redundant_with[usable_names[index]] = {
            "retained": usable_names[blocker],
            "abs_spearman_rho": float(abs(corr[index, blocker])),
        }

    forced_modality_names: list[str] = []
    for modality in ("T1", "FLAIR"):
        if any(usable_modalities[i] == modality for i in selected_local):
            continue
        candidates = [i for i, value in enumerate(usable_modalities) if value == modality]

        def candidate_key(index: int) -> tuple[float, str]:
            maximum = max((abs(corr[index, kept]) for kept in selected_local), default=0.0)
            return maximum, usable_names[index]

        restored = min(candidates, key=candidate_key)
        selected_local.append(restored)
        forced_modality_names.append(usable_names[restored])
        redundant_with.pop(usable_names[restored], None)

    selected_local = sorted(set(selected_local))
    selected_names = [usable_names[i] for i in selected_local]
    selected_modalities = [_feature_modality(name) for name in selected_names]
    modality_counts = {
        modality: selected_modalities.count(modality) for modality in ("T1", "FLAIR")
    }
    modality_weights = {
        modality: 1.0 / np.sqrt(modality_counts[modality])
        for modality in ("T1", "FLAIR")
    }
    features = scaled[:, selected_local].copy()
    for column, modality in enumerate(selected_modalities):
        features[:, column] *= modality_weights[modality]

    selected_nonfinite = int(nonfinite_per_usable[selected_local].sum())
    selected_corr = np.abs(corr[np.ix_(selected_local, selected_local)])
    if len(selected_local) > 1:
        selected_corr[np.eye(len(selected_local), dtype=bool)] = np.nan
        max_selected_corr = float(np.nanmax(selected_corr))
    else:
        max_selected_corr = np.nan
    return features, {
        "n_features_input": int(len(feature_names)),
        "n_features_usable": int(len(usable_names)),
        "n_features_selected": int(len(selected_names)),
        "n_features_selected_t1": int(modality_counts["T1"]),
        "n_features_selected_flair": int(modality_counts["FLAIR"]),
        "t1_block_weight": float(modality_weights["T1"]),
        "flair_block_weight": float(modality_weights["FLAIR"]),
        "within_lesion_corr_threshold": float(corr_threshold),
        "max_abs_spearman_among_selected": max_selected_corr,
        "n_nonfinite_selected_scaled_values": selected_nonfinite,
        "selected_feature_names": selected_names,
        "dropped_insufficient_support_feature_names": insufficient_names,
        "dropped_near_constant_feature_names": near_constant_names,
        "dropped_redundant_feature_names": sorted(redundant_with),
        "redundant_feature_map": redundant_with,
        "forced_modality_feature_names": forced_modality_names,
    }


def kmeans_cluster(
    features: np.ndarray,
    k: int,
    seed: int = KMEANS_SEED,
    n_init: int = N_KMEANS_INIT,
) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(features)


def select_adaptive_k(
    features: np.ndarray,
    candidates: tuple[int, ...] = K_ADAPTIVE_CANDIDATES,
    min_silhouette: float = SILHOUETTE_MIN,
    min_per_cluster: int = ADAPTIVE_MIN_CLUSTER_VOXELS,
    seed: int = KMEANS_SEED,
    n_init: int = N_KMEANS_INIT,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, np.ndarray | None, float]:
    n_voxels = features.shape[0]
    best_k, best_score, best_labels = 1, -1.0, None
    for k in candidates:
        if n_voxels < k * min_per_cluster:
            continue
        if progress is not None:
            progress(f"adaptive clustering: testing k={k}")
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(features)
        if len(np.unique(labels)) < 2 or np.bincount(labels).min() < min_per_cluster:
            continue
        score = silhouette_score(
            features,
            labels,
            metric="euclidean",
            sample_size=min(5000, n_voxels),
            random_state=seed,
        )
        if score > best_score:
            best_k, best_score, best_labels = k, float(score), labels
    if best_score < min_silhouette:
        return 1, None, best_score if best_score > -1.0 else np.nan
    return best_k, best_labels, best_score


def canonicalise(
    coords: np.ndarray,
    labels: np.ndarray,
    mask_3d: np.ndarray,
    spacing: tuple[float, ...],
) -> np.ndarray:
    padded = np.pad(mask_3d, 1, mode="constant", constant_values=0)
    distance = distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    values = distance[coords[:, 0], coords[:, 1], coords[:, 2]]
    unique = np.unique(labels)
    mean_distance = {label: values[labels == label].mean() for label in unique}
    order = sorted(unique, key=lambda label: mean_distance[label])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[label] for label in labels])


def radial_shell_core(
    mask_3d: np.ndarray,
    spacing: tuple[float, ...],
    shell_mm: float = SHELL_THICKNESS_MM,
) -> dict[int, np.ndarray]:
    padded = np.pad(mask_3d, 1, mode="constant", constant_values=0)
    distance = distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    return {
        0: ((mask_3d > 0) & (distance <= shell_mm)).astype(np.uint8),
        1: ((mask_3d > 0) & (distance > shell_mm)).astype(np.uint8),
    }


def zone_name(method: str, zone_id: int, chosen_k: int | None = None) -> str:
    if method == METHOD_GEOMETRIC:
        return {0: "geometric_shell", 1: "geometric_core"}[zone_id]
    k = chosen_k if chosen_k is not None else K_FIXED
    if k == 1:
        return "whole_lesion" if zone_id == 0 else "not_applicable"
    if zone_id >= k:
        return "not_applicable"
    if zone_id == 0:
        return "peripheral_habitat"
    if zone_id == k - 1:
        return "central_habitat"
    return f"intermediate_habitat_{zone_id}"


def _meta_row(
    method: str,
    zone_id: int,
    n_voxels: int,
    missing: int,
    eligible: bool,
    chosen_k: int | None = None,
    silhouette: float = np.nan,
    habitat_valid: bool | float = np.nan,
) -> dict[str, object]:
    return {
        "method": method,
        "zone_id": zone_id,
        "zone_name": zone_name(method, zone_id, chosen_k),
        "n_voxels": int(n_voxels),
        "missing": int(missing),
        "eligible_for_split": bool(eligible),
        "chosen_k": chosen_k if chosen_k is not None else np.nan,
        "silhouette_score": silhouette,
        "habitat_valid": habitat_valid,
        "mask_label_value": int(zone_id) + 1,
    }


def _label_volume(
    shape: tuple[int, ...], coords_by_zone: Mapping[int, np.ndarray | None]
) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.uint8)
    for zone_id, coords in coords_by_zone.items():
        if coords is not None and len(coords):
            volume[coords[:, 0], coords[:, 1], coords[:, 2]] = zone_id + 1
    return volume


def process_region(
    raw: np.ndarray,
    feature_names: list[str],
    mask_3d: np.ndarray,
    spacing_zyx: tuple[float, ...],
    progress: Callable[[str], None] | None = None,
) -> HabitatComputation:
    """Apply all three reference methods to one region."""
    report = progress or (lambda _message: None)
    rows: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    coords = np.argwhere(mask_3d > 0)
    n_voxels = len(coords)
    eligible = n_voxels >= MIN_LESION_VOXELS
    if eligible:
        report("preparing lesion-local T1+FLAIR representation")
        features, representation = prepare_lesion_feature_matrix(
            raw, feature_names, mask_3d, progress=progress
        )
        representation["representation_status"] = "computed"
    else:
        features = None
        representation = {
            "representation_status": "not_computed_below_min_lesion_voxels",
            "n_features_input": int(len(feature_names)),
            "n_features_usable": np.nan,
            "n_features_selected": np.nan,
            "n_features_selected_t1": np.nan,
            "n_features_selected_flair": np.nan,
            "t1_block_weight": np.nan,
            "flair_block_weight": np.nan,
            "within_lesion_corr_threshold": float(WITHIN_LESION_CORR_THRESHOLD),
            "max_abs_spearman_among_selected": np.nan,
            "n_nonfinite_selected_scaled_values": np.nan,
            "selected_feature_names": [],
            "dropped_insufficient_support_feature_names": [],
            "dropped_near_constant_feature_names": [],
            "dropped_redundant_feature_names": [],
            "redundant_feature_map": {},
            "forced_modality_feature_names": [],
        }

    fixed_sizes: list[int] = []
    fixed_output_k = 1
    fixed_reason: str | None = None
    fixed_status = "lesion_below_min_lesion_voxels"
    fixed_threshold = max(
        FIXED_K2_MIN_CLUSTER_VOXELS,
        int(np.ceil(FIXED_K2_MIN_CLUSTER_FRACTION * n_voxels)),
    )
    if not eligible:
        fixed_reason = "lesion_below_min_lesion_voxels"
        masks[METHOD_FIXED] = _label_volume(mask_3d.shape, {0: coords, 1: None})
        rows.append(_meta_row(METHOD_FIXED, 0, n_voxels, int(n_voxels < 10), False, 1, habitat_valid=False))
        rows.append(_meta_row(METHOD_FIXED, 1, 0, 1, False, 1, habitat_valid=False))
    else:
        labels = kmeans_cluster(features, K_FIXED)  # type: ignore[arg-type]
        fixed_sizes = np.bincount(labels, minlength=K_FIXED).astype(int).tolist()
        if len(np.unique(labels)) < K_FIXED or min(fixed_sizes) == 0:
            fixed_reason = "kmeans_returned_fewer_than_two_nonempty_clusters"
            fixed_status = "degenerate_one_cluster"
            masks[METHOD_FIXED] = _label_volume(mask_3d.shape, {0: coords, 1: None})
            rows.append(_meta_row(METHOD_FIXED, 0, n_voxels, int(n_voxels < 10), True, 1, habitat_valid=False))
            rows.append(_meta_row(METHOD_FIXED, 1, 0, 1, True, 1, habitat_valid=False))
        else:
            fixed_output_k = K_FIXED
            fixed_valid = min(fixed_sizes) >= fixed_threshold
            fixed_status = "valid_two_cluster" if fixed_valid else "two_cluster_below_validity_threshold"
            labels = canonicalise(coords, labels, mask_3d, spacing_zyx)
            zone_coords = {zone: coords[labels == zone] for zone in range(K_FIXED)}
            for zone, zone_values in zone_coords.items():
                rows.append(_meta_row(METHOD_FIXED, zone, len(zone_values), int(len(zone_values) < 10), True, 2, habitat_valid=fixed_valid))
            masks[METHOD_FIXED] = _label_volume(mask_3d.shape, zone_coords)

    if not eligible:
        masks[METHOD_ADAPTIVE] = _label_volume(mask_3d.shape, {0: coords, 1: None, 2: None})
        rows.append(_meta_row(METHOD_ADAPTIVE, 0, n_voxels, int(n_voxels < 10), False, 1, habitat_valid=False))
        for zone in (1, 2):
            rows.append(_meta_row(METHOD_ADAPTIVE, zone, 0, 1, False, 1, habitat_valid=False))
    else:
        chosen_k, labels, silhouette = select_adaptive_k(features, progress=progress)  # type: ignore[arg-type]
        adaptive_valid = chosen_k > 1
        if chosen_k == 1:
            masks[METHOD_ADAPTIVE] = _label_volume(mask_3d.shape, {0: coords, 1: None, 2: None})
            rows.append(_meta_row(METHOD_ADAPTIVE, 0, n_voxels, 0, True, 1, silhouette, adaptive_valid))
            for zone in (1, 2):
                rows.append(_meta_row(METHOD_ADAPTIVE, zone, 0, 1, True, 1, silhouette, adaptive_valid))
        else:
            labels = canonicalise(coords, labels, mask_3d, spacing_zyx)  # type: ignore[arg-type]
            zone_coords: dict[int, np.ndarray | None] = {}
            for zone in range(K_ADAPTIVE_MAX):
                if zone >= chosen_k:
                    zone_coords[zone] = None
                    rows.append(_meta_row(METHOD_ADAPTIVE, zone, 0, 1, True, chosen_k, silhouette, adaptive_valid))
                else:
                    zone_values = coords[labels == zone]
                    zone_coords[zone] = zone_values
                    rows.append(_meta_row(METHOD_ADAPTIVE, zone, len(zone_values), int(len(zone_values) < 10), True, chosen_k, silhouette, adaptive_valid))
            masks[METHOD_ADAPTIVE] = _label_volume(mask_3d.shape, zone_coords)

    geometric = radial_shell_core(mask_3d, spacing_zyx, SHELL_THICKNESS_MM)
    geometric_coords: dict[int, np.ndarray | None] = {}
    for zone, zone_mask in geometric.items():
        zone_values = np.argwhere(zone_mask > 0)
        geometric_coords[zone] = zone_values if len(zone_values) else None
        rows.append(_meta_row(METHOD_GEOMETRIC, zone, len(zone_values), int(len(zone_values) < 10), len(zone_values) >= 10, habitat_valid=np.nan))
    masks[METHOD_GEOMETRIC] = _label_volume(mask_3d.shape, geometric_coords)

    qc: dict[str, object] = {
        "n_voxels": n_voxels,
        "fixed_k2_raw_cluster_sizes": fixed_sizes,
        "fixed_k2_min_cluster_voxels": FIXED_K2_MIN_CLUSTER_VOXELS,
        "fixed_k2_min_cluster_fraction": FIXED_K2_MIN_CLUSTER_FRACTION,
        "fixed_k2_validity_threshold_voxels": fixed_threshold,
        "fixed_k2_output_k": fixed_output_k,
        "fixed_k2_status": fixed_status,
        "fixed_k2_fallback_reason": fixed_reason,
        **representation,
    }
    return HabitatComputation(masks=masks, rows=tuple(rows), qc=qc)
