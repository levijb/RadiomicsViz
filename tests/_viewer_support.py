"""Synthetic output trees for the viewer tests: the real stages with fake extractors."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from radiomicviz.config import load_project_config
from radiomicviz.config.templates import template_text
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import load_protocol

ROI_FEATURES = ("original_shape_Elongation", "original_firstorder_Mean")
MAP_NAMES = (
    "original_firstorder_Mean",
    "original_firstorder_InterquartileRange",
    "original_firstorder_TotalEnergy",
    "original_glcm_JointAverage",
)
SOFTWARE = {
    "python": "3.10.10",
    "pyradiomics": "3.0.1",
    "numpy": "1.26.4",
    "simpleitk": "2.5.5",
    "pywavelets": "1.8.0",
}
HABITAT_SOFTWARE = {**SOFTWARE, "scipy": "1.11.4", "scikit-learn": "1.1.3"}
SHAPE = (12, 12, 12)


def write_volume(path: Path, array: np.ndarray, *, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), direction=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin(tuple(float(value) for value in origin))
    if direction is not None:
        image.SetDirection(tuple(float(value) for value in direction))
    sitk.WriteImage(image, str(path))
    return path


def texture(seed: int) -> np.ndarray:
    z, y, x = np.indices(SHAPE, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return (20 * x + 7 * np.sin(y) + 3 * z + rng.normal(0, 1, SHAPE)).astype(np.float32) + 50


def block_mask(corner: tuple[int, int, int], count: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    coordinates = np.argwhere(np.ones((4, 4, 4), dtype=bool))[:count] + np.asarray(corner)
    mask[tuple(coordinates.T)] = 1
    return mask


def build_inputs(root: Path, *, regions=None, modalities=("T1", "FLAIR")) -> Path:
    """Write images, separated masks and project.yaml; return the project path."""
    regions = regions or {
        "sub-001": {"lesion_001": ((2, 2, 2), 64), "lesion_002": ((7, 7, 7), 20)},
        "sub-002": {"lesion_001": ((3, 3, 3), 64)},
    }
    data_root = root / "data"
    for number, (subject, lesions) in enumerate(sorted(regions.items()), start=1):
        for index, modality in enumerate(modalities):
            values = texture(number * 10 + index)
            if index % 2:
                values = values.max() - values + 1
            name = "T1w" if modality == "T1" else modality
            write_volume(data_root / subject / "anat" / f"{subject}_{name}_preprocessed.nii.gz", values)
        for region, (corner, count) in lesions.items():
            write_volume(data_root / subject / "lesions" / f"{region}.nii.gz", block_mask(corner, count))
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(root / "output")
    project["modalities"] = {
        modality: {
            "image_pattern": "{subject}/anat/{subject}_"
            + ("T1w" if modality == "T1" else modality) + "_preprocessed.nii.gz"
        }
        for modality in modalities
    }
    project_path = root / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return project_path


def write_manifest(project_path: Path) -> Path:
    written = write_manifest_outputs(build_manifest_preview(load_project_config(project_path)))
    assert written.manifest_path is not None
    return written.manifest_path


def protocol():
    base = load_protocol()
    roi = replace(
        base.roi_preset("b32"),
        expected_feature_count=2,
        expected_shape_feature_count=1,
        expected_diagnostics_count=4,
        expected_feature_schema_sha256=hashlib.sha256("\n".join(ROI_FEATURES).encode()).hexdigest(),
    )
    voxel = replace(
        base.voxelwise_preset("b32"),
        expected_map_count=len(MAP_NAMES),
        expected_map_names=MAP_NAMES,
        expected_map_schema_sha256=hashlib.sha256("\n".join(MAP_NAMES).encode()).hexdigest(),
    )
    return replace(base, expected_software=SOFTWARE, variants={"b32": roi}, voxelwise_variants={"b32": voxel})


class FakeRoiExtractor:
    """Whole-region and habitat-zone extractor returning a two-feature schema."""

    def __init__(self, fail_mask_text: str | None = None) -> None:
        self.fail_mask_text = fail_mask_text

    def execute(self, image: str, mask: str, *, label: int):
        if self.fail_mask_text and self.fail_mask_text in mask:
            raise ValueError("deliberate extraction failure")
        values = sitk.GetArrayFromImage(sitk.ReadImage(mask))
        count = int(np.count_nonzero(values == label))
        return OrderedDict((
            ("diagnostics_Versions_PyRadiomics", "v3.0.1"),
            ("diagnostics_Configuration_Settings", {
                "binCount": 32, "resampledPixelSpacing": [1, 1, 1],
                "interpolator": "sitkBSpline", "normalize": False, "label": label,
            }),
            ("diagnostics_Mask-original_VoxelNum", count),
            ("diagnostics_Mask-interpolated_VoxelNum", count),
            (ROI_FEATURES[0], 0.5 + count / 1000),
            (ROI_FEATURES[1], float(count) + label / 100),
        ))


class FakeVoxelExtractor:
    """Voxelwise extractor producing cropped maps exactly like the stage expects."""

    def execute(self, image: str, mask: str, *, label: int, voxelBased: bool):
        assert voxelBased is True
        source = sitk.ReadImage(image)
        mask_image = sitk.ReadImage(mask)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_image)
        box = stats.GetBoundingBox(label)
        start, size = box[:3], box[3:]
        lower = tuple(max(0, start[axis] - 1) for axis in range(3))
        upper = tuple(min(source.GetSize()[axis], start[axis] + size[axis] + 1) for axis in range(3))
        crop = tuple(upper[axis] - lower[axis] for axis in range(3))
        source_crop = sitk.RegionOfInterest(source, crop, lower)
        mask_crop = sitk.RegionOfInterest(mask_image, crop, lower)
        foreground = sitk.GetArrayFromImage(mask_crop) == label
        values = sitk.GetArrayFromImage(source_crop).astype(np.float64)

        def feature(array: np.ndarray) -> sitk.Image:
            result = np.full(array.shape, np.nan, dtype=np.float64)
            result[foreground] = array[foreground]
            image_out = sitk.GetImageFromArray(result)
            image_out.CopyInformation(source_crop)
            return image_out

        return OrderedDict([
            ("diagnostics_Configuration_Settings", {
                "binCount": 32, "normalize": False, "label": label, "resampledPixelSpacing": None,
            }),
            (MAP_NAMES[0], feature(values)),
            (MAP_NAMES[1], feature(np.sqrt(np.abs(values) + 1.0))),
            (MAP_NAMES[2], feature(values ** 2)),
            (MAP_NAMES[3], feature(np.full(values.shape, np.nan))),
        ])


def run_stages(manifest: Path, stages=("whole", "voxelwise", "habitats", "habitat_features"),
               *, fail_mask_text: str | None = None):
    """Run the requested stages in pipeline order with fakes; return their results."""
    test_protocol = protocol()
    results = {}
    if "whole" in stages:
        from radiomicviz.whole import load_whole_plan, run_whole

        results["whole"] = run_whole(
            load_whole_plan(manifest), test_protocol,
            extractor_factory=lambda _path: FakeRoiExtractor(fail_mask_text), software=SOFTWARE,
        )
    if "voxelwise" in stages:
        from radiomicviz.voxelwise import load_voxelwise_plan, run_voxelwise

        results["voxelwise"] = run_voxelwise(
            load_voxelwise_plan(manifest), test_protocol,
            extractor_factory=lambda _path: FakeVoxelExtractor(), software=SOFTWARE,
        )
    if "habitats" in stages:
        from radiomicviz.habitats import load_habitat_plan, run_habitats

        results["habitats"] = run_habitats(load_habitat_plan(manifest), test_protocol, software=HABITAT_SOFTWARE)
    if "habitat_features" in stages:
        from radiomicviz.habitat_features import load_habitat_feature_plan, run_habitat_features

        results["habitat_features"] = run_habitat_features(
            load_habitat_feature_plan(manifest), test_protocol,
            extractor_factory=lambda _path: FakeRoiExtractor(), software=SOFTWARE,
        )
    return results


def build_tree(root: Path, stages=("whole", "voxelwise", "habitats", "habitat_features"), **kwargs):
    """Complete synthetic tree; returns (manifest path, output root, stage results)."""
    project = build_inputs(root, **kwargs)
    manifest = write_manifest(project)
    return manifest, manifest.parent.parent, run_stages(manifest, stages)


def build_combined_tree(root: Path, stages=("whole",)):
    """Connected-component input: materialize, write the manifest, run stages."""
    from radiomicviz.lesions import build_materialization_plan, run_materialization

    data_root = root / "data"
    for number, subject in enumerate(("sub-001", "sub-002"), start=1):
        for index, name in enumerate(("T1w", "FLAIR")):
            write_volume(data_root / subject / "anat" / f"{subject}_{name}_preprocessed.nii.gz", texture(number * 10 + index))
        combined = block_mask((2, 2, 2), 64) | block_mask((7, 7, 7), 20)
        combined[0, 11, 0] = 1  # one-voxel component, below the minimum
        write_volume(data_root / subject / "lesions" / "all_lesions.nii.gz", combined)
    project = yaml.safe_load(template_text("project"))
    project["data_root"] = str(data_root)
    project["output_root"] = str(root / "output")
    project["regions"].update(mode="connected_components", mask_pattern="{subject}/lesions/all_lesions.nii.gz",
                              id_regex=None, label=1, connectivity=26)
    project_path = root / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    run_materialization(build_materialization_plan(load_project_config(project_path)))
    manifest = write_manifest(project_path)
    run_stages(manifest, stages)
    return manifest, manifest.parent.parent


def write_handmade_manifest(root: Path, keys, *, modalities=("T1", "FLAIR")) -> Path:
    """A manifest written directly (bypassing the builder) for identifier edge cases."""
    data_root = root / "data"
    output_root = root / "output"
    image = write_volume(data_root / "image.nii.gz", texture(1))
    mask = write_volume(data_root / "mask.nii.gz", block_mask((2, 2, 2), 64))
    rows = []
    for subject, session, region in keys:
        for modality in modalities:
            rows.append({
                "subject_id": subject, "session_id": session, "region_id": region,
                "region_type": "lesion", "modality": modality,
                "image_path": str(image), "mask_path": str(mask), "label": "1",
                "space_id": "native", "image_size_bytes": "1", "image_mtime_utc": "x",
                "image_sha256": "0" * 64, "mask_size_bytes": "1", "mask_mtime_utc": "x",
                "mask_sha256": "0" * 64,
            })
    manifest = output_root / "manifest" / "regions_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (manifest.parent / "run_metadata.json").write_text(json.dumps({
        "stage": "manifest", "status": "complete", "data_root": str(data_root),
        "output_root": str(output_root), "reference_status": "reference-compatible",
    }), encoding="utf-8")
    return manifest


def snapshot_tree(*roots: Path) -> dict[str, tuple[int, int, str]]:
    """Every file and directory entry below the roots: (size, mtime_ns, sha256)."""
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
            result[str(path)] = (stat.st_size if path.is_file() else 0, stat.st_mtime_ns, digest)
    return result
