"""Viewer catalog: layout contract (T1) and status model (T3). No Flask needed."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import pytest

import _viewer_support as support
from radiomicviz.viewer import catalog as layout
from radiomicviz.viewer.catalog import Catalog


def _known_paths(catalog: Catalog) -> set[Path]:
    """Every output file the catalog can reach, derived the way the viewer derives it."""
    snapshot = catalog.snapshot
    root, variant = snapshot.output_root, snapshot.variant
    known = {
        snapshot.manifest_path,
        snapshot.manifest_path.with_name("input_checks.csv"),
        snapshot.manifest_path.with_name("errors.csv"),
        snapshot.manifest_path.with_name("run_metadata.json"),
        root / "qc" / "logs" / "manifest.log",
    }
    for stage in layout.STAGES:
        known.update(layout.stage_report_paths(root, stage, variant).values())
        known.update(layout.stage_table_paths(root, stage, variant).values())
    for rid in snapshot.order:
        region = snapshot.regions[rid]
        session_dir = layout.materialized_session_dir(root, region.subject, region.session, region.region_type)
        known.update(session_dir / ".materialization" / name for name in ("complete.json", "inventory.csv", "provenance.json"))
        known.update({session_dir / region.region / "mask.nii.gz", session_dir / region.region / "metadata.json"})
        for unit in region.units["whole"]:
            known.update({unit.output_dir / "completion.json", unit.output_dir / "whole_features.csv"})
        for unit in region.units["voxelwise"]:
            known.add(unit.output_dir / "completion.json")
            known.update(unit.output_dir / f"{name}.nrrd" for name in (unit.completion or {}).get("map_names", []))
        known.add(layout.habitat_completion_path(root, region.key, variant))
        for method in layout.HABITAT_METHODS:
            for source_grid in (False, True):
                known.add(layout.habitat_mask_path(root, region.key, method, variant, source_grid=source_grid))
        for unit in region.units["habitat_features"]:
            known.update({unit.output_dir / "completion.json", unit.output_dir / "features.csv"})
    return known


def test_layout_drift_every_stage_artifact_is_discovered(tmp_path):
    manifest, output_root, results = support.build_tree(tmp_path)
    assert all(result.valid for result in results.values())
    catalog = Catalog(manifest)
    known = _known_paths(catalog)
    on_disk = {
        path for path in output_root.rglob("*")
        if path.is_file() and ".bak-" not in str(path) and ".tmp" not in path.parts
    }
    assert on_disk - known == set(), "files written by a stage that the viewer cannot reach"
    regions = catalog.snapshot.regions.values()
    for region in regions:
        for stage in layout.STAGES:
            assert region.stage_status(stage)[0] == layout.COMPLETE, (region.key, stage)
        for unit in region.units["voxelwise"]:
            for name in unit.completion["map_names"]:
                assert (unit.output_dir / f"{name}.nrrd").is_file()
    assert len(catalog.snapshot.order) == 3
    assert {region.voxels for region in regions} == {64, 20}


def test_layout_drift_materialized_masks_and_provenance(tmp_path):
    manifest, output_root = support.build_combined_tree(tmp_path)
    catalog = Catalog(manifest)
    on_disk = {
        path for path in output_root.rglob("*")
        if path.is_file() and not path.name.endswith(".lock") and ".bak-" not in str(path)
    }
    assert on_disk - _known_paths(catalog) == set()
    from radiomicviz.lesions.materialize import materialized_session_root

    regions = catalog.snapshot.regions.values()
    assert sorted(region.region for region in regions) == ["lesion_001", "lesion_001", "lesion_002", "lesion_002"]
    for region in regions:
        expected = materialized_session_root(output_root, region.subject, region.session, region.region_type)
        assert layout.materialized_session_dir(output_root, region.subject, region.session, region.region_type).resolve() == expected
        assert region.stage_status("whole")[0] == layout.COMPLETE


def test_catalog_paths_equal_the_stages_own_helpers(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path)
    from radiomicviz.habitat_features import load_habitat_feature_plan
    from radiomicviz.habitats import load_habitat_plan
    from radiomicviz.habitats import pipeline as habitat_pipeline
    from radiomicviz.habitats.reference import METHODS
    from radiomicviz.voxelwise import pipeline as voxel_pipeline
    from radiomicviz.whole import load_whole_plan

    assert layout.HABITAT_METHODS == METHODS
    for unit in load_whole_plan(manifest).units:
        key = (unit.subject_id, unit.session_id, unit.region_id)
        assert layout.whole_unit_dir(output_root, key, unit.modality, "b32") == unit.output_dir(output_root, "b32")
        assert layout.voxelwise_unit_dir(output_root, key, unit.modality, "b32") == voxel_pipeline._output_dir(unit, output_root, "b32")
    for unit in load_habitat_plan(manifest).units:
        key = (unit.subject_id, unit.session_id, unit.region_id)
        cropped = habitat_pipeline._mask_paths(unit, output_root, "b32")
        source = habitat_pipeline._source_grid_mask_paths(unit, output_root, "b32")
        for method in METHODS:
            assert layout.habitat_mask_path(output_root, key, method, "b32", source_grid=False) == cropped[method]
            assert layout.habitat_mask_path(output_root, key, method, "b32", source_grid=True) == source[method]
        assert layout.habitat_completion_path(output_root, key, "b32") == habitat_pipeline._completion_path(unit, output_root, "b32")
    for unit in load_habitat_feature_plan(manifest).units:
        key = (unit.subject_id, unit.session_id, unit.region_id)
        assert layout.habitat_feature_unit_dir(output_root, key, unit.method, unit.modality, "b32") == unit.output_dir(output_root, "b32")


def _region(catalog: Catalog, subject: str, region: str):
    return next(r for r in catalog.snapshot.regions.values() if (r.subject, r.region) == (subject, region))


def test_status_model_partial_trees(tmp_path):
    project = support.build_inputs(tmp_path)
    manifest = support.write_manifest(project)
    empty = Catalog(manifest)
    assert {r.stage_status("whole")[0] for r in empty.snapshot.regions.values()} == {layout.MISSING}
    support.run_stages(manifest, ("whole",))
    whole_only = Catalog(manifest)
    region = _region(whole_only, "sub-001", "lesion_001")
    assert region.stage_status("whole") == (layout.COMPLETE, "")
    assert region.stage_status("voxelwise")[0] == layout.MISSING
    support.run_stages(manifest, ("voxelwise", "habitats"))
    voxel_dir = layout.voxelwise_unit_dir(manifest.parent.parent, region.key, "T1", "b32")
    shutil.rmtree(voxel_dir)
    habitat_completion = layout.habitat_completion_path(manifest.parent.parent, _region(whole_only, "sub-002", "lesion_001").key, "b32")
    habitat_completion.unlink()
    partial = Catalog(manifest)
    assert _region(partial, "sub-001", "lesion_001").stage_status("voxelwise")[0] == layout.MISSING
    blocked = _region(partial, "sub-002", "lesion_001").stage_status("habitats")
    assert blocked == (layout.BLOCKED, "outputs exist without a completion record")
    assert _region(partial, "sub-001", "lesion_002").stage_status("habitats")[0] == layout.COMPLETE


def test_skipped_failed_blocked_and_in_progress_units(tmp_path):
    regions = {"sub-001": {"lesion_001": ((2, 2, 2), 64), "lesion_002": ((7, 7, 7), 5)},
               "sub-002": {"lesion_001": ((3, 3, 3), 64)}}
    project = support.build_inputs(tmp_path, regions=regions)
    manifest = support.write_manifest(project)
    support.run_stages(manifest, ("whole",), fail_mask_text="sub-002")
    output_root = manifest.parent.parent
    catalog = Catalog(manifest)
    skipped = _region(catalog, "sub-001", "lesion_002").stage_status("whole")
    assert skipped[0] == layout.SKIPPED and skipped[1]
    failed = _region(catalog, "sub-002", "lesion_001").stage_status("whole")
    assert failed[0] == layout.FAILED and "deliberate extraction failure" in failed[1]
    key = ("sub-001", "ses-01", "lesion_001")
    layout.voxelwise_unit_dir(output_root, key, "T1", "b32").mkdir(parents=True)
    staging = output_root / ".tmp" / "voxelwise" / (
        "u-" + hashlib.sha256("sub-001__ses-01__lesion_001__FLAIR".encode()).hexdigest()[:12] + "-0123abcd"
    )
    staging.mkdir(parents=True)
    catalog.rescan()
    units = {unit.modality: unit for unit in _region(catalog, "sub-001", "lesion_001").units["voxelwise"]}
    assert units["T1"].status == layout.BLOCKED
    assert units["FLAIR"].status == layout.IN_PROGRESS
    assert catalog.snapshot.staging_active["voxelwise"] is True


def test_non_reference_organs_and_arbitrary_modalities(tmp_path):
    project = support.build_inputs(tmp_path, modalities=("T2", "PD"))
    manifest = support.write_manifest(project)
    from radiomicviz.whole import load_whole_plan, run_whole

    run_whole(
        load_whole_plan(manifest), support.protocol(), allow_version_mismatch=True,
        extractor_factory=lambda _path: support.FakeRoiExtractor(),
        software={**support.SOFTWARE, "simpleitk": "2.5.6"},
    )
    catalog = Catalog(manifest)
    region = _region(catalog, "sub-001", "lesion_001")
    assert region.modalities == ("PD", "T2")
    assert region.reference_statuses() == {"non-reference"}
    assert region.stage_status("habitats")[0] == layout.NOT_APPLICABLE
    assert region.stage_status("whole")[0] == layout.COMPLETE


def test_identifiers_are_opaque_and_unsafe_ones_are_never_used_as_paths(tmp_path):
    keys = [
        ("sub__a", "ses 01", "région_ü"),
        ("sub-<b>x", "ses-01", "lesion_001"),
        ("<b>x</b>", "ses-01", "lesion_001"),
        ("sub-001", "ses-01", "liver"),
        ("sub-001", "ses-02", "liver"),
    ]
    manifest = support.write_handmade_manifest(tmp_path, keys)
    catalog = Catalog(manifest)
    regions = catalog.snapshot.regions
    assert len(regions) == 5
    assert all(len(rid) == 12 and all(c in "0123456789abcdef" for c in rid) for rid in regions)
    unsafe = [region for region in regions.values() if region.unsafe]
    assert [region.subject for region in unsafe] == ["<b>x</b>"]
    assert unsafe[0].stage_status("whole")[0] == layout.BLOCKED
    sessions = {region.session for region in regions.values() if region.region == "liver"}
    assert sessions == {"ses-01", "ses-02"}
    for region in regions.values():
        if not region.unsafe:
            assert region.stage_status("whole")[0] == layout.MISSING


def test_empty_manifest_and_unreadable_manifest(tmp_path):
    manifest = support.write_handmade_manifest(tmp_path, [("sub-001", "ses-01", "lesion_001")])
    with manifest.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    manifest.write_text(",".join(header) + "\n", encoding="utf-8")
    assert Catalog(manifest).snapshot.order == []
    other = tmp_path / "other.csv"
    other.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(layout.CatalogError) as error:
        Catalog(other)
    assert str(tmp_path) not in str(error.value)
