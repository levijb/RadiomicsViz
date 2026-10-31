"""Viewer HTTP API: read-only proof (T4), safety (T5), robustness (T6), data (T7), blind mode."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

import _viewer_support as support  # noqa: E402
from radiomicviz.viewer import create_app  # noqa: E402
from radiomicviz.viewer import catalog as layout  # noqa: E402

TOKEN = "test-token"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tree")
    manifest, output_root, _ = support.build_tree(root)
    return manifest, output_root, root / "data"


def _client(manifest, **kwargs):
    app = create_app(manifest, access_token=TOKEN, **kwargs)
    client = app.test_client()
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


def _first_region(client):
    return client.get("/api/regions").get_json()["regions"][0]["rid"]


def _every_endpoint(client):
    responses = [client.get(path) for path in (
        "/", "/static/viewer.js", "/static/viewer.css", "/static/vendor/niivue/niivue.umd.js",
        "/api/project", "/api/status", "/api/regions?sort=-voxels&q=lesion", "/api/qc",
        "/api/cohort/regions", "/api/downloads",
    )]
    for region in client.get("/api/regions").get_json()["regions"]:
        rid = region["rid"]
        detail = client.get(f"/api/region/{rid}").get_json()
        responses.append(client.post(f"/api/region/{rid}/verify"))
        layers = detail["layers"] + [item for groups in detail["feature_maps"].values() for items in groups.values() for item in items]
        for layer in layers:
            responses.append(client.get(f"/api/region/{rid}/layer/{layer['id']}"))
            responses.append(client.get(f"/api/region/{rid}/layer/{layer['id']}/display"))
        for modality in ("T1", "FLAIR"):
            responses.append(client.get(f"/api/region/{rid}/features/whole?modality={modality}"))
            for method in layout.HABITAT_METHODS:
                responses.append(client.get(f"/api/region/{rid}/features/habitat?method={method}&modality={modality}"))
            responses.append(client.get(
                f"/api/cohort/distribution?stage=whole&modality={modality}&feature=original_firstorder_Mean&rid={rid}"))
            responses.append(client.get(
                f"/api/cohort/distribution?stage=habitat_features&modality={modality}&method=fixed_k2"
                f"&zone=peripheral_habitat&feature=original_firstorder_Mean&rid={rid}"))
    for stage in layout.STAGES:
        responses.append(client.get(f"/api/summary/{stage}"))
    for item in client.get("/api/downloads").get_json()["files"]:
        responses.append(client.get(f"/api/download/{item['id']}"))
    responses += [client.post("/api/rescan"), client.get("/api/region/ffffffffffff"), client.get("/api/nothing")]
    return responses


def test_viewer_is_read_only_for_every_endpoint(tree):
    manifest, output_root, data_root = tree
    before = support.snapshot_tree(data_root, output_root)
    client = _client(manifest, verify_integrity=True)
    responses = _every_endpoint(client)
    for response in responses:
        response.close()  # the test client leaves streamed file responses open otherwise
    assert all(response.status_code < 500 for response in responses)
    assert sum(response.status_code == 200 for response in responses) > 100
    after = support.snapshot_tree(data_root, output_root)
    assert after == before


def test_security_headers_tokens_hosts_and_methods(tree):
    manifest, _, _ = tree
    app = create_app(manifest, access_token=TOKEN)
    anonymous = app.test_client()
    assert anonymous.get("/").status_code == 404
    assert anonymous.get("/api/project").status_code == 404
    assert anonymous.get("/?token=wrong").status_code == 404
    client = _client(manifest)
    page = client.get("/")
    csp = page.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("style-src")[0]
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert "Access-Control-Allow-Origin" not in page.headers
    assert client.get("/api/project", headers={"Host": "evil.example:5000"}).status_code == 400
    loopback_v6 = client.get("/api/project", headers={"Host": "[::1]:5000"})
    assert loopback_v6.status_code != 400  # accepted host; the cookie is scoped to the other host name
    assert client.put("/api/project").status_code == 405
    assert client.delete("/api/project").status_code == 405
    assert client.get("/api/rescan").status_code == 405
    assert client.post("/api/rescan", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/static/").status_code == 404


@pytest.mark.parametrize("path", [
    "/api/region/..%2f..%2fetc",
    "/api/region/%2e%2e",
    "/api/region/C:%5cWindows",
    "/api/region/{rid}/layer/..%2f..%2fmask.nii.gz",
    "/api/region/{rid}/layer/%2Fetc%2Fpasswd",
    "/api/region/{rid}/layer/0123456789ab",
    "/api/download/..%2fmanifest",
    "/api/download/0123456789ab",
    "/static/..%2f..%2fapp.py",
    "/api/summary/..%2fwhole",
])
def test_paths_in_parameters_are_refused(tree, path):
    manifest, output_root, data_root = tree
    client = _client(manifest)
    response = client.get(path.format(rid=_first_region(client)))
    assert response.status_code in {400, 404}
    body = response.get_data(as_text=True)
    assert str(output_root) not in body and str(data_root) not in body and "Traceback" not in body


def test_explicit_remote_mode_accepts_the_reachable_host_name(tree):
    manifest, _, _ = tree
    app = create_app(manifest, access_token=TOKEN, allowed_hosts=("*",))
    client = app.test_client()
    headers = {"Host": "viewer-node.example:8765"}
    assert client.get(f"/?token={TOKEN}", headers=headers).status_code == 303
    assert client.get("/api/project", headers=headers).status_code == 200


def test_link_escape_out_of_the_roots_is_refused(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path / "t", stages=("whole", "voxelwise", "habitats"))
    region = next(r for r in layout.Catalog(manifest).snapshot.regions.values() if r.region == "lesion_002")
    method_dir = layout.habitat_mask_path(output_root, region.key, "fixed_k2", "b32", source_grid=False).parent
    outside = tmp_path / "outside"
    shutil.copytree(method_dir, outside)
    shutil.rmtree(method_dir)
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(outside), str(method_dir))
        else:
            os.symlink(outside, method_dir, target_is_directory=True)
    except (OSError, ImportError, AttributeError) as exc:  # pragma: no cover - platform without links
        pytest.skip(f"cannot create a link here: {exc}")
    client = _client(manifest)
    detail = client.get(f"/api/region/{region.rid}").get_json()
    layer = next(item for item in detail["layers"] if item["kind"] == "habitat_mask_cropped" and item["method"] == "fixed_k2")
    assert layer["available"] is False
    response = client.get(f"/api/region/{region.rid}/layer/{layer['id']}")
    assert response.status_code == 403 and response.get_json()["error"] == "outside_roots"


def test_cohort_table_link_escape_is_not_read(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path / "t", stages=("whole",))
    table = layout.stage_table_paths(output_root, "whole", "b32")["whole_region"]
    outside = tmp_path / "outside-table"
    shutil.copytree(table.parent, outside)
    shutil.rmtree(table.parent)
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(outside), str(table.parent))
        else:
            os.symlink(outside, table.parent, target_is_directory=True)
    except (OSError, ImportError, AttributeError) as exc:  # pragma: no cover - platform without links
        pytest.skip(f"cannot create a link here: {exc}")
    client = _client(manifest)
    rid = _first_region(client)
    response = client.get(f"/api/region/{rid}/features/whole?modality=T1")
    assert response.status_code == 404 and response.get_json()["error"] == "no_whole_row"
    summary = client.get("/api/summary/whole").get_json()
    assert summary["available"] is False and "outside output_root" in summary["message"]
    assert not any(item["name"] == "whole_region.csv" for item in client.get("/api/downloads").get_json()["files"])


def test_bad_files_and_vanishing_files_keep_the_server_up(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path / "t", stages=("whole", "voxelwise"))
    client = _client(manifest)
    rid = _first_region(client)
    detail = client.get(f"/api/region/{rid}").get_json()
    image = next(layer for layer in detail["layers"] if layer["kind"] == "image")
    feature = next(iter(next(iter(detail["feature_maps"].values())).values()))[0]
    region = layout.Catalog(manifest).region(rid)
    entry = next(iter(region.entries.values()))
    image_path = Path(entry.image_path)
    data = image_path.read_bytes()
    image_path.write_bytes(data[: len(data) // 2])
    response = client.get(f"/api/region/{rid}/layer/{image['id']}")
    assert response.status_code == 422 and response.get_json()["error"] == "unreadable"
    map_path = layout.voxelwise_unit_dir(output_root, region.key, feature_modality(detail, feature["id"]), "b32") / f"{feature['feature']}.nrrd"
    map_path.write_bytes(b"NRRD0004\ngarbage")
    assert client.get(f"/api/region/{rid}/layer/{feature['id']}").status_code == 422
    map_path.unlink()
    response = client.get(f"/api/region/{rid}/layer/{feature['id']}/display")
    assert response.status_code == 404 and response.get_json()["error"] == "missing_output"
    stray = output_root / ".tmp" / "whole" / "u-000000000000-00000000"
    stray.mkdir(parents=True)
    (output_root / "tables" / "whole" / "b32" / "whole_region.csv.bak-20990101T000000").write_text("x", encoding="utf-8")
    assert client.post("/api/rescan").status_code == 200
    assert client.get("/api/project").status_code == 200
    downloads = client.get("/api/downloads").get_json()["files"]
    assert not any(".bak-" in item["relative_path"] or ".tmp" in item["relative_path"] for item in downloads)


def test_missing_data_root_degrades_to_output_layers(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path / "t", stages=("whole", "voxelwise"))
    (tmp_path / "t" / "data").rename(tmp_path / "t" / "data-moved")
    client = _client(manifest)
    rid = _first_region(client)
    detail = client.get(f"/api/region/{rid}").get_json()
    images = [layer for layer in detail["layers"] if layer["kind"] == "image"]
    assert images and not any(layer["available"] for layer in images)
    response = client.get(f"/api/region/{rid}/layer/{images[0]['id']}")
    assert response.status_code == 404 and response.get_json()["error"] == "input_not_found"
    feature = next(iter(next(iter(detail["feature_maps"].values())).values()))[0]
    assert client.get(f"/api/region/{rid}/layer/{feature['id']}").status_code == 200
    assert client.get("/api/status").status_code == 200


def test_odd_identifiers_and_empty_manifest_open_without_errors(tmp_path):
    keys = [("sub__a", "ses 01", "région_ü"), ("<b>x</b>", "ses-01", "lesion_001"), ("sub-001", "ses-02", "liver")]
    manifest = support.write_handmade_manifest(tmp_path / "odd", keys)
    client = _client(manifest)
    for path in ("/", "/api/project", "/api/status", "/api/qc", "/api/cohort/regions", "/api/downloads"):
        assert client.get(path).status_code == 200
    regions = client.get("/api/regions").get_json()["regions"]
    assert sorted(row["subject"] for row in regions) == sorted(key[0] for key in keys)
    for row in regions:
        detail = client.get(f"/api/region/{row['rid']}")
        assert detail.status_code == 200
        if row["subject"] == "<b>x</b>":
            assert detail.get_json()["layers"] == [] and detail.get_json()["unsafe_identifier"] is True
    assert "<b>x</b>" not in client.get("/api/regions").get_data(as_text=True)  # JSON escapes markup
    empty = support.write_handmade_manifest(tmp_path / "empty", [("sub-001", "ses-01", "lesion_001")])
    header = empty.read_text(encoding="utf-8").splitlines()[0]
    empty.write_text(header + "\n", encoding="utf-8")
    client = _client(empty)
    assert client.get("/api/status").get_json()["regions"] == []
    assert client.get("/").status_code == 200


def feature_modality(detail, layer_id):
    for modality, groups in detail["feature_maps"].items():
        for items in groups.values():
            if any(item["id"] == layer_id for item in items):
                return modality
    raise AssertionError("layer not found")


def test_tables_distributions_and_downloads_match_the_files(tree):
    manifest, output_root, _ = tree
    client = _client(manifest)
    regions = client.get("/api/regions").get_json()["regions"]
    region = next(item for item in regions if item["region"] == "lesion_001" and item["subject"] == "sub-001")
    with (output_root / "tables" / "whole" / "b32" / "whole_region.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if (r["subject_id"], r["region_id"], r["modality"]) == ("sub-001", "lesion_001", "T1"))
    whole = client.get(f"/api/region/{region['rid']}/features/whole?modality=T1").get_json()
    shown = {entry["name"]: entry["text"] for groups in whole["features"].values() for entries in groups.values() for entry in entries}
    assert shown == {name: row[name] for name in support.ROI_FEATURES}
    values = np.array([float(r["original_firstorder_Mean"]) for r in rows if r["modality"] == "T1" and not r["error"]])
    selected = float(row["original_firstorder_Mean"])
    distribution = client.get(
        f"/api/cohort/distribution?stage=whole&modality=T1&feature=original_firstorder_Mean&rid={region['rid']}"
    ).get_json()
    assert distribution["n"] == len(values) and distribution["n_nonfinite"] == 0
    expected = 100 * (np.sum(values < selected) + 0.5 * np.sum(values == selected)) / len(values)
    assert distribution["selected_percentile"] == pytest.approx(expected)
    assert sum(distribution["histogram"]["counts"]) == len(values)
    with (output_root / "tables" / "habitat_features" / "b32" / "habitat_features.csv").open(encoding="utf-8", newline="") as handle:
        zones = [r for r in csv.DictReader(handle)
                 if (r["subject_id"], r["region_id"], r["method"], r["modality"]) == ("sub-001", "lesion_001", "fixed_k2", "T1")]
    habitat = client.get(f"/api/region/{region['rid']}/features/habitat?method=fixed_k2&modality=T1").get_json()
    assert [(z["zone_name"], z["zone_status"], z["source_zone_voxels"]) for z in habitat["zones"]] == [
        (r["zone_name"], r["zone_status"], r["source_zone_voxels"]) for r in zones
    ]
    for item in client.get("/api/downloads").get_json()["files"]:
        with client.get(f"/api/download/{item['id']}") as response:
            body = response.data
        assert hashlib.sha256(body).hexdigest() == hashlib.sha256((output_root / item["relative_path"]).read_bytes()).hexdigest()


def test_materialization_provenance_is_shown_and_downloadable(tmp_path):
    manifest, output_root = support.build_combined_tree(tmp_path)
    client = _client(manifest)
    region = next(item for item in client.get("/api/regions").get_json()["regions"] if item["region"] == "lesion_002")
    info = client.get(f"/api/region/{region['rid']}").get_json()["materialization"]
    assert info["mode"] == "connected_components" and info["connectivity"] == 26
    assert info["region"]["n_voxels"] == 20 and [item["n_voxels"] for item in info["excluded_regions"]] == [1]
    assert str(output_root) not in json.dumps(info) and str(tmp_path) not in json.dumps(info)
    names = {item["relative_path"].rsplit("/", 1)[-1] for item in client.get("/api/downloads").get_json()["files"]}
    assert {"complete.json", "inventory.csv", "provenance.json"} <= names


def test_blind_mode_hides_identifiers_and_disables_downloads(tree):
    manifest, output_root, data_root = tree
    client = _client(manifest, blind=True)
    texts = [client.get(path).get_data(as_text=True) for path in ("/", "/api/project", "/api/status", "/api/regions", "/api/qc")]
    rid = client.get("/api/regions").get_json()["regions"][0]["rid"]
    texts += [client.get(f"/api/region/{rid}").get_data(as_text=True),
              client.get(f"/api/region/{rid}/features/whole?modality=T1").get_data(as_text=True)]
    for text in texts:
        for secret in ("sub-001", "sub-002", "lesion_001", str(output_root), str(data_root)):
            assert secret not in text
    assert client.get("/api/downloads").get_json()["files"] == []
    assert client.get("/api/download/0123456789ab").status_code == 403
    assert "S001" in texts[3]


def test_project_collapses_repeated_upstream_reference_deviations(tmp_path):
    manifest, output_root, _ = support.build_tree(tmp_path, stages=("whole",))
    metadata_path = layout.stage_report_paths(output_root, "whole", "b32")["run_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reason = "platform is Windows-x86_64; reference platform is Linux-x86_64"
    metadata["reference_compatible"] = False
    metadata["reference_deviations"] = [
        reason,
        reason,
        f"upstream voxelwise unit sub-001__ses-01__lesion_001/T1: {reason}",
        f"upstream voxelwise unit sub-002__ses-01__lesion_001/FLAIR: {reason}",
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    deviations = _client(manifest).get("/api/project").get_json()["stages"]["whole"]["reference_deviations"]
    assert deviations == [
        f"{reason} (2 occurrences)",
        f"2 upstream voxelwise unit(s): {reason}",
    ]
    assert not any("sub-" in item for item in deviations)


def test_error_responses_hide_paths_and_tracebacks(tree, monkeypatch):
    manifest, output_root, data_root = tree
    client = _client(manifest)
    rid = _first_region(client)
    from radiomicviz.viewer import _tables

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"failure while reading {output_root}")

    monkeypatch.setattr(_tables, "group_row", boom)
    import radiomicviz.viewer.app as app_module

    monkeypatch.setattr(app_module, "group_row", boom)
    response = client.get(f"/api/region/{rid}/features/whole?modality=T1")
    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Traceback" not in body and str(output_root) not in body and "reference" in body
    assert client.get("/api/project").status_code == 200
    assert sys.modules.get("flask") is not None
    assert json.loads(body)["error"] == "internal_error"
