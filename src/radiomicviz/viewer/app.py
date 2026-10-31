"""Flask application for the read-only RadiomicsViz viewer.

The client never sends a filesystem path: every region, layer and download is an opaque
identifier resolved by the catalog. Responses never contain absolute machine paths or
tracebacks. Nothing is written beneath data_root or output_root.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file
from flask.json.provider import DefaultJSONProvider

from radiomicviz import __version__
from radiomicviz._integrity import fingerprint_matches
from radiomicviz.viewer import catalog as layout
import numpy as np

from radiomicviz.viewer._tables import TableStore, distribution, feature_parts, group_row, percentile_ranks
from radiomicviz.viewer._volumes import VolumeError, VolumeStore, is_nifti

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
    "connect-src 'self' data: blob:; font-src 'self'; worker-src 'self' blob:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
MASK_KINDS = {"region_mask", "habitat_mask_cropped", "habitat_mask_source_grid"}
GRID_NOTE = (
    "Voxelwise maps and cropped habitat masks share the cropped input-image grid (region "
    "bounding box plus a one-voxel margin, no resampling); source-grid habitat masks, "
    "region masks and images use the full input-image grid. Voxel counts on the two "
    "grids are expected to agree."
)
DISCLAIMER = "Research software. Not for clinical use."
_logger = logging.getLogger("radiomicviz.viewer")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def build_app(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    cache_mb: int = 512,
    verify_integrity: bool = False,
    blind: bool = False,
    access_token: str | None = None,
    allowed_hosts: tuple[str, ...] = (),
) -> Flask:
    catalog = layout.Catalog(manifest_path, variant=variant)
    volumes = VolumeStore(cache_mb)
    tables = TableStore()
    token = access_token or secrets.token_urlsafe(32)
    cookie_name = "radiomicviz_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    hosts = {host.strip("[]").casefold() for host in (*LOOPBACK_HOSTS, *allowed_hosts) if host}
    summaries: dict[str, tuple[tuple, dict[str, Any]]] = {}
    summary_lock = threading.Lock()
    cohort_cache: dict[tuple, Any] = {}
    cohort_lock = threading.Lock()

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.json = _HtmlSafeJSONProvider(app)
    app.config.update(
        MAX_CONTENT_LENGTH=64 * 1024,
        RADIOMICVIZ_CATALOG=catalog,
        RADIOMICVIZ_BLIND=blind,
    )

    def load_table(path: Path):
        """Load a cohort table only when links cannot escape the output tree."""
        return tables.get(path) if catalog.contained(path) else None

    # ------------------------------------------------------------------ privacy helpers
    def scrub(text: object) -> str:
        """Remove machine paths and, in blind mode, identifiers from free text."""
        value = str(text)
        snapshot = catalog.snapshot
        roots = [(snapshot.output_root, "<output_root>")]
        if snapshot.data_root is not None:
            roots.append((snapshot.data_root, "<data_root>"))
        for root, placeholder in roots:
            for variant_text in {str(root), str(root).replace("\\", "/"), root.as_posix()}:
                value = value.replace(variant_text, placeholder)
        value = re.sub(r"(?<![\w<])[A-Za-z]:[\\/][^\s,;'\"<>|]*", "<path>", value)
        value = re.sub(r"(?<![\w<])/(?:home|Users|mnt|gpfs|scratch|data|tmp|var)/[^\s,;'\"<>|]*", "<path>", value)
        if blind:
            for original, replacement in pseudonym_terms():
                value = value.replace(original, replacement)
        return value

    pseudonym_cache: dict[int, tuple[dict[str, tuple[str, str, str]], list[tuple[str, str]]]] = {}

    def pseudonyms() -> dict[str, tuple[str, str, str]]:
        snapshot = catalog.snapshot
        cached = pseudonym_cache.get(id(snapshot))
        if cached is None:
            subjects: dict[str, str] = {}
            sessions: dict[tuple[str, str], str] = {}
            names: dict[str, tuple[str, str, str]] = {}
            for number, rid in enumerate(snapshot.order, start=1):
                region = snapshot.regions[rid]
                subject = subjects.setdefault(region.subject, f"S{len(subjects) + 1:03d}")
                session_count = sum(1 for key in sessions if key[0] == region.subject)
                session = sessions.setdefault(
                    (region.subject, region.session), f"E{session_count + 1:02d}"
                )
                names[rid] = (subject, session, f"R{number:04d}")
            # Subjects are unique; session and region names repeat across subjects, so free
            # text gets generic placeholders for them rather than a possibly wrong alias.
            terms: list[tuple[str, str]] = [(original, alias) for original, alias in subjects.items()]
            terms.extend((key[1], "<session>") for key in sessions)
            terms.extend((snapshot.regions[rid].region, "<region>") for rid in names)
            terms = sorted(set(terms), key=lambda pair: len(pair[0]), reverse=True)
            pseudonym_cache.clear()
            pseudonym_cache[id(snapshot)] = (names, terms)
            cached = pseudonym_cache[id(snapshot)]
        return cached[0]

    def pseudonym_terms() -> list[tuple[str, str]]:
        pseudonyms()
        return pseudonym_cache[id(catalog.snapshot)][1]

    def identity(region: layout.Region) -> dict[str, str]:
        if blind:
            subject, session, name = pseudonyms()[region.rid]
            return {"subject": subject, "session": session, "region": name}
        return {"subject": region.subject, "session": region.session, "region": region.region}

    # ------------------------------------------------------------------ request guards
    @app.before_request
    def guard() -> Response | None:
        if "*" not in hosts and _host_name(request.host or "") not in hosts:
            return _json_error(400, "bad_host", "the Host header is not allowed")
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin.split("://", 1)[-1].split("/", 1)[0] != request.host:
                return _json_error(403, "bad_origin", "cross-origin requests are refused")
        if request.endpoint == "index" and request.args.get("token"):
            return None
        supplied = request.cookies.get(cookie_name, "")
        if not supplied or not hmac.compare_digest(supplied, token):
            if request.path.startswith("/api/"):
                return _json_error(404, "not_found", "not found")
            abort(404)
        return None

    @app.after_request
    def headers(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store, private, max-age=0"
        response.headers.pop("Access-Control-Allow-Origin", None)
        if not response.is_streamed and not response.direct_passthrough:
            # Large in-memory bodies go out in pieces: a single large write followed by the
            # dev server closing the connection lost its tail on the Windows test host.
            body = response.get_data()
            if len(body) > STREAM_CHUNK:
                response.response = _chunks(body)
                response.headers["Content-Length"] = str(len(body))
        return response

    @app.errorhandler(ApiError)
    def api_error(error: ApiError):
        return _json_error(error.status, error.code, scrub(error.message))

    @app.errorhandler(VolumeError)
    def volume_error(error: VolumeError):
        status = 404 if error.code == "missing" else 422
        return _json_error(status, error.code, error.message)

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return _json_error(404, "not_found", "not found")
        return Response("Not found", status=404, mimetype="text/plain")

    @app.errorhandler(405)
    def not_allowed(_error):
        return _json_error(405, "method_not_allowed", "method not allowed")

    @app.errorhandler(413)
    def too_large(_error):
        return _json_error(413, "too_large", "request body too large")

    @app.errorhandler(Exception)
    def unexpected(error: Exception):
        reference = secrets.token_hex(4)
        _logger.error("viewer error %s: %s: %s", reference, type(error).__name__, error)
        return _json_error(500, "internal_error", f"unexpected viewer error (reference {reference})")

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        supplied = request.args.get("token")
        if supplied is not None:
            if not hmac.compare_digest(supplied, token):
                abort(404)
            response = redirect("/", code=303)
            response.set_cookie(cookie_name, token, httponly=True, samesite="Strict", path="/")
            return response
        return render_template(
            "index.html",
            title="RadiomicsViz viewer",
            blind=blind,
            disclaimer=DISCLAIMER,
            version=__version__,
        )

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    # ------------------------------------------------------------------ helpers
    def region_or_404(rid: str) -> layout.Region:
        if not re.fullmatch(r"[0-9a-f]{12,64}", rid or ""):
            raise ApiError(404, "unknown_region", "unknown region")
        region = catalog.region(rid)
        if region is None:
            raise ApiError(404, "unknown_region", "unknown region")
        return region

    def region_summary(region: layout.Region) -> dict[str, Any]:
        statuses = {}
        for stage in layout.STAGES:
            status, reason = region.stage_status(stage)
            statuses[stage] = {"status": status, "reason": scrub(reason)}
        return {
            "rid": region.rid,
            **identity(region),
            "region_type": region.region_type,
            "modalities": list(region.modalities),
            "voxels": region.voxels,
            "stages": statuses,
            "reference_statuses": sorted(region.reference_statuses()),
            "unsafe_identifier": region.unsafe,
        }

    def region_layers(region: layout.Region) -> dict[str, dict[str, Any]]:
        """Every displayable file of a region, keyed by an opaque layer id."""
        layers: dict[str, dict[str, Any]] = {}
        snapshot = catalog.snapshot
        if region.unsafe:
            return layers

        def add(kind: str, path: Path, **info: Any) -> None:
            lid = layout.stable_id(kind, str(path.name), *(str(value) for value in info.values()))
            layers[lid] = {"id": lid, "kind": kind, "path": path, **info}

        masks_seen: set[str] = set()
        for modality, entry in sorted(region.entries.items()):
            add("image", Path(entry.image_path), modality=modality, stage="input")
            if entry.mask_path not in masks_seen:
                masks_seen.add(entry.mask_path)
                add("region_mask", Path(entry.mask_path), modality=modality, stage="input", label=entry.label)
        for unit in region.units.get("voxelwise", []):
            if unit.status != layout.COMPLETE or not isinstance(unit.completion, Mapping):
                continue
            names = unit.completion.get("map_names") or []
            for name in names:
                if isinstance(name, str) and unit.output_dir is not None:
                    add("feature_map", unit.output_dir / f"{name}.nrrd", modality=unit.modality,
                        feature=name, stage="voxelwise")
        habitat_units = region.units.get("habitats", [])
        if habitat_units and habitat_units[0].status == layout.COMPLETE:
            for method in layout.HABITAT_METHODS:
                for source_grid in (False, True):
                    path = layout.habitat_mask_path(
                        snapshot.output_root, region.key, method, snapshot.variant, source_grid=source_grid
                    )
                    add("habitat_mask_source_grid" if source_grid else "habitat_mask_cropped",
                        path, method=method, stage="habitats")
        return layers

    def layer_or_404(region: layout.Region, lid: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{12}", lid or ""):
            raise ApiError(404, "unknown_layer", "unknown layer")
        layer = region_layers(region).get(lid)
        if layer is None:
            raise ApiError(404, "unknown_layer", "unknown layer")
        path: Path = layer["path"]
        if not catalog.inside_roots(path):
            raise ApiError(403, "outside_roots", "the file resolves outside data_root and output_root")
        if not path.is_file():
            raise ApiError(404, "input_not_found" if layer["stage"] == "input" else "missing_output",
                           "the file is not on disk")
        return layer

    def public_layer(layer: Mapping[str, Any]) -> dict[str, Any]:
        path: Path = layer["path"]
        exists = path.is_file() and catalog.inside_roots(path)
        suffix = ".nii.gz" if path.name.casefold().endswith(".nii.gz") else ".nii"
        info = {key: value for key, value in layer.items() if key != "path"}
        info["available"] = exists
        info["name"] = f"{layer['id']}{suffix}"
        return info

    # ------------------------------------------------------------------ API: project and status
    @app.get("/api/project")
    def api_project():
        from radiomicviz.protocols import load_protocol

        snapshot = catalog.snapshot
        protocol = load_protocol()
        presets = {}
        for kind, getter in (("roi", protocol.roi_preset), ("voxelwise", protocol.voxelwise_preset)):
            try:
                preset = getter(snapshot.variant)
            except Exception:  # noqa: BLE001 - an unknown variant is reported, not raised
                continue
            presets[kind] = {
                "preset_id": preset.preset_id,
                "expected_sha256": preset.parameter_sha256,
                "file_sha256": _sha256(Path(preset.parameter_path)),
            }
        stages = {}
        for stage in layout.STAGES:
            metadata = snapshot.stage_metadata.get(stage) or {}
            stages[stage] = {
                "command": layout.STAGE_COMMANDS[stage],
                "ran": bool(metadata),
                "status": metadata.get("status"),
                "reference_compatible": metadata.get("reference_compatible"),
                "reference_deviations": [
                    scrub(item)
                    for item in _summarize_reference_deviations(
                        metadata.get("reference_deviations", []) or []
                    )
                ],
                "software": metadata.get("software"),
                "preset_sha256": metadata.get("preset_sha256") or metadata.get("voxelwise_preset_sha256"),
                "finished_utc": metadata.get("finished_utc"),
                "staging_present": snapshot.staging_active.get(stage, False),
            }
        regions = [snapshot.regions[rid] for rid in snapshot.order]
        manifest_meta = snapshot.manifest_metadata or {}
        return jsonify({
            "package_version": __version__,
            "variant": snapshot.variant,
            "protocol": {"id": protocol.protocol_id, "version": protocol.protocol_version,
                         "sha256": protocol.sha256},
            "presets": presets,
            "roots": {
                "output_root": "hidden" if blind else snapshot.output_root.name,
                "data_root": "hidden" if blind or snapshot.data_root is None else snapshot.data_root.name,
            },
            "manifest": {
                "reference_status": manifest_meta.get("reference_status"),
                "rows": sum(len(region.entries) for region in regions),
                "warnings": manifest_meta.get("warnings"),
            },
            "counts": {
                "subjects": len({region.subject for region in regions}),
                "sessions": len({(region.subject, region.session) for region in regions}),
                "regions": len(regions),
                "region_types": sorted({region.region_type for region in regions}),
                "modalities": sorted({modality for region in regions for modality in region.modalities}),
            },
            "stages": stages,
            "blind": blind,
            "verify_integrity": verify_integrity,
            "changed_since_scan": catalog.changed_since_scan(),
            "disclaimer": DISCLAIMER,
            "grid_note": GRID_NOTE,
        })

    @app.get("/api/status")
    def api_status():
        snapshot = catalog.snapshot
        counts: dict[str, dict[str, int]] = {stage: {} for stage in layout.STAGES}
        matrix = []
        for rid in snapshot.order:
            region = snapshot.regions[rid]
            row = region_summary(region)
            for stage in layout.STAGES:
                status = row["stages"][stage]["status"]
                counts[stage][status] = counts[stage].get(status, 0) + 1
            matrix.append(row)
        return jsonify({"counts": counts, "regions": matrix, "changed_since_scan": catalog.changed_since_scan()})

    @app.get("/api/regions")
    def api_regions():
        snapshot = catalog.snapshot
        args = request.args
        query = (args.get("q") or "").casefold()
        stage = args.get("stage") or ""
        status = args.get("status") or ""
        min_voxels = _int_arg(args.get("min_voxels"))
        max_voxels = _int_arg(args.get("max_voxels"))
        rows = []
        for rid in snapshot.order:
            row = region_summary(snapshot.regions[rid])
            if args.get("subject") and row["subject"] != args.get("subject"):
                continue
            if args.get("session") and row["session"] != args.get("session"):
                continue
            if args.get("region_type") and row["region_type"] != args.get("region_type"):
                continue
            if stage in layout.STAGES and status and row["stages"][stage]["status"] != status:
                continue
            if min_voxels is not None and (row["voxels"] is None or row["voxels"] < min_voxels):
                continue
            if max_voxels is not None and (row["voxels"] is None or row["voxels"] > max_voxels):
                continue
            if query and query not in " ".join((row["subject"], row["session"], row["region"])).casefold():
                continue
            rows.append(row)
        sort = args.get("sort") or "id"
        if sort == "voxels":
            rows.sort(key=lambda item: (item["voxels"] is None, item["voxels"] or 0))
        elif sort == "-voxels":
            rows.sort(key=lambda item: (item["voxels"] is None, -(item["voxels"] or 0)))
        offset = max(0, _int_arg(args.get("offset")) or 0)
        limit = min(1000, max(1, _int_arg(args.get("limit")) or 200))
        return jsonify({"total": len(rows), "offset": offset, "limit": limit, "regions": rows[offset:offset + limit]})

    # ------------------------------------------------------------------ API: one region
    @app.get("/api/region/<rid>")
    def api_region(rid: str):
        region = region_or_404(rid)
        layers = region_layers(region)
        public = [public_layer(layer) for layer in layers.values()]
        feature_maps: dict[str, dict[str, list[dict[str, str]]]] = {}
        for layer in public:
            if layer["kind"] == "feature_map":
                parts = feature_parts(layer["feature"])
                feature_class = parts[1] if parts else "other"
                feature_maps.setdefault(layer["modality"], {}).setdefault(feature_class, []).append(
                    {"id": layer["id"], "feature": layer["feature"], "available": layer["available"]}
                )
        units = []
        for stage, stage_units in region.units.items():
            for unit in stage_units:
                completion = unit.completion or {}
                units.append({
                    "stage": stage,
                    "command": layout.STAGE_COMMANDS[stage],
                    "modality": unit.modality,
                    "method": unit.method,
                    "status": unit.status,
                    "reason": scrub(unit.reason),
                    "reference_status": unit.reference_status,
                    "variant": completion.get("variant", catalog.snapshot.variant),
                    "mask_voxels": completion.get("mask_voxels"),
                    "finished_utc": completion.get("finished_utc"),
                    "schema_version": completion.get("schema_version"),
                    "preset_sha256": completion.get("preset_sha256"),
                })
        habitat_units = region.units.get("habitats", [])
        habitat_rows = []
        if habitat_units and habitat_units[0].status == layout.COMPLETE:
            index = load_table(layout.stage_table_paths(catalog.output_root, "habitats", catalog.variant)["habitat_index"])
            if index is not None:
                for number in range(index.n_rows):
                    row = index.row(number)
                    if (row.get("subject_id"), row.get("session_id"), row.get("region_id")) == region.key:
                        habitat_rows.append({
                            key: row.get(key, "") for key in (
                                "method", "zone_id", "zone_name", "n_voxels", "chosen_k",
                                "silhouette_score", "habitat_valid", "mask_label_value",
                            )
                        })
        checks = [
            {key: scrub(check.get(key, "")) for key in ("check_id", "status", "modality", "measured_value", "threshold", "message")}
            for check in region.checks
        ]
        stage_checks = []
        keys = {layout.unit_key(*region.key)} | {
            unit.unit_key for units_ in region.units.values() for unit in units_
        }
        for stage in layout.STAGES:
            for row in catalog.snapshot.stage_checks.get(stage, []):
                if row.get("unit_key") in keys and row.get("status") in {"WARN", "FAIL"}:
                    stage_checks.append({
                        "stage": stage,
                        **{key: scrub(row.get(key, "")) for key in ("check_id", "status", "measured_value", "threshold", "message")},
                    })
        provenance = []
        if verify_integrity:
            provenance = verify_region(region)
        materialization = materialization_info(region)
        centroid = None
        mask_layer = next((layer for layer in layers.values() if layer["kind"] == "region_mask"), None)
        if mask_layer is not None and mask_layer["path"].is_file() and catalog.inside_roots(mask_layer["path"]):
            try:
                centroid = volumes.centroid(mask_layer["path"], mask_layer.get("label"))
            except VolumeError:
                centroid = None
        return jsonify({
            "centroid_mm": centroid,
            "materialization": materialization,
            **region_summary(region),
            "layers": [layer for layer in public if layer["kind"] != "feature_map"],
            "feature_maps": feature_maps,
            "units": units,
            "habitat_index": habitat_rows,
            "input_checks": checks,
            "stage_checks": stage_checks,
            "integrity": provenance,
            "grid_note": GRID_NOTE,
            "label_order_note": (
                "Habitat labels are ordered from peripheral to central by mean distance from "
                "the region boundary; the names describe geometry only."
            ),
        })

    def materialization_info(region: layout.Region) -> dict[str, Any] | None:
        """Provenance of a mask split by ``lesions materialize`` (None for supplied masks)."""
        if region.unsafe:
            return None
        session_dir = layout.materialized_session_dir(
            catalog.output_root, region.subject, region.session, region.region_type
        )
        complete = catalog.read_json(session_dir / ".materialization" / "complete.json")
        if not isinstance(complete, Mapping):
            return None
        regions = complete.get("regions") if isinstance(complete.get("regions"), list) else []
        entry = next((item for item in regions if isinstance(item, Mapping) and item.get("region_id") == region.region), None)
        metadata = catalog.read_json(session_dir / region.region / "metadata.json")
        info = {
            "stage": "lesions materialize",
            "mode": complete.get("mode"),
            "connectivity": complete.get("connectivity"),
            "foreground_label": complete.get("foreground_label"),
            "minimum_region_voxels": complete.get("minimum_region_voxels"),
            "source_mask": complete.get("source_mask"),
            "source_mask_sha256": complete.get("source_mask_sha256"),
            "created_utc": complete.get("created_utc"),
            "region": entry,
            "excluded_regions": complete.get("excluded_regions") or [],
            "region_metadata": metadata if isinstance(metadata, Mapping) else None,
        }
        return json.loads(scrub(json.dumps(info)))

    @app.get("/api/region/<rid>/layer/<lid>")
    def api_layer(rid: str, lid: str):
        region = region_or_404(rid)
        layer = layer_or_404(region, lid)
        kind, payload, _geometry = volumes.layer(layer["path"])
        if kind == "file":
            return send_file(payload, mimetype="application/octet-stream", conditional=False, max_age=0)
        return Response(payload, mimetype="application/octet-stream")

    @app.get("/api/region/<rid>/layer/<lid>/display")
    def api_layer_display(rid: str, lid: str):
        region = region_or_404(rid)
        layer = layer_or_404(region, lid)
        stats = volumes.stats(layer["path"], layer["kind"])
        geometry = stats.get("geometry", {})
        warnings = []
        if geometry.get("sheared"):
            warnings.append("non-orthogonal direction cosines: the viewer cannot display this volume faithfully")
        elif geometry.get("oblique"):
            warnings.append("oblique orientation: check alignment against the image before interpreting")
        return jsonify({
            "id": layer["id"], "kind": layer["kind"], "stage": layer["stage"],
            "modality": layer.get("modality"), "method": layer.get("method"), "feature": layer.get("feature"),
            "variant": catalog.variant, "stats": stats, "warnings": warnings,
            "served_as": "source NIfTI, unchanged" if is_nifti(layer["path"]) else "converted in memory to float32 NIfTI",
        })

    # ------------------------------------------------------------------ API: tables
    def whole_row(region: layout.Region, modality: str) -> tuple[dict[str, str] | None, tuple[str, ...], str]:
        path = layout.stage_table_paths(catalog.output_root, "whole", catalog.variant)["whole_region"]
        table = load_table(path)
        if table is not None:
            subjects, sessions, names, modalities = (table.text(column) for column in ("subject_id", "session_id", "region_id", "modality"))
            for number in range(table.n_rows):
                if (subjects[number], sessions[number], names[number], modalities[number]) == (*region.key, modality):
                    return table.row(number), table.fields, "tables/whole/<variant>/whole_region.csv"
        return None, (), ""

    @app.get("/api/region/<rid>/features/whole")
    def api_whole_features(rid: str):
        region = region_or_404(rid)
        modality = request.args.get("modality") or (region.modalities[0] if region.modalities else "")
        if modality not in region.entries:
            raise ApiError(404, "unknown_modality", "unknown modality for this region")
        row, fields, source = whole_row(region, modality)
        if row is None:
            raise ApiError(404, "no_whole_row", "no whole-region row for this region and modality; run 'radiomicviz whole' first")
        grouped = group_row(row, fields)
        table = load_table(layout.stage_table_paths(catalog.output_root, "whole", catalog.variant)["whole_region"])
        if table is not None:
            entries = [entry for image_types in grouped["features"].values()
                       for items in image_types.values() for entry in items]
            names = tuple(entry["name"] for entry in entries)
            key = (id(table), modality, names)
            with cohort_lock:
                matrix = cohort_cache.get(key)
            if matrix is None:
                modalities, errors = table.text("modality"), table.text("error")
                selection = [n for n in range(table.n_rows) if modalities[n] == modality and not errors[n]]
                matrix = table.matrix(names, selection)
                with cohort_lock:
                    cohort_cache.clear()
                    cohort_cache[key] = matrix
            values = np.asarray([entry["value"] if entry["value"] is not None else np.nan for entry in entries])
            ranks, counts = percentile_ranks(matrix, values)
            for entry, rank, count in zip(entries, ranks, counts):
                entry["cohort_percentile"] = None if np.isnan(rank) else float(rank)
                entry["cohort_n"] = int(count)
        if blind:
            grouped["identity"] = [_blind_identity(item, region, identity(region)) for item in grouped["identity"]]
        else:
            grouped["identity"] = [{"name": item["name"], "text": scrub(item["text"])} for item in grouped["identity"]]
        return jsonify({
            "stage": "whole", "variant": catalog.variant, "modality": modality, "source": source,
            "error": scrub(row.get("error", "")), "reference_status": row.get("reference_status", ""),
            "percentile_note": "Cohort percentile rank among successful rows of the same modality (viewer-derived).",
            **grouped,
        })

    @app.get("/api/region/<rid>/features/habitat")
    def api_habitat_features(rid: str):
        region = region_or_404(rid)
        method = request.args.get("method") or layout.HABITAT_METHODS[0]
        modality = request.args.get("modality") or layout.HABITAT_MODALITIES[0]
        if method not in layout.HABITAT_METHODS or modality not in layout.HABITAT_MODALITIES:
            raise ApiError(404, "unknown_method", "unknown method or modality")
        path = layout.stage_table_paths(catalog.output_root, "habitat_features", catalog.variant)["habitat_features"]
        table = load_table(path)
        if table is None:
            raise ApiError(404, "no_habitat_features", "no habitat-feature table; run 'radiomicviz habitat-features' first")
        keys = [table.text(column) for column in ("subject_id", "session_id", "region_id", "method", "modality")]
        zones = []
        for number in range(table.n_rows):
            if tuple(column[number] for column in keys) == (*region.key, method, modality):
                row = table.row(number)
                grouped = group_row(row, table.fields)
                zone = {
                    key: scrub(row.get(key, "")) for key in (
                        "zone_id", "zone_name", "zone_status", "error", "source_zone_voxels",
                        "habitat_grid_voxels", "zone_flagged_small", "mask_label_value", "reference_status",
                    )
                }
                zone["features"] = grouped["features"]
                zones.append(zone)
        return jsonify({
            "stage": "habitat_features", "variant": catalog.variant, "method": method, "modality": modality,
            "source": "tables/habitat_features/<variant>/habitat_features.csv", "zones": zones,
            "legacy_note": (
                "Legacy *_reference_format.csv tables are download-only. In them, 'Patient' is a row "
                "number within each table and 'Cluster' is the habitat label value."
            ),
        })

    @app.get("/api/cohort/distribution")
    def api_distribution():
        stage = request.args.get("stage") or "whole"
        modality = request.args.get("modality") or ""
        feature = request.args.get("feature") or ""
        rid = request.args.get("rid") or ""
        if stage not in {"whole", "habitat_features"} or feature_parts(feature) is None:
            raise ApiError(400, "bad_request", "choose stage=whole or habitat_features and a feature column")
        name = "whole_region" if stage == "whole" else "habitat_features"
        table = load_table(layout.stage_table_paths(catalog.output_root, stage, catalog.variant)[name])
        if table is None or feature not in table.index:
            raise ApiError(404, "no_table", "the cohort table or feature column is not on disk")
        modalities = table.text("modality")
        if stage == "whole":
            ok = table.text("error")
            selection = [n for n in range(table.n_rows) if modalities[n] == modality and not ok[n]]
        else:
            methods, zones, statuses = table.text("method"), table.text("zone_name"), table.text("zone_status")
            method, zone_name = request.args.get("method") or "", request.args.get("zone") or ""
            selection = [
                n for n in range(table.n_rows)
                if modalities[n] == modality and methods[n] == method and zones[n] == zone_name
                and statuses[n] == "complete"
            ]
        values = table.numeric(feature)[selection] if selection else table.numeric(feature)[:0]
        selected = None
        if rid:
            region = region_or_404(rid)
            subjects, sessions, names = (table.text(column) for column in ("subject_id", "session_id", "region_id"))
            for number in selection:
                if (subjects[number], sessions[number], names[number]) == region.key:
                    selected = float(table.numeric(feature)[number])
                    break
        result = distribution(values, selected)
        result.update({"stage": stage, "variant": catalog.variant, "modality": modality, "feature": feature})
        return jsonify(result)

    @app.get("/api/cohort/regions")
    def api_cohort_regions():
        snapshot = catalog.snapshot
        sizes = [snapshot.regions[rid].voxels for rid in snapshot.order]
        import numpy as np

        values = np.asarray([size for size in sizes if size is not None], dtype=np.float64)
        result = {"region_voxels": distribution(values, None), "habitat_zone_fractions": {}}
        index = load_table(layout.stage_table_paths(catalog.output_root, "habitats", catalog.variant)["habitat_index"])
        if index is not None:
            by_key: dict[tuple, dict[str, list[float]]] = {}
            subjects, sessions, regions, methods, names, voxels = (
                index.text(column) for column in ("subject_id", "session_id", "region_id", "method", "zone_name", "n_voxels")
            )
            totals: dict[tuple, int] = {}
            for number in range(index.n_rows):
                key = (subjects[number], sessions[number], regions[number], methods[number])
                totals[key] = totals.get(key, 0) + (_int_arg(voxels[number]) or 0)
            fractions: dict[str, list[float]] = {}
            for number in range(index.n_rows):
                key = (subjects[number], sessions[number], regions[number], methods[number])
                count = _int_arg(voxels[number]) or 0
                if totals.get(key) and names[number] != "not_applicable":
                    fractions.setdefault(f"{methods[number]}:{names[number]}", []).append(count / totals[key])
            result["habitat_zone_fractions"] = {
                key: distribution(np.asarray(values_), None, bins=10) for key, values_ in sorted(fractions.items())
            }
        return jsonify(result)

    # ------------------------------------------------------------------ API: QC, summaries, downloads
    @app.get("/api/qc")
    def api_qc():
        snapshot = catalog.snapshot
        wanted_stage = request.args.get("stage") or ""
        wanted_status = request.args.get("status") or ""
        by_key = {layout.unit_key(*snapshot.regions[rid].key): rid for rid in snapshot.order}
        rows = []

        def emit(stage: str, kind: str, row: Mapping[str, str]) -> None:
            status = row.get("status") or ("FAIL" if kind == "error" else "")
            if wanted_stage and stage != wanted_stage:
                return
            if wanted_status and status != wanted_status:
                return
            if row.get("subject_id") or row.get("region_id"):
                key = "__".join((row.get("subject_id", ""), row.get("session_id", ""), row.get("region_id", "")))
            else:
                key = "__".join((row.get("unit_key") or "").split("__")[:3])
            rid = by_key.get(key)
            rows.append({
                "stage": stage, "kind": kind, "status": status, "rid": rid,
                "code": scrub(row.get("check_id") or row.get("error_code") or ""),
                "modality": row.get("modality", ""), "method": row.get("method", ""),
                "measured_value": scrub(row.get("measured_value", "")),
                "threshold": scrub(row.get("threshold", "")),
                "message": scrub(row.get("message", "")),
            })

        for row in snapshot.input_checks:
            if row.get("status") in {"WARN", "FAIL"}:
                emit("manifest", "check", row)
        for row in snapshot.manifest_errors:
            emit("manifest", "error", row)
        for stage in layout.STAGES:
            for row in snapshot.stage_checks.get(stage, []):
                if row.get("status") in {"WARN", "FAIL"}:
                    emit(stage, "check", row)
            for row in snapshot.stage_errors.get(stage, []):
                emit(stage, "error", row)
        offset = max(0, _int_arg(request.args.get("offset")) or 0)
        limit = min(2000, max(1, _int_arg(request.args.get("limit")) or 500))
        return jsonify({"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]})

    @app.get("/api/summary/<stage>")
    def api_summary(stage: str):
        if stage not in layout.STAGES:
            raise ApiError(404, "unknown_stage", "unknown stage")
        snapshot = catalog.snapshot
        key = (id(snapshot), catalog.current_signature())
        with summary_lock:
            cached = summaries.get(stage)
            if cached is not None and cached[0] == key:
                return jsonify(cached[1])
        result = compute_summary(stage)
        with summary_lock:
            summaries[stage] = (key, result)
        return jsonify(result)

    def compute_summary(stage: str) -> dict[str, Any]:
        sources = [
            *layout.stage_table_paths(catalog.output_root, stage, catalog.variant).values(),
            *layout.stage_report_paths(catalog.output_root, stage, catalog.variant).values(),
        ]
        if any(path.exists() and not catalog.contained(path) for path in sources):
            return {
                "stage": stage,
                "available": False,
                "message": "a summary input resolves outside output_root",
                "viewer_computed": True,
            }
        function: Callable[..., Any]
        if stage == "whole":
            from radiomicviz.whole import summarize_whole as function
        elif stage == "voxelwise":
            from radiomicviz.voxelwise import summarize_voxelwise as function
        elif stage == "habitats":
            from radiomicviz.habitats import summarize_habitats as function
        else:
            from radiomicviz.habitat_features import summarize_habitat_features as function
        temporary = Path(tempfile.mkdtemp(prefix="radiomicviz-viewer-"))
        try:
            result = function(catalog.manifest_path, variant=catalog.variant, summary_dir=temporary / "summary")
            report = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
            markdown = Path(result.markdown_path).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - the stage explains why no summary exists
            return {"stage": stage, "available": False, "message": scrub(str(exc)), "viewer_computed": True}
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return {
            "stage": stage, "available": True, "viewer_computed": True,
            "note": "Computed now by the stage's own summary engine into a temporary folder; nothing was saved.",
            "report": json.loads(scrub(json.dumps(report))),
            "markdown": scrub(markdown),
        }

    def download_targets() -> dict[str, tuple[str, Path]]:
        snapshot = catalog.snapshot
        targets: dict[str, tuple[str, Path]] = {}
        candidates: list[tuple[str, Path]] = [
            ("manifest", snapshot.manifest_path),
            ("manifest", snapshot.manifest_path.with_name("input_checks.csv")),
            ("manifest", snapshot.manifest_path.with_name("errors.csv")),
        ]
        sessions = {
            (region.subject, region.session, region.region_type)
            for region in snapshot.regions.values() if not region.unsafe
        }
        for subject, session, region_type in sorted(sessions):
            provenance = layout.materialized_session_dir(snapshot.output_root, subject, session, region_type) / ".materialization"
            for name in ("complete.json", "inventory.csv", "provenance.json"):
                candidates.append(("materialize", provenance / name))
        for stage in layout.STAGES:
            for path in layout.stage_table_paths(snapshot.output_root, stage, snapshot.variant).values():
                candidates.append((stage, path))
            for path in layout.stage_report_paths(snapshot.output_root, stage, snapshot.variant).values():
                candidates.append((stage, path))
            directory = layout.summary_dir(snapshot.output_root, stage, snapshot.variant)
            try:
                summary_files = sorted(entry for entry in directory.iterdir() if entry.is_file())
            except OSError:
                summary_files = []
            for path in summary_files:
                if not path.name.startswith(".") and ".bak-" not in path.name and not path.name.endswith(".tmp"):
                    candidates.append((stage, path))
        for stage, path in candidates:
            if path.is_file() and catalog.inside_roots(path):
                did = layout.stable_id("download", path.relative_to(snapshot.output_root).as_posix())
                targets[did] = (stage, path)
        return targets

    @app.get("/api/downloads")
    def api_downloads():
        if blind:
            return jsonify({"blind": True, "files": [], "note": "Downloads are disabled in blind mode because the files contain identifiers."})
        snapshot = catalog.snapshot
        files = [
            {"id": did, "stage": stage, "name": path.name,
             "relative_path": path.relative_to(snapshot.output_root).as_posix(),
             "size_bytes": path.stat().st_size}
            for did, (stage, path) in download_targets().items()
        ]
        return jsonify({"blind": False, "files": files})

    @app.get("/api/download/<did>")
    def api_download(did: str):
        if blind:
            raise ApiError(403, "blind_mode", "downloads are disabled in blind mode")
        if not re.fullmatch(r"[0-9a-f]{12}", did or ""):
            raise ApiError(404, "unknown_download", "unknown download")
        target = download_targets().get(did)
        if target is None:
            raise ApiError(404, "unknown_download", "unknown download")
        return send_file(target[1], as_attachment=True, download_name=target[1].name, conditional=False, max_age=0)

    # ------------------------------------------------------------------ API: rescan and verify
    @app.post("/api/rescan")
    def api_rescan():
        before = len(catalog.snapshot.order)
        snapshot = catalog.rescan()
        return jsonify({"regions_before": before, "regions": len(snapshot.order), "changed_since_scan": False})

    def verify_region(region: layout.Region) -> list[dict[str, Any]]:
        results = []
        for stage, units in region.units.items():
            for unit in units:
                completion = unit.completion
                if not isinstance(completion, Mapping) or unit.completion_path is None:
                    continue
                for label, recorded, path in _fingerprints(stage, completion, unit, region, catalog):
                    if not catalog.inside_roots(path):
                        ok = False
                    else:
                        ok = fingerprint_matches(recorded, path)
                    results.append({"stage": stage, "modality": unit.modality, "method": unit.method,
                                    "file": label, "verified": bool(ok)})
        return results

    @app.post("/api/region/<rid>/verify")
    def api_verify(rid: str):
        region = region_or_404(rid)
        results = verify_region(region)
        return jsonify({
            "rid": region.rid, "checked": len(results),
            "failed": sum(not item["verified"] for item in results), "results": results,
            "method": "SHA-256 and size compared with each completion record (path and mtime ignored)",
        })

    return app


# ---------------------------------------------------------------------------- utilities
class _HtmlSafeJSONProvider(DefaultJSONProvider):
    """JSON whose text never contains raw ``<``, ``>`` or ``&`` (same values, escaped)."""

    sort_keys = False

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        text = super().dumps(obj, **kwargs)
        return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


STREAM_CHUNK = 64 * 1024


def _chunks(body: bytes):
    for start in range(0, len(body), STREAM_CHUNK):
        yield body[start:start + STREAM_CHUNK]


def _summarize_reference_deviations(values: Any, *, limit: int = 20) -> list[str]:
    """Collapse repeated and per-unit deviations into a readable cohort-level list."""
    exact: dict[str, int] = {}
    upstream: dict[tuple[str, str], int] = {}
    order: list[tuple[str, object]] = []
    for value in values if isinstance(values, (list, tuple)) else ():
        text = str(value).strip()
        if not text:
            continue
        match = re.fullmatch(r"upstream ([^:]+?) unit .*?: (.+)", text)
        if match:
            key = (match.group(1), match.group(2))
            if key not in upstream:
                order.append(("upstream", key))
            upstream[key] = upstream.get(key, 0) + 1
        else:
            if text not in exact:
                order.append(("exact", text))
            exact[text] = exact.get(text, 0) + 1
    result: list[str] = []
    for kind, key in order:
        if kind == "upstream":
            stage, reason = key  # type: ignore[misc]
            count = upstream[(stage, reason)]
            result.append(f"{count} upstream {stage} unit(s): {reason}")
        else:
            text = str(key)
            count = exact[text]
            result.append(text if count == 1 else f"{text} ({count} occurrences)")
    if len(result) > limit:
        omitted = len(result) - limit
        result = result[:limit] + [f"{omitted} additional distinct deviation(s)"]
    return result


def _json_error(status: int, code: str, message: str) -> Response:
    response = jsonify({"error": code, "message": message})
    response.status_code = status
    return response


def _int_arg(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _host_name(host: str) -> str:
    """Host part of a Host header, without port or IPv6 brackets, case-folded."""
    if host.startswith("["):
        return host[1:].split("]", 1)[0].casefold()
    return host.rsplit(":", 1)[0].casefold() if host.count(":") == 1 else host.casefold()


def _fingerprints(stage: str, completion: Mapping[str, Any], unit: Any, region: Any, catalog: Any):
    """Yield (label, recorded fingerprint, current path) for files a completion records.

    Output paths are re-derived from the current output root (the tree may have moved);
    inputs use the recorded paths, which the stages require to stay in place.
    """
    for name in ("input_image", "input_mask"):
        recorded = completion.get(name)
        if isinstance(recorded, Mapping) and isinstance(recorded.get("path"), str):
            yield name, recorded, Path(recorded["path"])
    unit_dir = unit.completion_path.parent
    groups = [("outputs", False)]
    if stage == "habitats":
        groups.append(("source_grid_outputs", True))
    for group, source_grid in groups:
        outputs = completion.get(group)
        if not isinstance(outputs, Mapping):
            continue
        for name, recorded in outputs.items():
            if not isinstance(recorded, Mapping):
                continue
            if stage == "voxelwise":
                path = unit_dir / f"{name}.nrrd"
            elif stage == "habitats" and name in layout.HABITAT_METHODS:
                path = layout.habitat_mask_path(
                    catalog.output_root, region.key, name, catalog.variant, source_grid=source_grid
                )
                name = f"{name}{' (source grid)' if source_grid else ''}"
            else:
                path = unit_dir / Path(str(recorded.get("path", ""))).name
            yield str(name), recorded, path


def _blind_identity(item: Mapping[str, str], region: layout.Region, names: Mapping[str, str]) -> dict[str, str]:
    hidden = {
        "subject_id": names["subject"], "session_id": names["session"], "region_id": names["region"],
        "image_path": "hidden", "mask_path": "hidden",
    }
    return {"name": item["name"], "text": hidden.get(item["name"], item["text"])}
