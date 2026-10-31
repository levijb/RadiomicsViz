/* RadiomicsViz read-only viewer. Every user-controlled string is rendered with textContent. */
"use strict";
(function () {
  const BLIND = document.body.dataset.blind === "true";
  const STAGES = ["whole", "voxelwise", "habitats", "habitat_features"];
  const STAGE_LABEL = { whole: "whole", voxelwise: "voxelwise", habitats: "habitats", habitat_features: "habitat-features" };
  const STATUSES = ["complete", "skipped", "failed", "blocked", "in-progress", "missing", "not-applicable"];
  const MAP_COLORMAPS = ["viridis", "plasma", "inferno", "hot", "warm", "cool", "gray"];
  const LABEL_COLORS = [[0, 0, 0], [230, 75, 53], [77, 187, 213], [0, 160, 135], [243, 155, 18]];
  const state = {
    project: null, regions: [], total: 0, offset: 0, rid: null, detail: null, nv: null,
    volumes: { background: null, mask: null, habitat: null, feature: null },
    featureLayers: [], featureIndex: -1, lockedRange: null, statsCache: {}, readoutNames: {},
    wholeData: null, habitatData: null, loadToken: 0,
  };

  // ------------------------------------------------------------------ helpers
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    }
    for (const child of children) {
      if (child === null || child === undefined) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
  function option(value, label, selected) {
    const node = el("option", { value: value, text: label === undefined ? value : label });
    if (selected) node.selected = true;
    return node;
  }
  function statusChip(status, reason) {
    return el("span", { class: "status status-" + status, title: reason || "", text: status });
  }
  function fmt(value) {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "number") {
      if (!isFinite(value)) return String(value);
      const magnitude = Math.abs(value);
      return magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6) ? value.toExponential(4) : String(Number(value.toPrecision(6)));
    }
    return String(value);
  }
  class ApiError extends Error { constructor(message, code) { super(message); this.code = code; } }
  async function api(path, options) {
    const response = await fetch(path, Object.assign({ credentials: "same-origin", headers: { Accept: "application/json" } }, options || {}));
    let body = null;
    try { body = await response.json(); } catch (error) { body = null; }
    if (!response.ok) throw new ApiError(body && body.message ? body.message : "request failed (" + response.status + ")", body && body.error);
    if (body === null) throw new ApiError("incomplete response from the viewer server", "incomplete");
    return body;
  }
  function showError(container, error) {
    clear(container).append(el("p", { class: "fail-text", text: "Could not load: " + error.message }));
  }
  function regionName(row) { return row.subject + " / " + row.session + " / " + row.region; }
  function safeFileName(text) { return String(text).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120); }

  // ------------------------------------------------------------------ tabs
  for (const button of document.querySelectorAll(".tabs button")) {
    button.addEventListener("click", () => selectTab(button.dataset.tab));
  }
  function selectTab(name) {
    for (const button of document.querySelectorAll(".tabs button")) button.classList.toggle("active", button.dataset.tab === name);
    for (const tab of document.querySelectorAll(".tab")) tab.classList.toggle("active", tab.id === "tab-" + name);
    if (name === "qc") loadQc();
    if (name === "cohort") loadCohortRegions();
    if (name === "regions") {
      if (state.pendingRid) { const rid = state.pendingRid; state.pendingRid = null; openRegion(rid, true); }
      else if (state.nv) { if (state.nv.resizeListener) state.nv.resizeListener(); state.nv.drawScene(); }
    }
  }
  function regionsVisible() { return $("tab-regions").classList.contains("active"); }
  for (const button of document.querySelectorAll(".subtabs button")) {
    button.addEventListener("click", () => {
      for (const other of document.querySelectorAll(".subtabs button")) other.classList.toggle("active", other === button);
      for (const panel of document.querySelectorAll(".subtab")) panel.classList.toggle("active", panel.id === "sub-" + button.dataset.sub);
    });
  }

  // ------------------------------------------------------------------ overview
  async function loadProject() {
    state.project = await api("/api/project");
    const project = state.project;
    $("project-line").textContent = project.protocol.id + " " + project.protocol.version + ", variant " + project.variant +
      ", " + project.counts.regions + " regions";
    const table = clear($("project-table"));
    const rows = [
      ["Subjects", project.counts.subjects], ["Sessions", project.counts.sessions],
      ["Regions", project.counts.regions], ["Region types", project.counts.region_types.join(", ")],
      ["Modalities", project.counts.modalities.join(", ")],
      ["Protocol", project.protocol.id + " " + project.protocol.version], ["Variant", project.variant],
      ["Manifest input status", project.manifest.reference_status || "unknown"],
      ["Output folder", project.roots.output_root], ["Data folder", project.roots.data_root],
    ];
    for (const [kind, preset] of Object.entries(project.presets)) {
      const ok = preset.expected_sha256 === preset.file_sha256;
      rows.push([kind + " preset", preset.preset_id + (ok ? " (SHA-256 matches the frozen value)" : " (SHA-256 MISMATCH)")]);
    }
    for (const [label, value] of rows) table.append(el("tr", null, el("td", { text: label }), el("td", { text: value })));
    renderBanners();
  }
  function renderBanners() {
    const project = state.project;
    const box = clear($("banners"));
    const nonReference = [];
    for (const [stage, info] of Object.entries(project.stages)) {
      if (info.ran && info.reference_compatible === false) nonReference.push(STAGE_LABEL[stage] + ": " + (info.reference_deviations || []).join("; "));
    }
    if (project.manifest.reference_status && project.manifest.reference_status !== "reference-compatible") {
      nonReference.push("manifest: " + project.manifest.reference_status);
    }
    if (nonReference.length) {
      box.append(el("div", { class: "banner fail" }, el("strong", { text: "Non-reference run. " }),
        "Results were explicitly allowed to differ from the protocol: " + nonReference.join(" | ")));
    }
    const missing = Object.entries(project.stages).filter(([, info]) => !info.ran).map(([stage]) => STAGE_LABEL[stage]);
    if (missing.length) box.append(el("div", { class: "banner info", text: "Partial outputs: no run metadata yet for " + missing.join(", ") + "." }));
    const staging = Object.entries(project.stages).filter(([, info]) => info.staging_present).map(([stage]) => STAGE_LABEL[stage]);
    if (staging.length) box.append(el("div", { class: "banner warn", text: "A run may be in progress (staging folders present): " + staging.join(", ") + ". Rescan when it finishes." }));
    if (project.changed_since_scan) box.append(el("div", { class: "banner warn", text: "Outputs changed since the last scan. Use Rescan." }));
    if (BLIND) box.append(el("div", { class: "banner info", text: "Blind mode: identifiers are replaced by pseudonyms and downloads are disabled." }));
  }
  async function loadStatus() {
    const data = await api("/api/status");
    const counts = clear($("stage-counts"));
    counts.append(el("tr", null, el("th", { text: "Stage" }), ...STATUSES.map((status) => el("th", { text: status }))));
    for (const stage of STAGES) {
      counts.append(el("tr", null, el("td", { text: STAGE_LABEL[stage] }), ...STATUSES.map((status) => el("td", { text: data.counts[stage][status] || "" }))));
    }
    const matrix = clear($("status-matrix"));
    matrix.append(el("tr", null, el("th", { text: "Region" }), el("th", { text: "Type" }), el("th", { text: "Voxels" }),
      ...STAGES.map((stage) => el("th", { text: STAGE_LABEL[stage] }))));
    for (const row of data.regions.slice(0, 2000)) {
      const link = el("a", { href: "#", text: regionName(row), onclick: (event) => { event.preventDefault(); openRegion(row.rid); } });
      matrix.append(el("tr", null, el("td", null, link), el("td", { text: row.region_type }), el("td", { text: row.voxels === null ? "" : row.voxels }),
        ...STAGES.map((stage) => el("td", null, statusChip(row.stages[stage].status, row.stages[stage].reason)))));
    }
    fillFilters(data.regions);
  }
  function fillFilters(rows) {
    const subjects = [...new Set(rows.map((row) => row.subject))].sort();
    const sessions = [...new Set(rows.map((row) => row.session))].sort();
    const types = [...new Set(rows.map((row) => row.region_type))].sort();
    for (const [id, values] of [["f-subject", subjects], ["f-session", sessions], ["f-type", types]]) {
      const select = $(id);
      const current = select.value;
      clear(select).append(option("", "all"));
      for (const value of values) select.append(option(value, value, value === current));
    }
    const stage = $("f-stage");
    if (stage.options.length === 1) {
      for (const name of STAGES) stage.append(option(name, STAGE_LABEL[name]));
      for (const name of STATUSES) $("f-status").append(option(name));
    }
  }
  $("rescan").addEventListener("click", async () => {
    try {
      await api("/api/rescan", { method: "POST" });
      await loadProject();
      await loadStatus();
      await loadRegions(true);
      if (state.rid) await openRegion(state.rid, true);
    } catch (error) { showError($("banners"), error); }
  });

  // ------------------------------------------------------------------ region list
  function filterQuery(offset) {
    const params = new URLSearchParams();
    const pairs = [["q", "f-q"], ["subject", "f-subject"], ["session", "f-session"], ["region_type", "f-type"],
      ["stage", "f-stage"], ["status", "f-status"], ["min_voxels", "f-min"], ["max_voxels", "f-max"], ["sort", "f-sort"]];
    for (const [name, id] of pairs) if ($(id).value) params.set(name, $(id).value);
    params.set("offset", offset); params.set("limit", 200);
    return "/api/regions?" + params.toString();
  }
  async function loadRegions(reset) {
    if (reset) { state.offset = 0; state.regions = []; }
    const data = await api(filterQuery(state.offset));
    state.total = data.total;
    state.regions = state.regions.concat(data.regions);
    state.offset = state.regions.length;
    const list = clear($("region-items"));
    for (const row of state.regions) {
      const dots = el("span", { class: "dots" }, ...STAGES.map((stage) => statusChip(row.stages[stage].status.charAt(0), row.stages[stage].status + ": " + row.stages[stage].reason)));
      const item = el("li", { "data-rid": row.rid, tabindex: "0" }, el("span", { text: regionName(row) }), dots,
        el("div", { class: "muted", text: row.region_type + (row.voxels === null ? "" : ", " + row.voxels + " voxels") }));
      if (row.rid === state.rid) item.classList.add("selected");
      item.addEventListener("click", () => openRegion(row.rid));
      list.append(item);
    }
    $("region-count").textContent = state.total + " regions match";
    $("more-regions").hidden = state.regions.length >= state.total;
  }
  for (const id of ["f-subject", "f-session", "f-type", "f-stage", "f-status", "f-sort"]) $(id).addEventListener("change", () => loadRegions(true));
  for (const id of ["f-q", "f-min", "f-max"]) $(id).addEventListener("input", debounce(() => loadRegions(true), 250));
  $("more-regions").addEventListener("click", () => loadRegions(false));
  function debounce(fn, ms) { let timer = null; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); }; }

  // ------------------------------------------------------------------ NiiVue
  function ensureViewer() {
    if (state.nv) return state.nv;
    if (!window.niivue || !window.niivue.Niivue) throw new Error("the bundled NiiVue library did not load");
    const nv = new window.niivue.Niivue({
      backColor: [0, 0, 0, 1], show3Dcrosshair: true, isOrientationTextVisible: true,
      isNearestInterpolation: true, crosshairColor: [1, 0.85, 0, 1], crosshairWidth: 0.5, isColorbar: true,
    });
    nv.attachToCanvas($("gl"));
    nv.setInterpolation(true);
    nv.onLocationChange = (location) => renderReadout(location);
    state.nv = nv;
    return nv;
  }
  function layerUrl(layer) { return "/api/region/" + state.rid + "/layer/" + layer.id; }
  async function layerStats(layer) {
    const key = state.rid + ":" + layer.id;
    if (!state.statsCache[key]) state.statsCache[key] = await api(layerUrl(layer) + "/display");
    return state.statsCache[key];
  }
  async function addVolume(role, layer, options) {
    const nv = ensureViewer();
    await removeVolume(role);
    if (!layer) return null;
    const volume = await nv.addVolumeFromUrl(Object.assign({ url: layerUrl(layer), name: layer.name }, options));
    state.volumes[role] = volume;
    state.readoutNames[volume.id] = role === "feature" ? layer.feature : role === "habitat" ? "habitat " + layer.method : role === "mask" ? "region mask" : "image " + layer.modality;
    return volume;
  }
  async function removeVolume(role) {
    const volume = state.volumes[role];
    if (volume && state.nv) state.nv.removeVolume(volume);
    state.volumes[role] = null;
  }
  function applyWindow(volume, low, high) {
    volume.cal_min = low;
    volume.cal_max = high;
    state.nv.updateGLVolume();
  }
  async function loadBackground() {
    const layer = state.detail.layers.find((item) => item.id === $("c-background").value);
    if (!layer || !layer.available) { await removeVolume("background"); return; }
    const stats = await layerStats(layer);
    const volume = await addVolume("background", layer, { colormap: "gray", opacity: 1 });
    applyWindow(volume, stats.stats.cal_min, stats.stats.cal_max);
    showWarnings(layer, stats);
    await loadMask();
    await loadHabitat();
    await loadFeature();
  }
  async function loadMask() {
    const layer = state.detail.layers.find((item) => item.kind === "region_mask" && item.available);
    if (!$("c-mask").checked || !layer) { await removeVolume("mask"); state.nv && state.nv.drawScene(); return; }
    const volume = await addVolume("mask", layer, { colormap: "red", opacity: Number($("c-mask-opacity").value), colorbarVisible: false });
    applyWindow(volume, 0.5, 1);
  }
  async function loadHabitat() {
    const method = $("c-habitat").value;
    const kind = $("c-habitat-grid").value === "source" ? "habitat_mask_source_grid" : "habitat_mask_cropped";
    const layer = state.detail.layers.find((item) => item.kind === kind && item.method === method && item.available);
    clear($("habitat-legend"));
    if (!method || !layer) { await removeVolume("habitat"); return; }
    const volume = await addVolume("habitat", layer, { colormap: "gray", opacity: Number($("c-habitat-opacity").value), colorbarVisible: false });
    const zones = state.detail.habitat_index.filter((row) => row.method === method);
    const maxLabel = Math.max(1, ...zones.map((row) => Number(row.mask_label_value) || 0));
    const lut = { R: [], G: [], B: [], A: [], I: [], labels: [] };
    for (let value = 0; value <= maxLabel; value++) {
      const color = LABEL_COLORS[value % LABEL_COLORS.length];
      const zone = zones.find((row) => Number(row.mask_label_value) === value);
      lut.R.push(color[0]); lut.G.push(color[1]); lut.B.push(color[2]); lut.A.push(value === 0 ? 0 : 255); lut.I.push(value);
      lut.labels.push(zone ? zone.zone_name : (value === 0 ? "" : "label " + value));
    }
    volume.setColormapLabel(lut);
    state.nv.updateGLVolume();
    const legend = $("habitat-legend");
    for (let value = 1; value <= maxLabel; value++) {
      const swatch = el("i");
      swatch.style.background = "rgb(" + LABEL_COLORS[value % LABEL_COLORS.length].join(",") + ")";
      legend.append(el("span", null, swatch, lut.labels[value]));
    }
    legend.append(el("div", { class: "muted", text: state.detail.label_order_note }));
  }
  async function loadFeature(keepWindow) {
    const id = $("c-map-feature").value;
    const layer = state.featureLayers.find((item) => item.id === id);
    if (!$("c-map-modality").value || !layer || !layer.available) { await removeVolume("feature"); updateLayerInfo(); return; }
    const stats = await layerStats(layer);
    const volume = await addVolume("feature", layer, { colormap: $("c-map-cmap").value || "viridis", opacity: Number($("c-map-opacity").value) });
    let low = stats.stats.cal_min, high = stats.stats.cal_max;
    if ($("c-map-lock").checked && state.lockedRange) { [low, high] = state.lockedRange; }
    else if (keepWindow && $("c-map-min").value !== "" && $("c-map-max").value !== "") { low = Number($("c-map-min").value); high = Number($("c-map-max").value); }
    $("c-map-min").value = fmt(low); $("c-map-max").value = fmt(high);
    applyWindow(volume, low, high);
    applyThreshold();
    showWarnings(layer, stats);
    updateLayerInfo();
  }
  function applyThreshold() {
    const volume = state.volumes.feature;
    if (!volume) return;
    volume.colormapType = $("c-map-threshold").checked ? 1 : 0;
    state.nv.updateGLVolume();
  }
  function showWarnings(layer, stats) {
    const box = $("layer-warnings");
    for (const warning of stats.warnings || []) box.append(el("p", { class: "warn-text", text: (layer.feature || layer.kind) + ": " + warning }));
    if (stats.stats && stats.stats.all_nonfinite) box.append(el("p", { class: "warn-text", text: (layer.feature || layer.kind) + ": no finite voxels; nothing to display." }));
  }
  function renderReadout(location) {
    const box = clear($("readout"));
    if (!location) return;
    const mm = location.mm || [];
    box.append(el("div", { text: "world (mm, RAS): " + Array.from(mm).slice(0, 3).map((value) => fmt(Number(value))).join(", ") }));
    for (const item of location.values || []) {
      const name = state.readoutNames[item.id] || item.name || "";
      box.append(el("div", { text: name + ": " + fmt(Number(item.value)) }));
    }
  }

  // ------------------------------------------------------------------ region view
  async function openRegion(rid, keepTab) {
    state.rid = rid;
    for (const item of document.querySelectorAll("#region-items li")) item.classList.toggle("selected", item.dataset.rid === rid);
    if (!keepTab) { state.pendingRid = null; selectTab("regions"); }
    if (!regionsVisible()) { state.pendingRid = rid; return; }
    const token = ++state.loadToken;
    let detail;
    try { detail = await api("/api/region/" + rid); } catch (error) { showError($("region-view"), error); return; }
    if (token !== state.loadToken) return;
    state.detail = detail;
    state.statsCache = {};
    $("region-empty").hidden = true;
    $("region-view").hidden = false;
    $("region-title").textContent = regionName(detail) + " (" + detail.region_type + ")";
    const badges = clear($("region-badges"));
    for (const stage of STAGES) badges.append(el("span", { text: " " + STAGE_LABEL[stage] + " " }), statusChip(detail.stages[stage].status, detail.stages[stage].reason));
    if (detail.reference_statuses.some((value) => value !== "reference-compatible")) badges.append(el("span", { class: "badge-nonref", text: " non-reference" }));
    clear($("layer-warnings"));
    if (detail.unsafe_identifier) $("layer-warnings").append(el("p", { class: "fail-text", text: "This region has an identifier that is not a plain folder name; its outputs are not read." }));
    $("grid-note").textContent = detail.grid_note;
    const background = clear($("c-background"));
    for (const layer of detail.layers.filter((item) => item.kind === "image")) {
      background.append(option(layer.id, layer.modality + (layer.available ? "" : " (input not found)")));
    }
    const habitat = clear($("c-habitat"));
    habitat.append(option("", "none"));
    for (const method of [...new Set(detail.layers.filter((item) => item.kind.startsWith("habitat")).map((item) => item.method))]) habitat.append(option(method));
    const mapModality = clear($("c-map-modality"));
    mapModality.append(option("", "none"));
    for (const modality of Object.keys(detail.feature_maps).sort()) mapModality.append(option(modality));
    if (Object.keys(detail.feature_maps).length) mapModality.value = Object.keys(detail.feature_maps).sort()[0];
    fillFeatureSelect();
    fillTablesControls();
    try {
      ensureViewer();
      await loadBackground();
    } catch (error) { $("layer-warnings").append(el("p", { class: "fail-text", text: "Viewer: " + error.message })); }
    await Promise.all(detail.layers.filter((layer) => layer.available).map((layer) => layerStats(layer).catch(() => null)));
    if (token !== state.loadToken) return;
    updateLayerInfo();
    renderUnits();
    renderZones();
    loadWholeTable();
    loadHabitatTable();
    document.title = "RadiomicsViz viewer: " + (BLIND ? detail.region : regionName(detail));
  }
  function fillFeatureSelect() {
    const modality = $("c-map-modality").value;
    const query = $("c-map-search").value.toLowerCase();
    const select = clear($("c-map-feature"));
    state.featureLayers = [];
    const groups = (state.detail && state.detail.feature_maps[modality]) || {};
    for (const featureClass of Object.keys(groups).sort()) {
      const group = el("optgroup", { label: featureClass });
      for (const item of groups[featureClass]) {
        if (query && !item.feature.toLowerCase().includes(query)) continue;
        state.featureLayers.push(Object.assign({ name: item.id + ".nii", modality: modality, kind: "feature_map", stage: "voxelwise" }, item));
        group.append(option(item.id, item.feature + (item.available ? "" : " (missing)")));
      }
      if (group.children.length) select.append(group);
    }
  }
  $("c-background").addEventListener("change", () => loadBackground());
  $("c-mask").addEventListener("change", () => loadMask());
  $("c-mask-opacity").addEventListener("input", () => { const volume = state.volumes.mask; if (volume) { volume.opacity = Number($("c-mask-opacity").value); state.nv.updateGLVolume(); } });
  $("c-habitat").addEventListener("change", () => loadHabitat());
  $("c-habitat-grid").addEventListener("change", () => loadHabitat());
  $("c-habitat-opacity").addEventListener("input", () => { const volume = state.volumes.habitat; if (volume) { volume.opacity = Number($("c-habitat-opacity").value); state.nv.updateGLVolume(); } });
  $("c-map-modality").addEventListener("change", () => { fillFeatureSelect(); loadFeature(); });
  $("c-map-search").addEventListener("input", debounce(() => { fillFeatureSelect(); }, 200));
  $("c-map-feature").addEventListener("change", () => loadFeature());
  for (const name of MAP_COLORMAPS) $("c-map-cmap").append(option(name));
  $("c-map-cmap").addEventListener("change", () => { const volume = state.volumes.feature; if (volume) { state.nv.setColormap(volume.id, $("c-map-cmap").value); } });
  for (const id of ["c-map-min", "c-map-max"]) $(id).addEventListener("change", () => {
    const volume = state.volumes.feature;
    if (!volume) return;
    const low = Number($("c-map-min").value), high = Number($("c-map-max").value);
    if (isFinite(low) && isFinite(high) && high > low) {
      applyWindow(volume, low, high);
      if ($("c-map-lock").checked) state.lockedRange = [low, high];
    }
  });
  $("c-map-opacity").addEventListener("input", () => { const volume = state.volumes.feature; if (volume) { volume.opacity = Number($("c-map-opacity").value); state.nv.updateGLVolume(); } });
  $("c-map-threshold").addEventListener("change", applyThreshold);
  $("c-map-lock").addEventListener("change", () => {
    state.lockedRange = $("c-map-lock").checked ? [Number($("c-map-min").value), Number($("c-map-max").value)] : null;
  });
  $("c-slice").addEventListener("change", () => {
    const nv = state.nv; if (!nv) return;
    const types = { multiplanar: nv.sliceTypeMultiplanar, axial: nv.sliceTypeAxial, coronal: nv.sliceTypeCoronal, sagittal: nv.sliceTypeSagittal, render: nv.sliceTypeRender };
    nv.setSliceType(types[$("c-slice").value]);
  });
  $("c-radiological").addEventListener("change", () => { if (state.nv) state.nv.setRadiologicalConvention($("c-radiological").checked); });
  $("c-centroid").addEventListener("click", () => {
    const nv = state.nv;
    if (!nv || !state.detail || !state.detail.centroid_mm) return;
    nv.scene.crosshairPos = nv.mm2frac(state.detail.centroid_mm);
    nv.drawScene();
    nv.createOnLocationChange && nv.createOnLocationChange();
  });
  $("c-reset").addEventListener("click", resetView);
  function resetView() {
    const nv = state.nv; if (!nv) return;
    nv.scene.crosshairPos = [0.5, 0.5, 0.5];
    if (nv.setPan2Dxyzmm) nv.setPan2Dxyzmm([0, 0, 0, 1]);
    nv.drawScene();
  }
  $("c-screenshot").addEventListener("click", () => {
    if (!state.nv || !state.detail) return;
    const feature = state.volumes.feature ? $("c-map-feature").selectedOptions[0].textContent : "image";
    const name = BLIND ? state.detail.region : regionName(state.detail);
    state.nv.saveScene(safeFileName("radiomicviz_" + name + "_" + feature) + ".png");
  });
  function updateLayerInfo() {
    const table = clear($("layer-info"));
    if (!state.detail) return;
    table.append(el("tr", null, ...["Layer", "Stage", "Shown", "Grid size", "Spacing (mm)", "Origin (mm)", "Voxels / labels", "Served as"].map((text) => el("th", { text }))));
    const shown = new Set(Object.values(state.volumes).filter(Boolean).map((volume) => volume.name));
    const layers = state.detail.layers.concat(state.featureLayers.filter((layer) => state.volumes.feature && layer.name === state.volumes.feature.name));
    for (const layer of layers) {
      const stats = state.statsCache[state.rid + ":" + layer.id];
      const geometry = stats && stats.stats ? stats.stats.geometry : null;
      const counts = stats && stats.stats ? (stats.stats.labels ? Object.entries(stats.stats.labels).map(([label, count]) => label + ": " + count).join(", ") : "finite " + stats.stats.n_finite + ", non-finite " + stats.stats.n_nonfinite) : "";
      table.append(el("tr", null,
        el("td", { text: (layer.feature || layer.kind) + (layer.modality ? " (" + layer.modality + ")" : "") + (layer.method ? " " + layer.method : "") }),
        el("td", { text: layer.stage + ", " + (state.project ? state.project.variant : "") }),
        el("td", { text: shown.has(layer.name) ? "yes" : "" }),
        el("td", { text: geometry ? geometry.size.join(" x ") : (layer.available ? "" : "not on disk") }),
        el("td", { text: geometry ? geometry.spacing.map(fmt).join(", ") : "" }),
        el("td", { text: geometry ? geometry.origin.map(fmt).join(", ") : "" }),
        el("td", { text: counts }),
        el("td", { text: stats ? stats.served_as : "" })));
    }
  }

  // ------------------------------------------------------------------ region tables
  function fillTablesControls() {
    const detail = state.detail;
    const modality = clear($("t-whole-modality"));
    for (const name of detail.modalities) modality.append(option(name));
    const method = clear($("t-hab-method"));
    for (const name of ["fixed_k2", "adaptive_k1to3", "geometric_1mm"]) method.append(option(name));
  }
  async function loadWholeTable() {
    const token = state.wholeToken = (state.wholeToken || 0) + 1;
    const box = clear($("t-whole"));
    $("t-whole-meta").textContent = "";
    let data;
    try {
      data = await api("/api/region/" + state.rid + "/features/whole?modality=" + encodeURIComponent($("t-whole-modality").value));
    } catch (error) {
      if (token === state.wholeToken) { state.wholeData = null; showError(box, error); }
      return;
    }
    if (token !== state.wholeToken) return;
    state.wholeData = data;
    renderWholeTable();
  }
  function renderWholeTable() {
    const data = state.wholeData; const box = clear($("t-whole"));
    if (!data) return;
    const query = $("t-whole-search").value.toLowerCase();
    $("t-whole-meta").textContent = "Stage whole, variant " + data.variant + ", source " + data.source + ", reference status " + (data.reference_status || "unknown") +
      (data.error ? ", row note: " + data.error : "") + ". " + data.percentile_note;
    const identity = el("details", null, el("summary", { text: "Identity and status" }));
    const identityTable = el("table", { class: "grid" });
    for (const item of data.identity) identityTable.append(el("tr", null, el("td", { text: item.name }), el("td", { text: item.text })));
    identity.append(identityTable); box.append(identity);
    const diagnostics = el("details", null, el("summary", { text: "Diagnostics (" + data.diagnostics.length + ")" }));
    const diagnosticsTable = el("table", { class: "grid" });
    for (const item of data.diagnostics) if (!query || item.name.toLowerCase().includes(query)) diagnosticsTable.append(el("tr", null, el("td", { text: item.name }), el("td", { text: item.text })));
    diagnostics.append(diagnosticsTable); box.append(diagnostics);
    for (const featureClass of Object.keys(data.features)) {
      const section = el("details", { open: featureClass === "shape" ? "" : null }, el("summary", { text: featureClass }));
      const table = el("table", { class: "grid" }, el("tr", null, ...["Image type", "Feature", "Value", "Cohort percentile (n)"].map((text) => el("th", { text }))));
      let count = 0;
      for (const [imageType, entries] of Object.entries(data.features[featureClass])) {
        for (const entry of entries) {
          if (query && !entry.name.toLowerCase().includes(query)) continue;
          count++;
          const value = entry.kind === "number" ? fmt(entry.value) : entry.kind === "nonfinite" ? entry.text + " (non-finite)" : entry.text;
          const rank = entry.cohort_percentile === null || entry.cohort_percentile === undefined ? "" : fmt(entry.cohort_percentile) + " (n=" + entry.cohort_n + ")";
          table.append(el("tr", null, el("td", { text: imageType }), el("td", { text: entry.name }), el("td", { text: value }), el("td", { text: rank })));
        }
      }
      if (count) { section.firstChild.textContent = featureClass + " (" + count + ")"; section.append(table); box.append(section); }
    }
  }
  $("t-whole-modality").addEventListener("change", loadWholeTable);
  $("t-whole-search").addEventListener("input", debounce(renderWholeTable, 200));
  async function loadHabitatTable() {
    const token = state.habitatToken = (state.habitatToken || 0) + 1;
    const box = clear($("t-hab"));
    let data;
    try {
      data = await api("/api/region/" + state.rid + "/features/habitat?method=" + encodeURIComponent($("t-hab-method").value) + "&modality=" + encodeURIComponent($("t-hab-modality").value));
    } catch (error) {
      if (token === state.habitatToken) { state.habitatData = null; $("t-hab-meta").textContent = ""; showError(box, error); }
      return;
    }
    if (token !== state.habitatToken) return;
    state.habitatData = data;
    $("t-hab-meta").textContent = "Stage habitat-features, variant " + data.variant + ", method " + data.method + ", modality " + data.modality + ". " + data.legacy_note;
    for (const id of ["t-hab-a", "t-hab-b"]) {
      const select = clear($(id));
      data.zones.forEach((zone, index) => select.append(option(index, zone.zone_name + " (" + zone.zone_status + ")")));
    }
    if (data.zones.length > 1) $("t-hab-b").value = "1";
    renderHabitatTable();
  }
  function renderHabitatTable() {
    const data = state.habitatData; const box = clear($("t-hab"));
    if (!data) return;
    const zones = el("table", { class: "grid" }, el("tr", null, ...["Zone", "Status", "Voxels (source grid)", "Voxels (habitat grid)", "Small", "Label", "Error"].map((text) => el("th", { text }))));
    for (const zone of data.zones) {
      zones.append(el("tr", null, el("td", { text: zone.zone_name }), el("td", null, statusChip(zone.zone_status === "skipped" ? "skipped" : zone.zone_status === "failed" ? "failed" : "complete", zone.error)),
        el("td", { text: zone.source_zone_voxels }), el("td", { text: zone.habitat_grid_voxels }), el("td", { text: zone.zone_flagged_small }),
        el("td", { text: zone.mask_label_value }), el("td", { text: zone.error })));
    }
    box.append(zones);
    const a = data.zones[Number($("t-hab-a").value)], b = data.zones[Number($("t-hab-b").value)];
    if (!a || !b) return;
    const table = el("table", { class: "grid" }, el("tr", null, ...["Feature", a.zone_name, b.zone_name, "Difference (second minus first)"].map((text) => el("th", { text }))));
    for (const featureClass of Object.keys(a.features)) {
      for (const [imageType, entries] of Object.entries(a.features[featureClass])) {
        const other = ((b.features[featureClass] || {})[imageType]) || [];
        for (const entry of entries) {
          const match = other.find((item) => item.name === entry.name);
          const difference = entry.kind === "number" && match && match.kind === "number" ? fmt(match.value - entry.value) : "";
          table.append(el("tr", null, el("td", { text: entry.name }), el("td", { text: entry.kind === "number" ? fmt(entry.value) : entry.text }),
            el("td", { text: match ? (match.kind === "number" ? fmt(match.value) : match.text) : "" }), el("td", { text: difference })));
        }
      }
    }
    box.append(el("h4", { text: "Zone comparison (differences are viewer-derived)" }), table);
  }
  for (const id of ["t-hab-method", "t-hab-modality"]) $(id).addEventListener("change", loadHabitatTable);
  for (const id of ["t-hab-a", "t-hab-b"]) $(id).addEventListener("change", renderHabitatTable);
  function renderZones() {
    const table = clear($("t-zones"));
    $("zones-note").textContent = state.detail.label_order_note;
    table.append(el("tr", null, ...["Method", "Zone", "Label value", "Voxels", "Chosen k", "Silhouette", "Valid"].map((text) => el("th", { text }))));
    for (const row of state.detail.habitat_index) {
      table.append(el("tr", null, ...[row.method, row.zone_name, row.mask_label_value, row.n_voxels, row.chosen_k, row.silhouette_score, row.habitat_valid].map((text) => el("td", { text }))));
    }
  }
  function renderUnits() {
    const table = clear($("t-units"));
    table.append(el("tr", null, ...["Stage", "Modality", "Method", "Status", "Reason", "Reference status", "Voxels", "Finished (UTC)"].map((text) => el("th", { text }))));
    for (const unit of state.detail.units) {
      table.append(el("tr", null, el("td", { text: STAGE_LABEL[unit.stage] }), el("td", { text: unit.modality || "" }), el("td", { text: unit.method || "" }),
        el("td", null, statusChip(unit.status, unit.reason)), el("td", { text: unit.reason }),
        el("td", { class: unit.reference_status && unit.reference_status !== "reference-compatible" ? "badge-nonref" : "", text: unit.reference_status }),
        el("td", { text: unit.mask_voxels === null || unit.mask_voxels === undefined ? "" : unit.mask_voxels }), el("td", { text: unit.finished_utc || "" })));
    }
    const checks = clear($("t-checks"));
    checks.append(el("tr", null, ...["Source", "Check", "Status", "Measured", "Threshold", "Message"].map((text) => el("th", { text }))));
    for (const row of state.detail.input_checks.filter((item) => item.status !== "PASS")) {
      checks.append(el("tr", null, el("td", { text: "manifest" }), el("td", { text: row.check_id }), el("td", { text: row.status }), el("td", { text: row.measured_value }), el("td", { text: row.threshold }), el("td", { text: row.message })));
    }
    for (const row of state.detail.stage_checks) {
      checks.append(el("tr", null, el("td", { text: STAGE_LABEL[row.stage] }), el("td", { text: row.check_id }), el("td", { text: row.status }), el("td", { text: row.measured_value }), el("td", { text: row.threshold }), el("td", { text: row.message })));
    }
    const material = clear($("t-material"));
    const info = state.detail.materialization;
    if (info) {
      const region = info.region || {};
      const rows = [["Stage", info.stage], ["Mode", info.mode], ["Connectivity", info.connectivity], ["Minimum region voxels", info.minimum_region_voxels],
        ["Source mask", info.source_mask], ["Source mask SHA-256", info.source_mask_sha256], ["Component or label", region.component_id !== null && region.component_id !== undefined ? "component " + region.component_id : "label " + region.source_label],
        ["Voxels", region.n_voxels], ["Mask SHA-256", region.mask_sha256], ["Created (UTC)", info.created_utc],
        ["Excluded in this session", (info.excluded_regions || []).map((item) => item.region_id + " (" + item.n_voxels + " voxels)").join(", ") || "none"]];
      for (const [label, value] of rows) material.append(el("tr", null, el("td", { text: label }), el("td", { text: value === null || value === undefined ? "" : String(value) })));
    } else {
      material.append(el("tr", null, el("td", { text: "Masks were supplied as separate files; no materialization record." })));
    }
    clear($("verify-result"));
  }
  $("verify").addEventListener("click", async () => {
    const box = clear($("verify-result"));
    box.append(el("p", { class: "muted", text: "Verifying..." }));
    try {
      const result = await api("/api/region/" + state.rid + "/verify", { method: "POST" });
      clear(box).append(el("p", { class: result.failed ? "fail-text" : "", text: result.checked + " files checked, " + result.failed + " did not match. " + result.method }));
      const table = el("table", { class: "grid" }, el("tr", null, ...["Stage", "Modality", "Method", "File", "Matches"].map((text) => el("th", { text }))));
      for (const item of result.results) table.append(el("tr", null, ...[STAGE_LABEL[item.stage], item.modality || "", item.method || "", item.file, item.verified ? "yes" : "NO"].map((text) => el("td", { text }))));
      box.append(table);
    } catch (error) { showError(box, error); }
  });

  // ------------------------------------------------------------------ cohort
  function drawHistogram(canvas, data, label) {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#59636e"; context.font = "12px sans-serif";
    const counts = (data.histogram && data.histogram.counts) || [];
    const edges = (data.histogram && data.histogram.edges) || [];
    if (!counts.length) { context.fillText("No finite values.", 10, 20); return; }
    const left = 40, bottom = canvas.height - 30, width = canvas.width - 60, height = canvas.height - 50;
    const peak = Math.max(...counts);
    const step = width / counts.length;
    context.fillStyle = "#0b5cad";
    counts.forEach((count, index) => { const bar = (count / peak) * height; context.fillRect(left + index * step + 1, bottom - bar, Math.max(1, step - 2), bar); });
    context.fillStyle = "#1f2328";
    context.fillText(fmt(edges[0]), left, bottom + 14);
    context.fillText(fmt(edges[edges.length - 1]), left + width - 40, bottom + 14);
    context.fillText(label + " (n=" + data.n + ")", left, 14);
    if (data.selected_value !== null && data.selected_value !== undefined && isFinite(data.selected_value) && edges.length > 1) {
      const span = edges[edges.length - 1] - edges[0] || 1;
      const x = left + ((data.selected_value - edges[0]) / span) * width;
      context.strokeStyle = "#cf222e"; context.lineWidth = 2;
      context.beginPath(); context.moveTo(x, bottom); context.lineTo(x, 20); context.stroke();
    }
  }
  async function loadCohortControls() {
    const modalities = state.project ? state.project.counts.modalities : [];
    const select = clear($("k-modality"));
    for (const name of modalities) select.append(option(name));
    const method = clear($("k-method"));
    for (const name of ["fixed_k2", "adaptive_k1to3", "geometric_1mm"]) method.append(option(name));
    const zone = clear($("k-zone"));
    for (const name of ["peripheral_habitat", "central_habitat", "intermediate_habitat_1", "whole_lesion", "geometric_shell", "geometric_core"]) zone.append(option(name));
  }
  $("k-go").addEventListener("click", async () => {
    const params = new URLSearchParams({ stage: $("k-stage").value, modality: $("k-modality").value, feature: $("k-feature").value.trim() });
    if ($("k-stage").value === "habitat_features") { params.set("method", $("k-method").value); params.set("zone", $("k-zone").value); }
    if (state.rid) params.set("rid", state.rid);
    try {
      const data = await api("/api/cohort/distribution?" + params.toString());
      drawHistogram($("k-hist"), data, data.feature);
      const summary = data.summary ? "median " + fmt(data.summary.median) + ", IQR " + fmt(data.summary.q25) + " to " + fmt(data.summary.q75) + ", range " + fmt(data.summary.min) + " to " + fmt(data.summary.max) : "no finite values";
      $("k-meta").textContent = "Stage " + STAGE_LABEL[data.stage] + ", variant " + data.variant + ", modality " + data.modality + ": n=" + data.n + ", non-finite " + data.n_nonfinite +
        ", " + summary + (data.selected_percentile === null ? "" : ". Selected region percentile rank " + fmt(data.selected_percentile)) + ". Viewer-derived.";
    } catch (error) { $("k-meta").textContent = "Could not load: " + error.message; }
  });
  $("k-feature").addEventListener("focus", async () => {
    if ($("k-features").children.length || !state.rid) return;
    try {
      const data = await api("/api/region/" + state.rid + "/features/whole");
      for (const groups of Object.values(data.features)) for (const entries of Object.values(groups)) for (const entry of entries) $("k-features").append(option(entry.name));
    } catch (error) { /* the list stays empty; typing a column name still works */ }
  });
  async function loadCohortRegions() {
    try {
      const data = await api("/api/cohort/regions");
      drawHistogram($("k-sizes"), data.region_voxels, "region voxels");
      $("k-sizes-meta").textContent = "Region voxel counts from the manifest checks (n=" + data.region_voxels.n + "). Viewer-derived histogram.";
      const table = clear($("k-zones"));
      table.append(el("tr", null, ...["Method:zone", "n", "Median fraction", "Range"].map((text) => el("th", { text }))));
      for (const [key, value] of Object.entries(data.habitat_zone_fractions)) {
        table.append(el("tr", null, el("td", { text: key }), el("td", { text: value.n }), el("td", { text: value.summary ? fmt(value.summary.median) : "" }),
          el("td", { text: value.summary ? fmt(value.summary.min) + " to " + fmt(value.summary.max) : "" })));
      }
    } catch (error) { $("k-sizes-meta").textContent = "Could not load: " + error.message; }
  }

  // ------------------------------------------------------------------ QC and provenance
  async function loadQc() {
    const params = new URLSearchParams();
    if ($("q-stage").value) params.set("stage", $("q-stage").value);
    if ($("q-status").value) params.set("status", $("q-status").value);
    try {
      const data = await api("/api/qc?" + params.toString());
      $("q-count").textContent = data.total + " rows";
      const table = clear($("q-table"));
      table.append(el("tr", null, ...["Stage", "Kind", "Status", "Code", "Region", "Modality", "Method", "Measured", "Threshold", "Message"].map((text) => el("th", { text }))));
      for (const row of data.rows) {
        const link = row.rid ? el("a", { href: "#", text: "open", onclick: (event) => { event.preventDefault(); openRegion(row.rid); } }) : el("span", { text: "" });
        table.append(el("tr", null, el("td", { text: STAGE_LABEL[row.stage] || row.stage }), el("td", { text: row.kind }), el("td", { text: row.status }), el("td", { text: row.code }),
          el("td", null, link), el("td", { text: row.modality }), el("td", { text: row.method }), el("td", { text: row.measured_value }), el("td", { text: row.threshold }), el("td", { text: row.message })));
      }
    } catch (error) { showError($("q-table"), error); }
    renderSoftware();
    loadDownloads();
  }
  for (const id of ["q-stage", "q-status"]) $(id).addEventListener("change", loadQc);
  function renderSoftware() {
    const project = state.project; if (!project) return;
    const table = clear($("q-software"));
    table.append(el("tr", null, ...["Stage", "Ran", "Status", "Reference compatible", "Deviations", "Preset SHA-256", "Software", "Finished (UTC)"].map((text) => el("th", { text }))));
    for (const stage of STAGES) {
      const info = project.stages[stage];
      const software = info.software ? Object.entries(info.software).map(([name, version]) => name + " " + version).join(", ") : "";
      table.append(el("tr", null, el("td", { text: STAGE_LABEL[stage] }), el("td", { text: info.ran ? "yes" : "no" }), el("td", { text: info.status || "" }),
        el("td", { class: info.reference_compatible === false ? "badge-nonref" : "", text: info.reference_compatible === null || info.reference_compatible === undefined ? "" : String(info.reference_compatible) }),
        el("td", { text: (info.reference_deviations || []).join("; ") }), el("td", { text: info.preset_sha256 || "" }), el("td", { text: software }), el("td", { text: info.finished_utc || "" })));
    }
  }
  function summaryButtons() {
    const box = clear($("q-summary-buttons"));
    for (const stage of STAGES) {
      box.append(el("button", { type: "button", text: STAGE_LABEL[stage], onclick: async () => {
        $("q-summary").textContent = "Computing...";
        try {
          const data = await api("/api/summary/" + stage);
          $("q-summary").textContent = data.available ? data.note + "\n\n" + data.markdown : "No summary: " + data.message;
        } catch (error) { $("q-summary").textContent = "Could not load: " + error.message; }
      } }));
    }
    const stages = $("q-stage");
    if (stages.options.length === 1) for (const stage of ["manifest"].concat(STAGES)) stages.append(option(stage, STAGE_LABEL[stage] || stage));
  }
  async function loadDownloads() {
    const list = clear($("q-downloads"));
    try {
      const data = await api("/api/downloads");
      if (data.blind) { list.append(el("li", { text: data.note })); return; }
      for (const file of data.files) list.append(el("li", null, el("a", { href: "/api/download/" + file.id, text: file.relative_path }), " (" + file.size_bytes + " bytes)"));
    } catch (error) { showError(list, error); }
  }

  // ------------------------------------------------------------------ keyboard
  document.addEventListener("keydown", (event) => {
    if (event.target && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
    const rows = state.regions; const index = rows.findIndex((row) => row.rid === state.rid);
    if ((event.key === "j" || event.key === "k") && rows.length) {
      const next = event.key === "j" ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
      if (rows[next]) openRegion(rows[next].rid);
    } else if ((event.key === "n" || event.key === "p") && state.featureLayers.length) {
      const select = $("c-map-feature"); const current = state.featureLayers.findIndex((item) => item.id === select.value);
      const next = event.key === "n" ? Math.min(state.featureLayers.length - 1, current + 1) : Math.max(0, current - 1);
      select.value = state.featureLayers[next].id; loadFeature();
    } else if (event.key === "o") {
      if (state.volumes.feature) { removeVolume("feature").then(() => state.nv.drawScene()); } else { loadFeature(); }
    } else if (event.key === "r") { resetView(); }
  });

  // ------------------------------------------------------------------ start
  async function start() {
    try {
      await loadProject();
      await loadStatus();
      await loadRegions(true);
      await loadCohortControls();
      summaryButtons();
      if (state.regions.length) await openRegion(state.regions[0].rid, true);
    } catch (error) { showError($("banners"), error); }
    document.body.dataset.ready = "true";
  }
  window.addEventListener("error", (event) => { document.body.dataset.lastError = String(event.message || "error"); });
  start();
})();
