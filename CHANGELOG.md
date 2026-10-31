# Changelog

All notable changes to RadiomicsViz are recorded here. Versions follow PEP 440.

## 0.3.0.dev0 (unreleased)

This version starts a ground-up rewrite. It is not compatible with the 0.1 series.

### Added

- `radiomicviz view`, a read-only browser dashboard (optional `viewer` extra: Flask) with
  a vendored NiiVue 0.69.0 bundle. It shows images, masks, voxelwise maps, habitat masks,
  feature tables, cohort context, QC and provenance for any partial or complete output
  tree, binds to localhost with a secret launch token, and never writes to the data or output
  folders.
- `radiomicviz habitat-features`, reproducing the reference final analysis step with the
  unchanged whole-ROI preset on every non-empty fixed, adaptive and geometric habitat
  label for T1 and FLAIR. The stage uses authenticated full-source-grid masks, retains
  expected small-zone skips, resumes by region/method/modality, and writes a normalized
  cohort table plus six reference-schema method-by-modality tables.
- `radiomicviz habitat-features --summarize` for report-only cohort, status, voxel-count,
  feature-completeness and subject/session summaries.
- Expected final-stage handling for habitat zones below the ROI minimum or spanning fewer
  than two dimensions. These zones are still attempted; a PyRadiomics rejection retains
  the row and warning without failing the complete run.
- Independent NIfTI integrity and finite-value validation with SHA-256 input/output
  fingerprints. Truncated images, NaN/Inf intensities and vector inputs now stop at the
  manifest with concise diagnostics instead of reaching extraction.
- Resumable QC skips for one-dimensional regions and constant/non-finite ROI intensity
  ranges, plus full-source-grid companion masks for every habitat labeling.
- Small process-pool tasks and dead-worker recovery for whole and voxelwise extraction;
  completed units are preserved and invocation reports are still written after a worker
  exits unexpectedly.

- `radiomicviz whole --summarize` and `radiomicviz voxelwise --summarize` for derived,
  report-only exploratory analysis. Whole-region reports include per-modality descriptive
  statistics and missingness for every scalar feature; voxelwise reports include finite-map
  coverage from authenticated completion records. Both produce Markdown, JSON and CSV
  outputs without rerunning extraction or modifying radiomics results.
- `radiomicviz habitats --summarize` for report-only exploratory analysis of completed
  habitat runs. It writes Markdown, JSON and CSV summaries of cohort coverage, method
  status, adaptive k choices, habitat/cluster-size distributions, subject/session
  coverage and selected-feature frequencies without rerunning clustering or modifying
  habitat masks.
- Reference habitat clustering from paired T1 and FLAIR voxelwise maps. The package
  preserves the validated lesion-local median/IQR representation, joint Spearman
  redundancy filtering, modality block weighting, fixed k=2 primary analysis, adaptive
  k=1..3 sensitivity analysis, physical-distance label canonicalization and 1 mm
  geometric comparator. Outputs are atomic and resumable per region, with three uint8
  masks, seven method-zone index rows, detailed representation QC and authenticated
  provenance. Regions with unavailable upstream maps at or below 10 voxels are explicit
  skips; 11–49 voxel regions retain the reference fallback behavior.

- A package-wide 10-voxel minimum lesion size. Combined-mask materialization excludes
  smaller labels/components while preserving their details in provenance, and
  whole-region extraction independently records smaller externally supplied masks as
  resumable skips rather than PyRadiomics failures.
- Reference 32-bin voxelwise radiomics using the authenticated validation parameter file,
  manifest-driven region/modality units, 92 float64 NRRD feature maps, exact map-schema
  validation, concurrent local execution, per-unit resume and replacement, QC, errors and
  provenance. Because PyRadiomics requires an ROI strictly larger than its configured
  `minimumROISize`, regions with 10 or fewer voxels are retained as explicit, resumable
  voxelwise skips rather than failing the invocation or silently disappearing.
- Targeted handling of the known PyRadiomics voxelwise
  `original_glcm_JointAverage` all-nonfinite result: the map is retained with a concise QC
  warning instead of discarding the other 91 valid maps. New completion records include
  per-map finite-voxel counts. Completion records from before output fingerprints were
  introduced must be recomputed once with `--force`.
- Exact-message logging filters for the expected PyRadiomics fixed-bin-count advisory and
  symmetric-GLCM redundancy advisory. All other PyRadiomics and Python warnings remain
  visible.
- Reference 32-bin whole-region radiomics with an immutable PyRadiomics preset,
  minimal built-in protocol, concurrent local execution, per-unit resume and
  replacement, cohort exports, QC, errors and provenance.
- A pinned analysis environment, `environment.yml`.
- Distribution `radiomicsviz` with the import package and command `radiomicviz`.
- `radiomicviz --version` and `radiomicviz --help`.
- `radiomicviz project init`, which writes a commented `project.yaml` template. An
  existing file is never overwritten in place: the command refuses unless `--force` is
  given, and then renames the existing file to `<name>.bak-<timestamp>` first.
- Strict loader `radiomicviz.config.load_project_config`. It rejects unknown keys,
  missing keys, wrong
  types, duplicate keys and inconsistent settings, and reports every problem with its
  dotted key path.
- Deterministic discovery of subjects, sessions, modalities and separate binary region
  masks from `project.yaml`.
- Optional lesion materialization from either an integer-labeled map or a combined
  binary mask with explicitly configured 6- or 26-connectivity. Source masks remain
  read-only; verified binary outputs carry per-session completion and provenance records.
- Connected-component materialization numbers lesions by descending voxel count, so the
  largest component is `lesion_001`, with stable source order as the tie-breaker.
- Manifest validation of materialized-mask hashes and source hashes, preventing stale
  lesion splits from entering radiomics extraction.
- A canonical manifest whose `image_path` points directly to an analysis-ready image.
- Minimal geometry, spacing, mask and overlap checks with a preview-first
  `radiomicviz manifest` command.
- Concise manifest QC, errors, run metadata and logging artifacts. Each `--write` renames
  the previous artifacts with a `.bak-<timestamp>` suffix; there is no `--force` option.
- NumPy and SimpleITK runtime dependencies required by input validation.
- A direct-input contract: users supply final, co-registered images and masks. Image
  preparation is intentionally outside the package; only transformations configured in
  the PyRadiomics parameter preset occur during feature extraction.
- A concise README quick start from `project init` through habitat-zone radiomics and
  summaries, plus a separate methods and output-schema reference.
- Progress messages on stderr for every compute stage: one
  start line, then one line per subject/session. Stdout summaries are unchanged; the
  library entry points accept an optional `progress` callback.
- A manual, step-by-step conda installation procedure in the README and, in
  `INSTALLATION_DETAILS.md`, requirements, verification, troubleshooting and accurate
  pip-only/development guidance for the capabilities currently shipped by the package.
- Conda-managed SimpleITK 2.5.5, avoiding pip replacement failures when a cluster's conda
  configuration supplies the legacy distutils-installed SimpleITK 2.2.1 package.
- The `editables` build helper required by Hatchling when RadiomicsViz is installed in
  editable mode without build isolation.
