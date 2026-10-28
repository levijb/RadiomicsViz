# Changelog

All notable changes to RadiomicsViz are recorded here. Versions follow PEP 440.

## 0.3.0.dev0 (unreleased)

This version starts a ground-up rewrite. It is not compatible with the 0.1 series.

### Added

- Reference 32-bin whole-region radiomics with an immutable PyRadiomics preset,
  minimal built-in protocol, concurrent local execution, per-unit resume and
  replacement, cohort exports, QC, errors and provenance.
- A pinned analysis environment and one-command environment creation script.
- Distribution `radiomicsviz` with the import package and command `radiomicviz`.
- `radiomicviz --version` and `radiomicviz --help`.
- `radiomicviz project init`, which writes a commented `project.yaml` template. An
  existing file is never overwritten in place: the command refuses unless `--force` is
  given, and then renames the existing file to `<name>.bak-<timestamp>` first.
- Strict loader `radiomicviz.config.load_project_config`. It rejects unknown keys,
  missing keys, wrong
  types, duplicate keys and inconsistent settings, and report every problem with its
  dotted key path.
- Deterministic discovery of subjects, sessions, modalities and separate binary region
  masks from `project.yaml`.
- Optional lesion materialization from either an integer-labeled map or a combined
  binary mask with explicitly configured 6- or 26-connectivity. Source masks remain
  read-only; verified binary outputs carry per-session completion and provenance records.
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
- A README quick start from `project init` through manifest creation and whole-region
  extraction.
- A manual, step-by-step conda installation procedure, a simplified optional setup
  wrapper and accurate pip-only/development guidance for the capabilities currently
  shipped by the package.
- Conda-managed SimpleITK 2.5.5, avoiding pip replacement failures when a cluster's conda
  configuration supplies the legacy distutils-installed SimpleITK 2.2.1 package.
- The `editables` build helper required by Hatchling when RadiomicsViz is installed in
  editable mode without build isolation.
