# RadiomicsViz

RadiomicsViz is a manifest-driven MRI radiomics package for studies with either separate
binary lesion masks or one combined lesion mask per subject/session. The current release
provides deterministic input discovery, safe lesion-mask materialization and reference
whole-region PyRadiomics extraction.

## Input contract

RadiomicsViz does not prepare or modify images before extraction. Every configured image
must already be:

- the final image intended for analysis;
- co-registered with its region masks;
- three-dimensional and readable by SimpleITK; and
- in the intended analysis space and voxel spacing.

The package checks image/mask geometry and spacing but does not correct them. PyRadiomics
can still apply operations explicitly configured in its parameter file. The frozen
reference preset currently requests 1 x 1 x 1 mm resampling and filtered image variants;
those are extraction settings, not a separate image-preparation stage.

## Lesion-mask input choices

The lesion input workflow is selected by `regions.mode` in `project.yaml`:

- `per_region_masks` reads masks that are already separated, with one binary file per
  lesion. No splitting command is needed.
- `labeled_map` reads one combined integer map. Every positive integer label is preserved
  as one lesion.
- `connected_components` reads one combined binary mask and creates one lesion for each
  connected foreground component. The YAML must explicitly choose 6- or 26-connectivity.

Connected-components splitting cannot separate lesions that touch; touching foreground
voxels remain one lesion. RadiomicsViz does not apply watershed or invent boundaries.
The combined source files are never modified. Materialized binary masks are written under
`output_root/materialized_regions/` with source and output hashes, voxel counts and a
completion record.

## Installation

The supported run platform is Linux x86-64 with conda available. From the repository root:

```bash
bash scripts/create_environment.sh
conda activate radiomicviz-validation-v1
```

The script creates the pinned environment, installs PyRadiomics 3.0.1 and installs
RadiomicsViz in editable mode. It refuses to replace an existing environment of the same
name.

## Quick start

Create a project template:

```bash
radiomicviz project init
```

Edit `project.yaml` so its image patterns point directly to the final analysis images and
choose one of the lesion modes above.

For already separated masks, proceed directly to the manifest preview. For a combined
mask, first preview and then run materialization:

```bash
radiomicviz lesions materialize --project project.yaml --dry-run
radiomicviz lesions materialize --project project.yaml
```

The second command safely resumes an identical completed materialization. If its source or
settings changed, it stops instead of replacing output; use `--force` only after reviewing
the change. Then preview discovery and QC:

```bash
radiomicviz manifest --project project.yaml
```

If the preview is correct, write the canonical manifest:

```bash
radiomicviz manifest --project project.yaml --write
```

Preview whole-region extraction:

```bash
radiomicviz whole \
  --manifest out/manifest/regions_manifest.csv \
  --variant b32 \
  --dry-run
```

Run extraction, initially with a conservative worker count:

```bash
radiomicviz whole \
  --manifest out/manifest/regions_manifest.csv \
  --variant b32 \
  --jobs 2
```

## Project configuration

`project.yaml` describes only input discovery and the output root:

- subject and optional session discovery;
- one analysis-ready image pattern per modality;
- one mask pattern that finds either separate binary regions or exactly one combined mask;
- the lesion input mode and its mode-specific foreground, identifier and connectivity
  settings;
- the expected shared space; and
- whether non-reference voxel spacing is allowed with an explicit warning.

Patterns are relative to `data_root`. They can include `{subject}` and, for discovered
sessions, `{session}`. The template defaults to T1 and FLAIR but modality names and image
patterns can be edited.

For existing separated masks, keep the template form:

```yaml
regions:
  mode: per_region_masks
  region_type: lesion
  mask_pattern: '{subject}/lesions/*.nii.gz'
  id_regex: '^(?P<region>lesion_[0-9]+)\.nii\.gz$'
  label: 1
  connectivity: null
```

For one integer-labeled map, use:

```yaml
regions:
  mode: labeled_map
  region_type: lesion
  mask_pattern: '{subject}/lesions/labeled_lesions.nii.gz'
  id_regex: null
  label: null
  connectivity: null
```

For one binary mask containing all lesions, use connected components and make the
connectivity choice explicit:

```yaml
regions:
  mode: connected_components
  region_type: lesion
  mask_pattern: '{subject}/lesions/all_lesions.nii.gz'
  id_regex: null
  label: 1
  connectivity: 26
```

With discovered sessions, include `{session}` in every image and mask pattern.

The manifest stage verifies:

- exactly one image per subject, session and modality;
- three-dimensional images and masks;
- mask values of background 0 and the configured foreground label;
- non-empty regions;
- matching image/mask geometry;
- reference spacing of 1.0 +/- 0.001 mm per axis, unless explicitly overridden; and
- safe, unique subject, session and region identifiers.

`manifest --write` creates:

```text
<output_root>/manifest/regions_manifest.csv
<output_root>/manifest/input_checks.csv
<output_root>/manifest/errors.csv
<output_root>/manifest/run_metadata.json
<output_root>/qc/logs/manifest.log
```

Every manifest row contains one region and modality, including the direct `image_path`,
mask path, label and file fingerprints. Rewriting the manifest archives the previous
artifacts with `.bak-<timestamp>` suffixes.

## Whole-region radiomics

`radiomicviz whole` treats each region and modality as one independent, resumable unit.
The built-in `lesion-habitat-v1` protocol currently exposes the `b32` variant using the
frozen `roi-allfeatures-bin32.yaml` preset.

The preset SHA-256 is:

```text
cd2f3e1da1028daaec55b5573b7e2d986b7f34302617452103724d94ff0f942b
```

With the reference software stack, every successful unit must contain:

- 1,834 radiomic features;
- 14 shape features;
- 37 PyRadiomics diagnostic entries; and
- ordered feature-schema SHA-256
  `f4ea6da5d5191bfb24ad81f05cf7792d7c853294e460fcaf959621c00f02240e`.

The 14 shape features are expected for this PyRadiomics 3D shape implementation. Other
documentation may count additional 2D-only shape features, which are not part of this 3D
extraction.

The historical preset contains an inaccurate comment suggesting that
`resampledPixelSpacing: [1, 1, 1]` disables resampling. The parsed setting actually requests
1-mm isotropic resampling with SimpleITK B-spline interpolation. The preset bytes are
preserved for validation parity; the correction is documented here rather than editing the
reference file.

## Whole-region outputs

Primary tidy table:

```text
<output_root>/tables/whole/b32/whole_region.csv
```

Historical wide-format compatibility table:

```text
<output_root>/tables/whole/b32/whole_region_reference_format.csv
```

The compatibility table contains 1,880 columns: 9 identity columns, 37 diagnostics and
1,834 features. QC and provenance are written to:

```text
<output_root>/qc/whole_b32_qc.csv
<output_root>/qc/whole_b32_errors.csv
<output_root>/qc/whole_b32_run_metadata.json
<output_root>/qc/logs/whole_b32.log
```

Per-unit features and completion records are stored beneath:

```text
<output_root>/subjects/<subject>/sessions/<session>/regions/<region>/radiomics/whole/b32/<modality>/
```

A rerun resumes units whose inputs, preset, protocol, software and outputs still match.
Use `--force` to archive and recompute existing units. Failed units and non-finite feature
values remain visible in the cohort and QC outputs rather than being silently discarded.

Reference versions are enforced by default. `--allow-version-mismatch` permits an
intentional diagnostic run and stamps it non-reference with the exact deviations.

## Scope

The current package implements `project init`, optional combined-mask lesion
materialization, `manifest` and `whole`. Voxelwise radiomics, habitat clustering,
habitat-level radiomics and scheduler adapters are later stages.
