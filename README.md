# RadiomicsViz

RadiomicsViz is a manifest-driven MRI radiomics package for studies with either separate
binary lesion masks or one combined lesion mask per subject/session. The current release
provides deterministic input discovery, safe lesion-mask materialization and reference
whole-region and voxelwise PyRadiomics extraction.

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
  as one lesion when it contains at least 10 voxels.
- `connected_components` reads one combined binary mask and creates one lesion for each
  connected foreground component. Materialized lesions are numbered by descending voxel
  count, so the largest is `lesion_001`; equal-sized components retain their deterministic
  source-component order. The YAML must explicitly choose 6- or 26-connectivity.

The package-wide minimum lesion size is 10 voxels. Combined-mask labels or connected
components below this threshold are not materialized; their source labels/component IDs
and voxel counts remain recorded in the materialization completion and provenance files.
Exactly 10 voxels is eligible.

Connected-components splitting cannot separate lesions that touch; touching foreground
voxels remain one lesion. RadiomicsViz does not apply watershed or invent boundaries.
The combined source files are never modified. Materialized binary masks are written under
`output_root/materialized_regions/` with source and output hashes, voxel counts and a
completion record.

## Installation

### From source with conda (recommended)

The reference workflow is validated for Linux x86-64. Conda is recommended because it
installs the exact Python and compiled-library versions used by the reference protocol.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/levijb/RadiomicsViz.git
   cd RadiomicsViz
   ```

2. Create and activate the pinned environment:

   ```bash
   conda env create --file environment.yml
   conda activate radiomicviz
   ```

3. Install PyRadiomics and RadiomicsViz:

   ```bash
   python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
   python -m pip install --editable . --no-build-isolation --no-deps
   python -m pip check
   radiomicviz --version
   ```

PyRadiomics is installed after environment creation so its build can use the pinned NumPy,
setuptools and wheel already present. `--no-deps` prevents pip from replacing the versions
fixed by `environment.yml`.

For recovery steps, troubleshooting and further details, see
[INSTALLATION_DETAILS.md](INSTALLATION_DETAILS.md).

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

After the whole-region run has been reviewed, preview and run voxelwise extraction from
the same manifest:

```bash
radiomicviz voxelwise \
  --manifest out/manifest/regions_manifest.csv \
  --variant b32 \
  --dry-run

radiomicviz voxelwise \
  --manifest out/manifest/regions_manifest.csv \
  --variant b32 \
  --jobs 2
```

Voxelwise extraction is substantially more memory- and time-intensive than whole-region
extraction. Start with one or two workers and a small representative cohort before raising
`--jobs` on a cluster.

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
- a warning for externally supplied regions below the 10-voxel extraction minimum;
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

Before constructing a PyRadiomics extractor, the command independently counts the
configured label in every mask. Regions with fewer than 10 voxels receive a resumable
`skipped` completion record and QC warning rather than an extraction error. This check
also applies when users provide already separated binary masks, so those inputs cannot
bypass the minimum-size rule.

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

## Voxelwise radiomics

`radiomicviz voxelwise` uses the canonical manifest directly; no second CSV or lesion-tree
layout is needed. Its frozen 32-bin parameter file is the reference validation preset,
with SHA-256:

```text
ae781c704fd5688bd5ecf9f0a8083464e3981d21dbd2f1bd2a1e31ec48eb5cde
```

The extractor calls PyRadiomics with `voxelBased=True` and the manifest label at execution
time. The preset uses the original image only, does not resample, uses a one-voxel kernel,
keeps the kernel inside the lesion, initializes non-feature voxels to NaN, and requests 32
gray-level bins. GLCM MCC is excluded because it is numerically unstable in voxelwise mode.

Every successful region/modality unit contains 92 float64 NRRD maps:

- 18 first-order;
- 23 GLCM;
- 16 GLRLM;
- 16 GLSZM;
- 14 GLDM; and
- 5 NGTDM.

Maps retain the input spacing and direction and cover the region bounding box plus the
one-voxel kernel padding. They are written beneath:

```text
<output_root>/subjects/<subject>/sessions/<session>/regions/<region>/radiomics/voxelwise/b32/<modality>/
```

The preset requires at least 10 lesion voxels. Smaller regions receive a hash-checked
`skipped` completion record and a QC warning, so they remain visible without causing the
entire command to fail. Extraction exceptions, wrong map schemas, unreadable maps, wrong
geometry, non-float64 maps and maps without any finite values are recorded as failures.

Invocation reports are written to:

```text
<output_root>/qc/voxelwise_b32_qc.csv
<output_root>/qc/voxelwise_b32_errors.csv
<output_root>/qc/voxelwise_b32_run_metadata.json
<output_root>/qc/logs/voxelwise_b32.log
```

Identical reruns resume completed and expected-skipped units. A changed input, preset,
protocol, software stack or output fingerprint blocks reuse. `--force` archives the existing
unit directory before recomputing it.

## Scope

The current package implements `project init`, optional combined-mask lesion
materialization, `manifest`, `whole` and `voxelwise`. Habitat clustering, habitat-level
radiomics and scheduler adapters are later stages.
