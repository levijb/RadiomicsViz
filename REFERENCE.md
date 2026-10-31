# RadiomicsViz reference

Lookup material for `0.3.0.dev0`. Start with [README.md](README.md) for the workflow.

## Protocol and presets

The built-in `lesion-habitat-v1` protocol version 1.2.0 currently provides one variant,
`b32`, using 32 gray-level bins.

| Item | Reference value |
| --- | --- |
| Whole-region preset | `roi-allfeatures-bin32`; SHA-256 `cd2f3e1da1028daaec55b5573b7e2d986b7f34302617452103724d94ff0f942b` |
| Whole-region output | 1,834 features plus 37 diagnostics per successful unit; 14 features are 3-D shape features |
| Ordered whole-region feature schema | SHA-256 `f4ea6da5d5191bfb24ad81f05cf7792d7c853294e460fcaf959621c00f02240e` |
| Whole-region resampling | PyRadiomics requests `1 x 1 x 1 mm` with B-spline interpolation |
| Voxelwise preset | `voxelwise-original-bin32`; SHA-256 `ae781c704fd5688bd5ecf9f0a8083464e3981d21dbd2f1bd2a1e31ec48eb5cde` |
| Voxelwise output | 92 float64 NRRD maps per successful unit; original image only, no resampling, 3 x 3 x 3 neighborhood constrained to the region |
| Reference software | Python 3.10.10, PyRadiomics 3.0.1, SimpleITK 2.5.5, NumPy 1.26.4, PyWavelets 1.8.0 |
| Habitat software | The reference software plus SciPy 1.11.4 and scikit-learn 1.1.3 |

The whole-region preset file is byte-preserved. Its inline comment says the 1 mm setting
disables resampling, but the parsed PyRadiomics setting requests resampling; the comment
is not used as the scientific contract.

The 1,834 whole-region features consist of 105 original-image features and 91 features
from each of 19 filtered images: exponential, gradient, three LBP-3D images, Laplacian of
Gaussian at sigma 1, 3, and 5 mm, logarithm, square, square root, and eight wavelet bands.

The 92 voxelwise maps comprise 18 first-order, 23 GLCM, 16 GLRLM, 16 GLSZM, 14 GLDM, and
5 NGTDM maps. GLCM MCC is excluded because it is numerically unstable. On thin or small
regions, PyRadiomics 3.0.1 can produce an entirely non-finite
`original_glcm_JointAverage` map. That known map is retained with a warning and no
imputation; another entirely non-finite map is a failure.

## Minimum sizes

| Stage | Behavior |
| --- | --- |
| `lesions materialize` | Retains labels or components with at least 10 voxels; smaller candidates remain in provenance only |
| `manifest` | Warns below 50 voxels and for a region spanning fewer than two dimensions |
| `whole` | Skips fewer than 10 voxels, fewer than two dimensions, or an invalid intensity range with a resumable completion |
| `voxelwise` | Skips 10 voxels or fewer because PyRadiomics requires strictly more than `minimumROISize=10` |
| `habitats` | Uses full clustering at 50 voxels or more; uses a one-habitat radiomic fallback at 11-49; skips at 10 or fewer |
| `habitat-features` | Attempts every nonempty zone; a PyRadiomics rejection is an expected skip only when the zone is below 10 voxels or fewer than two dimensions |

Expected habitat-feature skips retain their rows, error text, and QC warnings but do not
make the command fail. Other habitat-feature extraction errors remain failures.

## Habitat methods

`habitats` combines paired T1 and FLAIR voxelwise maps and writes three label masks:

- `fixed_k2`: prespecified two-cluster KMeans, the primary habitat analysis;
- `adaptive_k1to3`: sensitivity analysis selecting k=1, 2, or 3;
- `geometric_1mm`: non-radiomic 1 mm shell and core comparison.

For regions with at least 50 voxels, processing is local to the region:

1. Scale each map by its median and interquartile range.
2. Replace non-finite scaled values with that map's finite local median.
3. Remove features jointly when absolute Spearman correlation is at least 0.90.
4. Weight each modality block by `1/sqrt(number of retained features)`.

KMeans uses seed 42 and 10 initializations. Labels are ordered from peripheral to central
by mean distance from the region boundary. `habitat_valid` is true when both fixed-k2
clusters contain at least `max(10 voxels, 5% of the region)`.

Each method has a cropped clustering-grid `mask.nii.gz` and a nearest-neighbor full-grid
`mask_source_grid.nii.gz`. Habitat-feature extraction uses the full-grid mask. Therefore,
`source_zone_voxels` is authoritative for extraction and for `NumofVoxels`; the
`habitat_grid_voxels` column records the cropped-grid count for comparison.

## Output files

Paths are relative to `output_root`. `<unit>` means
`subjects/<subject>/sessions/<session>/regions/<region>`.

| Stage | Principal files |
| --- | --- |
| `lesions materialize` | `materialized_regions/<subject>/<session>/<region_type>/<region>/mask.nii.gz` plus completion and provenance files |
| `manifest --write` | `manifest/regions_manifest.csv`, `input_checks.csv`, `errors.csv`, `run_metadata.json`, and `qc/logs/manifest.log` |
| `whole` | `tables/whole/b32/whole_region.csv`, a reference-format table, and `<unit>/radiomics/whole/b32/<modality>/` |
| `voxelwise` | `<unit>/radiomics/voxelwise/b32/<modality>/*.nrrd` and `completion.json` |
| `habitats` | `tables/habitats/b32/habitat_index.csv`, `lesion_qc.csv`, `<unit>/habitats/<method>/[b32/]mask*.nii.gz`, and a region completion |
| `habitat-features` | `tables/habitat_features/b32/habitat_features.csv`, three method tables, six method-modality reference-format tables, and `<unit>/habitat_features/<method>/b32/<modality>/features.csv` |

Each extraction stage also writes:

```text
qc/<stage>_b32_qc.csv
qc/<stage>_b32_errors.csv
qc/<stage>_b32_run_metadata.json
qc/logs/<stage>_b32.log
```

`--summarize` writes Markdown, JSON, and CSV summaries under the corresponding table
folder's `b32/summary/` directory. `--summary-dir` changes only that report location.

## Table columns

Feature columns follow `<image type>_<family>_<feature>`, for example
`wavelet-LLH_glcm_Contrast`. Diagnostics begin with `diagnostics_`.

The tidy tables are the primary outputs. They begin with explicit identifiers such as
`subject_id`, `session_id`, `region_id`, `method`, `zone_id`, and `modality`, followed by
provenance, diagnostics, and features. In `habitat_features.csv`, `zone_status` is
`complete`, `skipped`, `failed`, or `blocked`; `error` is empty for successful rows.

The compatibility tables preserve a legacy layout:

| Column | Meaning |
| --- | --- |
| `ID` | `subject_id` |
| `Exam` | `session_id` |
| `Label` | `region_id` |
| `Rim` | Empty compatibility field |
| `Image`, `Mask` | Input paths |
| `Cluster` | Habitat mask label; habitat-feature tables only |
| `NumofVoxels` | Region or full-source-grid zone voxel count |
| `Patient` | Whole table: `region_id`; habitat-feature tables: one-based row number within that table, not a patient identifier |
| `Reader` | `N-A` compatibility field |

Use the tidy tables for new analyses because their identifiers have one consistent
meaning. Each habitat-feature reference-format table has 1,881 columns: 10 compatibility
fields, 37 diagnostics, and 1,834 ordered features.

## Status and resume behavior

| Field | Values |
| --- | --- |
| QC `status` | `PASS`; `WARN` (retained but review); `FAIL` (not usable) |
| Unit outcome | `complete`, `resumed`, `skipped`, `failed`, or `blocked` |
| Manifest `reference_status` | `reference-compatible` or `non-reference-spacing` |
| Extraction `reference_status` | `reference-compatible` or `non-reference` |

A unit resumes only when its input and output fingerprints, preset, protocol, and checked
software still match. Modification times and folder locations do not determine identity.
`--force` archives the prior unit folder with a `.bak-<timestamp>` suffix and recomputes
it. Cohort reports and summaries also archive prior files before replacement. Nothing is
deleted automatically.

## Viewer

`radiomicviz view --manifest <output_root>/manifest/regions_manifest.csv` serves a
read-only dashboard on `127.0.0.1`. The printed address carries a secret launch token that the
browser exchanges for a session cookie, so other users of a shared machine cannot read the
data through the port. A non-loopback `--host` requires `--allow-remote`.

| Shown | Read from |
| --- | --- |
| Images and region masks | Manifest `image_path` and `mask_path` (served unchanged) |
| Voxelwise maps | `<unit>/radiomics/voxelwise/b32/<modality>/*.nrrd`, converted in memory to NIfTI |
| Habitat masks | `<unit>/habitats/<method>/[b32/]mask.nii.gz` and `mask_source_grid.nii.gz` |
| Status and provenance | Completion records, `qc/*_qc.csv`, `qc/*_errors.csv`, run metadata |
| Feature tables | `whole_region.csv` and `habitat_features.csv` (tidy tables only) |
| Summaries | Each stage's `--summarize` engine, run into a temporary folder |

Voxelwise maps and cropped habitat masks share a crop of the input-image grid (region
bounding box plus one voxel); source-grid masks, images, and region masks use the full
input grid. Display windows (2nd to 98th percentile of finite voxels), cohort percentiles,
and zone differences are computed by the viewer and labelled as such. `--blind` replaces
identifiers with pseudonyms and disables downloads; `--verify-integrity` checks the
SHA-256 fingerprints of a region's files when it is opened.

Blind mode is a display aid, not de-identification: it does not alter image anatomy or
arbitrary metadata embedded in source files. Use de-identified inputs and review screenshots
before sharing them.

The viewer never registers or resamples images, so it offers no group-average or
voxel-wise statistical maps. It is research software, not for clinical use.
