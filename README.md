# RadiomicsViz

RadiomicsViz is a manifest-driven MRI radiomics pipeline. It uses a fixed, versioned
PyRadiomics protocol to produce whole-region feature tables, per-voxel feature maps,
T1/FLAIR habitat masks, and whole-region features for each habitat zone. Every stage
writes QC records and resumable completion files.

Status: pre-release `0.3.0.dev0`. The reference environment targets Linux x86-64.

## Input contract

RadiomicsViz does not perform brain extraction, bias correction, intensity normalization,
or registration. Inputs must already be analysis-ready. For every subject, and optionally
every session, provide:

- one readable 3-D image per modality;
- region masks on the same voxel grid as the images;
- one folder per subject under `data_root` (a flat folder is not supported).

Masks may be supplied as one binary file per region, one integer-labeled map, or one
combined binary mask. The latter two forms are materialized into separate binary masks
before the manifest is written.

The manifest stage checks readability, finite image values, mask labels, geometry, and
voxel spacing. The reference input spacing is `1.0 +/- 0.001 mm` per axis. Separately,
the frozen whole-region preset asks PyRadiomics to resample to 1 mm with B-spline
interpolation; that preset operation is part of the validated feature definition.

## Pipeline

| Stage | Command | Main output |
| --- | --- | --- |
| Configure | `radiomicviz project init` | Commented `project.yaml` template |
| Separate combined masks | `radiomicviz lesions materialize` | One binary mask per retained region |
| Validate inputs | `radiomicviz manifest [--write]` | `manifest/regions_manifest.csv` |
| Whole-region radiomics | `radiomicviz whole` | One feature vector per region and modality |
| Voxelwise radiomics | `radiomicviz voxelwise` | 92 feature maps per region and modality |
| Habitat clustering | `radiomicviz habitats` | Fixed, adaptive, and geometric habitat masks |
| Habitat-zone radiomics | `radiomicviz habitat-features` | One feature vector per zone and modality |

Work is resumable by unit. Most units are one region-modality pair; `habitats` uses one
region, and `habitat-features` uses one region, method, and modality.

Common options are:

- `--dry-run` to show the plan without running the stage;
- `--jobs N` to process independent units concurrently;
- `--force` to archive incompatible existing units and recompute them;
- `--summarize` to create a read-only exploratory report for an existing run;
- `--variant b32`, the only protocol variant in this release.

## Install

Follow [INSTALLATION_DETAILS.md](INSTALLATION_DETAILS.md). After installation:

```bash
python -m pip check
radiomicviz --version
```

## Shortest end-to-end example

Work in a study folder outside the repository. This example assumes the default input
form: one binary file per region.

```bash
mkdir my-study && cd my-study
radiomicviz project init
# Edit project.yaml: set data_root and adjust the subject, image, and mask patterns.
# output_root defaults to ./out.
radiomicviz manifest --project project.yaml
radiomicviz manifest --project project.yaml --write

M="--manifest out/manifest/regions_manifest.csv"
radiomicviz whole $M --dry-run
radiomicviz whole $M --jobs 2
radiomicviz voxelwise $M --jobs 2
radiomicviz habitats $M --jobs 1
radiomicviz habitat-features $M --jobs 2
radiomicviz habitat-features $M --summarize
```

For `labeled_map` or `connected_components` input, run this before `manifest`:

```bash
radiomicviz lesions materialize --project project.yaml --dry-run
radiomicviz lesions materialize --project project.yaml
```

Start with one or two workers on a small cohort and measure memory before scaling up.
Habitat clustering loads paired T1 and FLAIR map sets for a region and should start with
`--jobs 1`.

## Configure project.yaml

`project.yaml` describes input discovery and the output root; scientific settings remain
inside the built-in protocol.

| Key | Meaning |
| --- | --- |
| `data_root` | Folder containing one folder per subject |
| `output_root` | Folder for all generated files; keep it outside `data_root` |
| `subjects` | Folder glob and regex containing a `(?P<subject>...)` group |
| `sessions.mode` | `single` or `discovered` |
| `modalities.<name>.image_pattern` | Exactly one analysis-ready image per subject/session |
| `regions.mode` | `per_region_masks`, `labeled_map`, or `connected_components` |
| `regions.mask_pattern` | Separated mask glob or one combined-mask path |
| `allow_nonreference_spacing` | Permit non-1-mm input and mark the manifest non-reference |

Patterns are relative to `data_root`, use forward slashes, and may contain `{subject}`.
With `sessions.mode: discovered`, every image and mask pattern must also contain
`{session}`. In `single` mode, `{session}` is replaced by `default_session_id`; if other
directories could be sessions, the manifest prints a warning so they are not silently
overlooked.

Habitat clustering requires modality keys named `T1` and `FLAIR`, even when the source
files use names such as T1w or FLAIR3D.

### Region input forms

`per_region_masks` reads already-separated binary masks:

```yaml
regions:
  mode: per_region_masks
  region_type: lesion
  mask_pattern: '{subject}/lesions/*.nii.gz'
  id_regex: '^(?P<region>lesion_[0-9]+)\.nii\.gz$'
  label: 1
  connectivity: null
```

`labeled_map` preserves every positive integer label as one region. `connected_components`
splits a configured binary foreground using 6- or 26-connectivity. Connected components
are numbered by descending voxel count, so the largest is `lesion_001`; touching
foreground voxels remain one component.

Materialization retains components or labels with at least 10 voxels. User-supplied masks
are checked again by each extraction stage. Whole-region extraction accepts 10 voxels;
voxelwise extraction requires more than 10.

## Non-reference runs

Setting `allow_nonreference_spacing: true` permits the manifest to contain non-1-mm
inputs and stamps it `non-reference-spacing`. Extraction still requires an explicit
acknowledgement:

```bash
radiomicviz whole --manifest out/manifest/regions_manifest.csv --allow-nonreference
```

Use the same flag for later extraction stages. The legacy name
`--allow-version-mismatch` remains accepted. Both flags record the deviations and stamp
the outputs `non-reference`; neither changes the scientific preset.

## Results

Open these tables first:

| Path under `output_root` | Contents |
| --- | --- |
| `tables/whole/b32/whole_region.csv` | Whole-region features by region and modality |
| `tables/habitats/b32/habitat_index.csv` | Seven habitat-zone records per completed region |
| `tables/habitat_features/b32/habitat_features.csv` | Habitat-zone features by method and modality |
| `qc/<stage>_b32_qc.csv` | `PASS`, `WARN`, and `FAIL` checks |
| `qc/<stage>_b32_errors.csv` | Blocking failures only |
| `tables/<stage>/b32/summary/*.md` | Human-readable `--summarize` report |

Voxelwise maps, habitat masks, and per-unit completion files are under:

```text
subjects/<subject>/sessions/<session>/regions/<region>/
```

The tidy tables above are the primary outputs. `*_reference_format.csv` files preserve a
legacy wide layout; their columns are defined in [REFERENCE.md](REFERENCE.md).

Tiny or one-dimensional habitat zones are still attempted in the final stage. If
PyRadiomics rejects such a zone, its table row is retained with `zone_status=skipped` and
a QC warning; the run does not fail. Unexpected extraction errors remain failures and
produce a nonzero exit code.

## Viewer

`radiomicviz view` opens a read-only browser dashboard for one output tree: images, region
masks, voxelwise maps, habitat masks, feature tables, cohort context, QC, and provenance.
It needs Flask (included in `environment.yml`, or the `viewer` extra) and never writes to
`data_root` or `output_root`.

```bash
radiomicviz view --manifest out/manifest/regions_manifest.csv --no-browser
```

Open the printed `http://127.0.0.1:<port>/?token=...` address. On a remote machine, first
run `ssh -L <port>:localhost:<port> <cluster-login>`; VS Code forwards printed ports
automatically. `--blind` hides identifiers. Details are in [REFERENCE.md](REFERENCE.md#viewer).
Blind mode does not de-identify image anatomy or embedded file metadata.

## Limitations

- Linux x86-64 with the pinned environment is the reference platform. Windows is used for
  development tests only and should use a short output and pytest temporary path.
- No preprocessing, registration, or watershed splitting of touching foreground is
  provided.
- Habitat clustering is defined for paired T1 and FLAIR feature maps.
- The package is not published on PyPI.

## More information

- [INSTALLATION_DETAILS.md](INSTALLATION_DETAILS.md): reference installation,
  verification, tests, and troubleshooting.
- [REFERENCE.md](REFERENCE.md): presets, thresholds, habitat methods, output schemas, and
  status values.
- [CHANGELOG.md](CHANGELOG.md): changes by version.
- `radiomicviz <command> --help`: command options.
