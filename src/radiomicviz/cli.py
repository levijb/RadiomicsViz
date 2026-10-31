"""Command-line interface for RadiomicsViz (program name ``radiomicviz``)."""

from __future__ import annotations

from pathlib import Path

import click

from radiomicviz import __version__
from radiomicviz.config import ConfigError, load_project_config
from radiomicviz.config.templates import write_template
from radiomicviz.protocols import ProtocolError, load_protocol
from radiomicviz.policy import MINIMUM_REGION_VOXELS

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(name="radiomicviz", context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "--version", prog_name="radiomicviz", message="%(prog)s %(version)s")
def main() -> None:
    """RadiomicsViz: a stage-wise, manifest-driven MRI radiomics pipeline.

    A run uses a built-in scientific protocol and project.yaml, which describes
    where analysis-ready images and region masks are located. Start with
    "radiomicviz project init".
    """


@main.group(context_settings=CONTEXT_SETTINGS)
def project() -> None:
    """Project configuration (project.yaml).

    The project file describes where analysis-ready images and region masks are and
    how subject, session and region identifiers are read from their paths.
    """


@project.command("init", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("project.yaml"),
    show_default=True,
    help="Where to write the template.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing file. The existing file is first renamed to <name>.bak-<timestamp>.",
)
def project_init(output: Path, force: bool) -> None:
    """Write a commented project.yaml template to edit.

    The template describes where analysis-ready images and region masks are and how
    subject, session and region identifiers are read from their paths.
    """
    _write("project", output, force)


@main.group(context_settings=CONTEXT_SETTINGS)
def lesions() -> None:
    """Prepare lesion masks for the manifest.

    Projects that already have one binary file per lesion do not use this command.
    Combined labeled or binary masks must be materialized before the manifest stage.
    """


@lesions.command("materialize", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--project",
    "project_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Validated project.yaml using labeled_map or connected_components mode.",
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override project.yaml output_root for this run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate inputs and show planned lesion masks without writing files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Archive and replace existing materialized masks instead of resuming them.",
)
def lesions_materialize(
    project_path: Path,
    output_root: Path | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Split one combined mask per subject/session into binary lesion masks.

    'labeled_map' preserves each positive integer label as one lesion.
    'connected_components' splits a binary foreground using the connectivity stated
    in project.yaml. Touching foreground voxels remain one lesion. Regions smaller than
    the package-wide minimum are recorded in provenance but are not materialized.
    """
    from radiomicviz.lesions import (
        MaterializationError,
        build_materialization_plan,
        run_materialization,
    )

    try:
        project_config = load_project_config(project_path)
        plan = build_materialization_plan(project_config, output_root=output_root)
        result = run_materialization(
            plan, dry_run=dry_run, force=force, progress=_echo_progress
        )
    except (ConfigError, MaterializationError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None

    click.echo("Lesion materialization plan" if dry_run else "Lesion materialization result")
    click.echo(f"  mode: {plan.project.regions.mode}")
    if plan.project.regions.connectivity is not None:
        click.echo(f"  connectivity: {plan.project.regions.connectivity}")
    click.echo(f"  minimum lesion size: {MINIMUM_REGION_VOXELS} voxels")
    click.echo(f"  output root: {plan.output_root}")
    click.echo(f"  subject/session units: {len(plan.units)}")
    for unit in result.units:
        detail = f" ({unit.message})" if unit.message else ""
        click.echo(
            f"  {unit.status:<29} {unit.subject_id}/{unit.session_id}: "
            f"{unit.n_regions} lesion(s), {unit.n_excluded_regions} below minimum "
            f"-> {unit.output_dir}{detail}"
        )
    if dry_run:
        click.echo("Dry run: no lesion masks were written.")
    else:
        click.echo(f"  completed: {result.completed}")
        click.echo(f"  resumed: {result.resumed}")
    if not result.valid:
        raise click.ClickException(
            f"lesion materialization has {result.blocked} blocked and {result.failed} failed "
            "unit(s); review the messages above"
        )


@main.command("manifest", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--project",
    "project_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Validated project.yaml describing input discovery.",
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override project.yaml output_root for this run.",
)
@click.option(
    "--write",
    is_flag=True,
    help=(
        "Write the manifest and QC artifacts after previewing. Previous artifacts are "
        "renamed with a .bak-<timestamp> suffix first. Without this flag nothing is written."
    ),
)
def manifest_command(
    project_path: Path,
    output_root: Path | None,
    write: bool,
) -> None:
    """Discover inputs, validate them, and build regions_manifest.csv.

    Separate binary region masks are read directly. Combined masks must first be split
    with 'radiomicviz lesions materialize'. The default is a read-only preview. Review
    it, then repeat with --write to create the canonical manifest, input checks, errors
    table, run metadata and one concise log. Rerun with --write after adding subjects;
    completed radiomics units can still resume when their inputs are unchanged.
    """
    from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs

    try:
        project_config = load_project_config(project_path)
        preview = build_manifest_preview(
            project_config, output_root=output_root, progress=_echo_progress
        )
    except (ConfigError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None

    _print_manifest_preview(preview)
    if not write:
        if preview.valid:
            click.echo("Preview only: no files written. Repeat with --write after review.")
            return
        raise click.ClickException("manifest preview has blocking failures; no files were written")

    try:
        result = write_manifest_outputs(preview)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"Wrote input checks to {result.checks_path}")
    click.echo(f"Wrote errors to {result.errors_path}")
    click.echo(f"Wrote run metadata to {result.metadata_path}")
    click.echo(f"Wrote log to {result.log_path}")
    if result.manifest_path is not None:
        click.echo(f"Wrote canonical manifest to {result.manifest_path}")
        click.echo(
            "Next: radiomicviz whole --manifest "
            f'"{result.manifest_path}" --dry-run'
        )
    if result.backups:
        click.echo(f"Archived {len(result.backups)} previous artifact(s) with a .bak-<timestamp> suffix")
    if not preview.valid:
        raise click.ClickException(
            "manifest has blocking failures; diagnostic artifacts were written but no canonical manifest was created"
        )


def _print_manifest_preview(preview) -> None:
    click.echo("Manifest preview")
    click.echo(f"  data root: {preview.data_root}")
    click.echo(f"  output root: {preview.output_root}")
    click.echo(f"  subjects: {preview.subjects_found}")
    click.echo(f"  sessions: {preview.sessions_found}")
    click.echo(f"  regions: {preview.regions_found}")
    click.echo(f"  modalities: {', '.join(preview.modalities_found) or 'none'}")
    click.echo(f"  rows: {len(preview.rows)}")
    click.echo(f"  warnings: {preview.warning_count}")
    click.echo(f"  failures: {preview.failure_count}")
    spacing_status = (
        "reference" if preview.reference_status == "reference-compatible"
        else "non-reference"
    )
    click.echo(f"  input spacing status: {spacing_status}")
    warnings = [check for check in preview.checks if check.status == "WARN"]
    for warning in warnings[:20]:
        scope = "/".join(
            value
            for value in (
                warning.subject_id,
                warning.session_id,
                warning.region_id,
                warning.modality,
            )
            if value
        )
        prefix = f" [{scope}]" if scope else ""
        click.echo(f"  WARN{prefix} {warning.check_id}: {warning.message}")
    if len(warnings) > 20:
        click.echo(f"  ... {len(warnings) - 20} additional warning(s)")
    for error in preview.errors[:10]:
        scope = "/".join(
            value for value in (error.subject_id, error.session_id, error.region_id, error.modality) if value
        )
        prefix = f" [{scope}]" if scope else ""
        click.echo(f"  FAIL{prefix} {error.error_code}: {error.message}")
    if len(preview.errors) > 10:
        click.echo(f"  ... {len(preview.errors) - 10} additional failure(s)")


@main.command("whole", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Canonical regions_manifest.csv written by the manifest stage.",
)
@click.option(
    "--protocol",
    "protocol_id",
    default="lesion-habitat-v1",
    show_default=True,
    help="Built-in scientific protocol identifier.",
)
@click.option(
    "--variant",
    default="b32",
    show_default=True,
    help="Protocol variant. The initial reference implementation provides b32.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of region-modality units to extract concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which units would run, resume or be blocked without writing files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Archive existing unit outputs and recompute them instead of resuming.",
)
@click.option(
    "--allow-version-mismatch",
    "--allow-nonreference",
    "allow_version_mismatch",
    is_flag=True,
    help="Explicitly allow a non-reference run and stamp completions, tables and metadata.",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Summarize existing whole-region outputs without rerunning extraction.",
)
@click.option(
    "--summary-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional report directory (default: tables/whole/<variant>/summary).",
)
@click.option(
    "--summary-top-features",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Number of features shown in the Markdown completeness table.",
)
def whole_command(
    manifest_path: Path,
    protocol_id: str,
    variant: str,
    jobs: int,
    dry_run: bool,
    force: bool,
    allow_version_mismatch: bool,
    summarize: bool,
    summary_dir: Path | None,
    summary_top_features: int,
) -> None:
    """Extract reference whole-region PyRadiomics features.

    Every manifest region and modality is one resumable unit. Each manifest image is
    assumed to be preprocessed, co-registered with its mask and analysis-ready.
    """
    from radiomicviz.whole import (
        WholePlanError,
        WholeStageError,
        WholeSummaryError,
        load_whole_plan,
        run_whole,
        summarize_whole,
    )

    if summarize:
        if dry_run or force:
            raise click.ClickException(
                "--summarize cannot be combined with --dry-run or --force"
            )
        try:
            summary = summarize_whole(
                manifest_path,
                variant=variant,
                summary_dir=summary_dir,
                top_features=summary_top_features,
            )
        except (WholeSummaryError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from None
        click.echo("Whole-region radiomics summary")
        click.echo(f"  manifest: {manifest_path.resolve()}")
        click.echo(f"  output root: {summary.output_root}")
        click.echo(f"  variant: {summary.variant}")
        click.echo(f"  units: {summary.total_units}")
        click.echo(f"  successful: {summary.successful_units}")
        click.echo(f"  skipped: {summary.skipped_units}")
        click.echo(f"  failed or blocked: {summary.failed_units}")
        click.echo(f"  features: {summary.feature_count}")
        click.echo(
            "  source run time: "
            + _format_duration(summary.source_run_runtime_seconds)
        )
        click.echo(f"  report directory: {summary.summary_dir}")
        click.echo(f"  Markdown: {summary.markdown_path}")
        click.echo(f"  JSON: {summary.json_path}")
        click.echo(f"  counts: {summary.counts_path}")
        click.echo(f"  numeric summaries: {summary.numeric_path}")
        click.echo(f"  by subject/session: {summary.subjects_path}")
        click.echo(f"  feature summaries: {summary.features_path}")
        if summary.backups:
            click.echo(f"  archived prior reports: {len(summary.backups)}")
        return
    if summary_dir is not None:
        raise click.ClickException("--summary-dir requires --summarize")

    try:
        plan = load_whole_plan(manifest_path)
        protocol = load_protocol(protocol_id)
        result = run_whole(
            plan,
            protocol,
            variant=variant,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
            allow_version_mismatch=allow_version_mismatch,
            progress=_echo_progress,
        )
    except (ProtocolError, WholePlanError, WholeStageError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None

    click.echo("Whole-region plan" if dry_run else "Whole-region result")
    click.echo(f"  manifest: {result.manifest_path}")
    click.echo(f"  output root: {result.output_root}")
    click.echo(f"  protocol: {result.protocol_id} {result.protocol_version}")
    click.echo(f"  variant: {result.variant}")
    click.echo(f"  preset: {result.preset_id}")
    click.echo(f"  preset SHA-256: {result.preset_sha256}")
    click.echo(f"  region-modality units: {len(result.units)}")
    click.echo(f"  concurrent jobs: {result.jobs}")
    click.echo(
        "  reference status: "
        + ("reference-compatible" if result.reference_compatible else "non-reference")
    )
    for deviation in result.reference_deviations:
        click.echo(f"    - {deviation}")
    if dry_run:
        for unit in result.units:
            click.echo(f"  {unit.status:<14} {unit.unit_key} -> {unit.output_dir}")
        _print_plan_totals(result.units)
        click.echo("Dry run: no extraction performed and no files written.")
        if result.blocked:
            raise click.ClickException(
                f"{result.blocked} unit(s) have outputs without a matching completion "
                "record; rerun with --force to archive and recompute them"
            )
        return

    click.echo(f"  completed: {result.succeeded}")
    click.echo(f"  resumed: {result.resumed}")
    click.echo(f"  skipped (< minimum ROI size): {result.skipped}")
    click.echo(f"  failed: {result.failed}")
    click.echo(f"  warnings: {result.warnings}")
    click.echo(f"  table: {result.table_path}")
    click.echo(f"  reference-format table: {result.reference_table_path}")
    click.echo(f"  QC: {result.checks_path}")
    click.echo(f"  errors: {result.errors_path}")
    click.echo(f"  metadata: {result.metadata_path}")
    click.echo(f"  log: {result.log_path}")
    if not result.valid:
        raise click.ClickException(
            f"whole-region extraction failed for {result.failed} unit(s); "
            f"see {result.errors_path}"
        )


@main.command("voxelwise", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Canonical regions_manifest.csv written by the manifest stage.",
)
@click.option(
    "--protocol",
    "protocol_id",
    default="lesion-habitat-v1",
    show_default=True,
    help="Built-in scientific protocol identifier.",
)
@click.option(
    "--variant",
    default="b32",
    show_default=True,
    help="Protocol variant. The initial reference implementation provides b32.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of region-modality units to extract concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which units would run, resume or be blocked without writing files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Archive existing unit outputs and recompute them instead of resuming.",
)
@click.option(
    "--allow-version-mismatch",
    "--allow-nonreference",
    "allow_version_mismatch",
    is_flag=True,
    help="Explicitly allow a non-reference run and stamp completions, tables and metadata.",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Summarize existing voxelwise outputs without rerunning extraction.",
)
@click.option(
    "--summary-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional report directory (default: tables/voxelwise/<variant>/summary).",
)
@click.option(
    "--summary-top-maps",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Number of lowest-coverage maps shown in the Markdown report.",
)
def voxelwise_command(
    manifest_path: Path,
    protocol_id: str,
    variant: str,
    jobs: int,
    dry_run: bool,
    force: bool,
    allow_version_mismatch: bool,
    summarize: bool,
    summary_dir: Path | None,
    summary_top_maps: int,
) -> None:
    """Extract reference voxelwise PyRadiomics feature maps.

    Every manifest region and modality is one resumable unit. Successful units contain
    one float64 NRRD per protocol feature. Regions at or below the preset's minimum ROI
    size (minimumROISize = 10) are recorded as expected skips because PyRadiomics requires
    the voxel count to be strictly greater than that setting.
    """
    from radiomicviz.voxelwise import (
        VoxelwisePlanError,
        VoxelwiseStageError,
        VoxelwiseSummaryError,
        load_voxelwise_plan,
        run_voxelwise,
        summarize_voxelwise,
    )

    if summarize:
        if dry_run or force:
            raise click.ClickException(
                "--summarize cannot be combined with --dry-run or --force"
            )
        try:
            summary = summarize_voxelwise(
                manifest_path,
                variant=variant,
                summary_dir=summary_dir,
                top_maps=summary_top_maps,
            )
        except (VoxelwiseSummaryError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from None
        click.echo("Voxelwise radiomics summary")
        click.echo(f"  manifest: {manifest_path.resolve()}")
        click.echo(f"  output root: {summary.output_root}")
        click.echo(f"  variant: {summary.variant}")
        click.echo(f"  units: {summary.total_units}")
        click.echo(f"  successful: {summary.successful_units}")
        click.echo(f"  skipped: {summary.skipped_units}")
        click.echo(f"  failed or blocked: {summary.failed_units}")
        click.echo(f"  maps: {summary.map_count}")
        click.echo(
            "  source run time: "
            + _format_duration(summary.source_run_runtime_seconds)
        )
        click.echo(f"  report directory: {summary.summary_dir}")
        click.echo(f"  Markdown: {summary.markdown_path}")
        click.echo(f"  JSON: {summary.json_path}")
        click.echo(f"  counts: {summary.counts_path}")
        click.echo(f"  numeric summaries: {summary.numeric_path}")
        click.echo(f"  by subject/session: {summary.subjects_path}")
        click.echo(f"  map coverage: {summary.maps_path}")
        if summary.backups:
            click.echo(f"  archived prior reports: {len(summary.backups)}")
        return
    if summary_dir is not None:
        raise click.ClickException("--summary-dir requires --summarize")

    try:
        plan = load_voxelwise_plan(manifest_path)
        protocol = load_protocol(protocol_id)
        result = run_voxelwise(
            plan,
            protocol,
            variant=variant,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
            allow_version_mismatch=allow_version_mismatch,
            progress=_echo_progress,
        )
    except (
        ProtocolError,
        VoxelwisePlanError,
        VoxelwiseStageError,
        OSError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from None

    click.echo("Voxelwise plan" if dry_run else "Voxelwise result")
    click.echo(f"  manifest: {result.manifest_path}")
    click.echo(f"  output root: {result.output_root}")
    click.echo(f"  protocol: {result.protocol_id} {result.protocol_version}")
    click.echo(f"  variant: {result.variant}")
    click.echo(f"  preset: {result.preset_id}")
    click.echo(f"  preset SHA-256: {result.preset_sha256}")
    click.echo(f"  region-modality units: {len(result.units)}")
    click.echo(f"  concurrent jobs: {result.jobs}")
    click.echo(
        "  reference status: "
        + ("reference-compatible" if result.reference_compatible else "non-reference")
    )
    for deviation in result.reference_deviations:
        click.echo(f"    - {deviation}")
    if dry_run:
        for unit in result.units:
            click.echo(f"  {unit.status:<14} {unit.unit_key} -> {unit.output_dir}")
        _print_plan_totals(result.units)
        click.echo("Dry run: no extraction performed and no files written.")
        if result.blocked:
            raise click.ClickException(
                f"{result.blocked} unit(s) have outputs without a matching completion "
                "record; rerun with --force to archive and recompute them"
            )
        return

    click.echo(f"  completed: {result.succeeded}")
    click.echo(f"  resumed: {result.resumed}")
    click.echo(f"  skipped (ROI <= minimumROISize): {result.skipped}")
    click.echo(f"  failed: {result.failed}")
    click.echo(f"  warnings: {result.warnings}")
    click.echo(f"  QC: {result.checks_path}")
    click.echo(f"  errors: {result.errors_path}")
    click.echo(f"  metadata: {result.metadata_path}")
    click.echo(f"  log: {result.log_path}")
    if not result.valid:
        raise click.ClickException(
            f"voxelwise extraction failed for {result.failed} unit(s); "
            f"see {result.errors_path}"
        )


@main.command("habitats", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Canonical regions_manifest.csv used by the completed voxelwise stage.",
)
@click.option(
    "--protocol",
    "protocol_id",
    default="lesion-habitat-v1",
    show_default=True,
    help="Built-in scientific protocol identifier.",
)
@click.option(
    "--variant",
    default="b32",
    show_default=True,
    help="Voxelwise protocol variant whose paired T1 and FLAIR maps will be clustered.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of regions to cluster concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which regions would run, resume or be blocked without writing files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Archive existing habitat outputs and recompute them instead of resuming.",
)
@click.option(
    "--allow-version-mismatch",
    "--allow-nonreference",
    "allow_version_mismatch",
    is_flag=True,
    help="Explicitly allow a non-reference run and stamp completions, tables and metadata.",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Summarize existing habitat outputs without rerunning clustering.",
)
@click.option(
    "--summary-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional report directory (default: tables/habitats/<variant>/summary).",
)
@click.option(
    "--summary-top-features",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Number of most frequently selected features shown in the Markdown report.",
)
def habitats_command(
    manifest_path: Path,
    protocol_id: str,
    variant: str,
    jobs: int,
    dry_run: bool,
    force: bool,
    allow_version_mismatch: bool,
    summarize: bool,
    summary_dir: Path | None,
    summary_top_features: int,
) -> None:
    """Create fixed, adaptive and geometric habitat masks.

    The reference method jointly uses paired T1 and FLAIR voxelwise feature maps.
    Both modalities must have completed the selected voxelwise variant first.
    """
    from radiomicviz.habitats import (
        HabitatPlanError,
        HabitatSummaryError,
        HabitatStageError,
        load_habitat_plan,
        run_habitats,
        summarize_habitats,
    )

    if summarize:
        if dry_run or force:
            raise click.ClickException(
                "--summarize cannot be combined with --dry-run or --force"
            )
        try:
            summary = summarize_habitats(
                manifest_path,
                variant=variant,
                summary_dir=summary_dir,
                top_features=summary_top_features,
            )
        except (HabitatSummaryError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from None

        click.echo("Habitat summary")
        click.echo(f"  manifest: {manifest_path.resolve()}")
        click.echo(f"  output root: {summary.output_root}")
        click.echo(f"  variant: {summary.variant}")
        click.echo(f"  regions represented: {summary.total_regions}")
        click.echo(f"  completed: {summary.completed_regions}")
        click.echo(f"  skipped: {summary.skipped_regions}")
        click.echo(f"  failed units: {summary.failed_units}")
        click.echo(f"  valid fixed k=2: {summary.fixed_valid_regions}")
        click.echo(f"  adaptive k=2 or k=3: {summary.adaptive_split_regions}")
        click.echo(
            "  source run time: "
            + _format_duration(summary.source_run_runtime_seconds)
        )
        click.echo(f"  report directory: {summary.summary_dir}")
        click.echo(f"  Markdown: {summary.markdown_path}")
        click.echo(f"  JSON: {summary.json_path}")
        click.echo(f"  counts: {summary.counts_path}")
        click.echo(f"  numeric summaries: {summary.numeric_path}")
        click.echo(f"  by subject/session: {summary.subjects_path}")
        click.echo(f"  feature frequencies: {summary.features_path}")
        if summary.backups:
            click.echo(f"  archived prior reports: {len(summary.backups)}")
        return

    if summary_dir is not None:
        raise click.ClickException("--summary-dir requires --summarize")

    try:
        plan = load_habitat_plan(manifest_path)
        protocol = load_protocol(protocol_id)
        result = run_habitats(
            plan,
            protocol,
            variant=variant,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
            allow_version_mismatch=allow_version_mismatch,
            progress=_echo_progress,
        )
    except (
        ProtocolError,
        HabitatPlanError,
        HabitatStageError,
        OSError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from None

    click.echo("Habitat plan" if dry_run else "Habitat result")
    click.echo(f"  manifest: {result.manifest_path}")
    click.echo(f"  output root: {result.output_root}")
    click.echo(f"  protocol: {result.protocol_id} {result.protocol_version}")
    click.echo(f"  variant: {result.variant}")
    click.echo(f"  implementation: {result.implementation_id}")
    click.echo(f"  scientific config SHA-256: {result.scientific_config_sha256}")
    click.echo(f"  region units: {len(result.units)}")
    click.echo(f"  concurrent jobs: {result.jobs}")
    click.echo(
        "  reference status: "
        + ("reference-compatible" if result.reference_compatible else "non-reference")
    )
    for deviation in result.reference_deviations:
        click.echo(f"    - {deviation}")
    if dry_run:
        for unit in result.units:
            detail = f" ({unit.message})" if unit.message else ""
            click.echo(
                f"  {unit.status:<14} {unit.unit_key} -> {unit.output_dir}{detail}"
            )
        _print_plan_totals(result.units)
        click.echo("Dry run: no clustering performed and no files written.")
        if result.blocked:
            raise click.ClickException(
                f"{result.blocked} region(s) are blocked; review the messages above"
            )
        return

    click.echo(f"  completed: {result.succeeded}")
    click.echo(f"  resumed: {result.resumed}")
    click.echo(f"  skipped (voxelwise maps unavailable): {result.skipped}")
    click.echo(f"  failed: {result.failed}")
    click.echo(f"  warnings: {result.warnings}")
    click.echo(f"  habitat index: {result.table_path}")
    click.echo(f"  lesion QC: {result.qc_table_path}")
    click.echo(f"  QC: {result.checks_path}")
    click.echo(f"  errors: {result.errors_path}")
    click.echo(f"  metadata: {result.metadata_path}")
    click.echo(f"  log: {result.log_path}")
    if not result.valid:
        raise click.ClickException(
            f"habitat clustering failed for {result.failed} region(s); "
            f"see {result.errors_path}"
        )


@main.command("habitat-features", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Canonical regions_manifest.csv used by the completed habitat stage.",
)
@click.option(
    "--protocol",
    "protocol_id",
    default="lesion-habitat-v1",
    show_default=True,
    help="Built-in scientific protocol identifier.",
)
@click.option(
    "--variant",
    default="b32",
    show_default=True,
    help="ROI preset variant to apply to every non-empty habitat zone.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of region-method-modality units to extract concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which units would run, resume or be blocked without writing files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Archive existing habitat-feature units and recompute them instead of resuming.",
)
@click.option(
    "--allow-version-mismatch",
    "--allow-nonreference",
    "allow_version_mismatch",
    is_flag=True,
    help="Explicitly allow a non-reference run and stamp completions, tables and metadata.",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Summarize existing habitat-zone feature outputs without rerunning extraction.",
)
@click.option(
    "--summary-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional report directory (default: tables/habitat_features/<variant>/summary).",
)
@click.option(
    "--summary-top-features",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Maximum number of incomplete features shown in the Markdown report.",
)
def habitat_features_command(
    manifest_path: Path,
    protocol_id: str,
    variant: str,
    jobs: int,
    dry_run: bool,
    force: bool,
    allow_version_mismatch: bool,
    summarize: bool,
    summary_dir: Path | None,
    summary_top_features: int,
) -> None:
    """Extract whole-ROI radiomics from every non-empty habitat zone.

    The command reproduces the reference final analysis step: the unchanged whole-ROI
    parameter preset is run on each label in the full source-grid fixed,
    adaptive and geometric habitat masks for both T1 and FLAIR.
    """
    from radiomicviz.habitat_features import (
        HabitatFeaturePlanError,
        HabitatFeatureStageError,
        HabitatFeatureSummaryError,
        load_habitat_feature_plan,
        run_habitat_features,
        summarize_habitat_features,
    )

    if summarize:
        if dry_run or force:
            raise click.ClickException(
                "--summarize cannot be combined with --dry-run or --force"
            )
        try:
            summary = summarize_habitat_features(
                manifest_path,
                variant=variant,
                summary_dir=summary_dir,
                top_features=summary_top_features,
            )
        except (HabitatFeatureSummaryError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from None
        click.echo("Habitat-zone radiomics summary")
        click.echo(f"  manifest: {manifest_path.resolve()}")
        click.echo(f"  output root: {summary.output_root}")
        click.echo(f"  variant: {summary.variant}")
        click.echo(f"  zone-modality rows: {summary.total_rows}")
        click.echo(f"  successful: {summary.successful_rows}")
        click.echo(f"  expected skipped: {summary.skipped_rows}")
        click.echo(f"  failed or blocked: {summary.failed_rows}")
        click.echo(f"  features: {summary.feature_count}")
        click.echo(
            "  source run time: "
            + _format_duration(summary.source_run_runtime_seconds)
        )
        click.echo(f"  report directory: {summary.summary_dir}")
        click.echo(f"  Markdown: {summary.markdown_path}")
        click.echo(f"  JSON: {summary.json_path}")
        click.echo(f"  counts: {summary.counts_path}")
        click.echo(f"  numeric summaries: {summary.numeric_path}")
        click.echo(f"  by subject/session: {summary.subjects_path}")
        click.echo(f"  feature summaries: {summary.features_path}")
        if summary.backups:
            click.echo(f"  archived prior reports: {len(summary.backups)}")
        return
    if summary_dir is not None:
        raise click.ClickException("--summary-dir requires --summarize")

    try:
        plan = load_habitat_feature_plan(manifest_path, variant=variant)
        protocol = load_protocol(protocol_id)
        result = run_habitat_features(
            plan,
            protocol,
            variant=variant,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
            allow_version_mismatch=allow_version_mismatch,
            progress=_echo_progress,
        )
    except (
        ProtocolError,
        HabitatFeaturePlanError,
        HabitatFeatureStageError,
        OSError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from None

    click.echo("Habitat-feature plan" if dry_run else "Habitat-feature result")
    click.echo(f"  manifest: {result.manifest_path}")
    click.echo(f"  habitat index: {result.habitat_index_path}")
    click.echo(f"  output root: {result.output_root}")
    click.echo(f"  protocol: {result.protocol_id} {result.protocol_version}")
    click.echo(f"  variant: {result.variant}")
    click.echo(f"  method-modality units: {len(result.units)}")
    zone_rows = (
        sum(len(unit.zones) for unit in plan.units) if dry_run else result.zone_rows
    )
    click.echo(f"  zone-modality rows: {zone_rows}")
    click.echo(f"  concurrent jobs: {result.jobs}")
    click.echo(
        "  reference status: "
        + ("reference-compatible" if result.reference_compatible else "non-reference")
    )
    for deviation in result.reference_deviations:
        click.echo(f"    - {deviation}")
    if dry_run:
        for unit in result.units:
            click.echo(f"  {unit.status:<14} {unit.unit_key} -> {unit.output_dir}")
        _print_plan_totals(result.units)
        click.echo("Dry run: no extraction performed and no files written.")
        if result.blocked:
            raise click.ClickException(
                f"{result.blocked} unit(s) have outputs without a matching completion "
                "record; rerun with --force to archive and recompute them"
            )
        return

    click.echo(f"  completed units: {result.completed}")
    click.echo(f"  resumed units: {result.resumed}")
    click.echo(f"  successful zone rows: {result.successful_zones}")
    click.echo(f"  expected skipped zone rows: {result.skipped_zones}")
    click.echo(f"  failed zone rows: {result.failed}")
    click.echo(f"  warnings: {result.warnings}")
    click.echo(f"  cohort table: {result.table_path}")
    if result.method_table_paths:
        click.echo(f"  method tables: {len(result.method_table_paths)}")
    if result.reference_table_paths:
        click.echo(f"  reference-format tables: {len(result.reference_table_paths)}")
    click.echo(f"  QC: {result.checks_path}")
    click.echo(f"  errors: {result.errors_path}")
    click.echo(f"  metadata: {result.metadata_path}")
    click.echo(f"  log: {result.log_path}")
    if not result.valid:
        raise click.ClickException(
            f"habitat-feature extraction failed for {result.failed} zone-modality "
            f"row(s); all rows were retained in {result.table_path} and details are in "
            f"{result.errors_path}"
        )


@main.command("view", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Canonical regions_manifest.csv of the output tree to open.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to listen on. A non-loopback host also needs --allow-remote.",
)
@click.option(
    "--allow-remote",
    is_flag=True,
    help="Permit a non-loopback --host; anyone who can reach the port and has the URL can read the data.",
)
@click.option(
    "--port",
    type=click.IntRange(min=0, max=65535),
    default=0,
    show_default=True,
    help="Port to listen on; 0 picks a free port.",
)
@click.option("--no-browser", is_flag=True, help="Do not open a web browser (for remote sessions).")
@click.option(
    "--variant",
    default="b32",
    show_default=True,
    help="Protocol variant whose outputs are shown.",
)
@click.option(
    "--cache-mb",
    type=click.IntRange(min=16),
    default=512,
    show_default=True,
    help="Memory limit for volumes converted for display.",
)
@click.option(
    "--verify-integrity",
    is_flag=True,
    help="Check the SHA-256 fingerprints of a region's files each time it is opened.",
)
@click.option(
    "--blind",
    is_flag=True,
    help="Replace subject, session and region identifiers with pseudonyms and disable downloads.",
)
def view_command(
    manifest_path: Path,
    host: str,
    allow_remote: bool,
    port: int,
    no_browser: bool,
    variant: str,
    cache_mb: int,
    verify_integrity: bool,
    blind: bool,
) -> None:
    """Open a read-only browser viewer for an existing output tree.

    The viewer shows images, region masks, voxelwise feature maps, habitat masks,
    feature tables, QC and provenance. It never writes to the data or output folders.
    It needs the optional viewer extra (Flask).
    """
    try:
        import flask  # noqa: F401
    except ImportError:
        raise click.ClickException(
            'the viewer needs Flask; install the viewer extra: python -m pip install "flask>=3.1,<3.2"'
        ) from None
    from radiomicviz.viewer.catalog import CatalogError
    from radiomicviz.viewer.server import ViewerStartError, run_server

    try:
        run_server(
            manifest_path,
            host=host,
            port=port,
            open_browser=not no_browser,
            variant=variant,
            cache_mb=cache_mb,
            verify_integrity=verify_integrity,
            blind=blind,
            allow_remote=allow_remote,
        )
    except (CatalogError, ViewerStartError) as exc:
        raise click.ClickException(str(exc)) from None


def _echo_progress(message: str) -> None:
    """Write a progress line to stderr so stdout summaries stay unchanged."""
    click.echo(message, err=True)


def _print_plan_totals(units) -> None:
    """Print compact status totals after a potentially long dry-run listing."""
    statuses: dict[str, int] = {}
    for unit in units:
        statuses[unit.status] = statuses.get(unit.status, 0) + 1
    rendered = ", ".join(
        f"{status}={count}" for status, count in sorted(statuses.items())
    )
    click.echo(f"  plan totals: {rendered or 'no units'}")


def _format_duration(seconds: float | None) -> str:
    """Format elapsed seconds for a compact CLI report."""
    if seconds is None:
        return "not available"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _write(name: str, output: Path, force: bool) -> None:
    try:
        written = write_template(name, output, force=force)
    except FileExistsError:
        raise click.ClickException(
            f"{output} already exists. Use --force to replace it; the existing file will be kept "
            "as a timestamped backup."
        ) from None
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise click.ClickException(str(exc)) from None
    if written.backup is not None:
        click.echo(f"Moved existing {written.path} to {written.backup}")
    click.echo(f"Wrote {name} template to {written.path}")
