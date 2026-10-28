"""Command-line interface for RadiomicsViz (program name ``radiomicviz``)."""

from __future__ import annotations

from pathlib import Path

import click

from radiomicviz import __version__
from radiomicviz.config import ConfigError, load_project_config
from radiomicviz.config.templates import write_template
from radiomicviz.lesions import (
    MaterializationError,
    build_materialization_plan,
    run_materialization,
)
from radiomicviz.manifest import build_manifest_preview, write_manifest_outputs
from radiomicviz.protocols import ProtocolError, load_protocol
from radiomicviz.whole import (
    WholePlanError,
    WholeStageError,
    load_whole_plan,
    run_whole,
)

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

    ``labeled_map`` preserves each positive integer label as one lesion.
    ``connected_components`` splits a binary foreground using the connectivity stated
    in project.yaml. Touching foreground voxels remain one lesion.
    """
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
    click.echo(f"  output root: {plan.output_root}")
    click.echo(f"  subject/session units: {len(plan.units)}")
    for unit in result.units:
        detail = f" ({unit.message})" if unit.message else ""
        click.echo(
            f"  {unit.status:<29} {unit.subject_id}/{unit.session_id}: "
            f"{unit.n_regions} lesion(s) -> {unit.output_dir}{detail}"
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
    with ``radiomicviz lesions materialize``. The default is a read-only preview. Review
    it, then repeat with --write to create the canonical manifest, input checks, errors
    table, run metadata and one concise log. Rerun with --write after adding subjects;
    completed radiomics units can still resume when their inputs are unchanged.
    """
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
    except OSError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"Wrote input checks to {result.checks_path}")
    click.echo(f"Wrote errors to {result.errors_path}")
    click.echo(f"Wrote run metadata to {result.metadata_path}")
    click.echo(f"Wrote log to {result.log_path}")
    if result.manifest_path is not None:
        click.echo(f"Wrote canonical manifest to {result.manifest_path}")
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
    click.echo(f"  reference status: {preview.reference_status}")
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
    is_flag=True,
    help="Run with non-reference software versions and stamp the deviation in every report.",
)
def whole_command(
    manifest_path: Path,
    protocol_id: str,
    variant: str,
    jobs: int,
    dry_run: bool,
    force: bool,
    allow_version_mismatch: bool,
) -> None:
    """Extract reference whole-region PyRadiomics features.

    Every manifest region and modality is one resumable unit. Each manifest image is
    assumed to be preprocessed, co-registered with its mask and analysis-ready.
    """
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
        click.echo("Dry run: no extraction performed and no files written.")
        if result.blocked:
            raise click.ClickException(
                f"{result.blocked} unit(s) have outputs without a matching completion "
                "record; rerun with --force to archive and recompute them"
            )
        return

    click.echo(f"  completed: {result.succeeded}")
    click.echo(f"  resumed: {result.resumed}")
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


def _echo_progress(message: str) -> None:
    """Write a progress line to stderr so stdout summaries stay unchanged."""
    click.echo(message, err=True)


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
