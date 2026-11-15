"""Command-line interface: ``datacleaner clean data.csv -o out/``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datacleaner import __version__
from datacleaner.core.config import PipelineConfig
from datacleaner.core.errors import DataCleanerError
from datacleaner.core.pipeline import Pipeline
from datacleaner.core.profile import profile_dataframe
from datacleaner.io import FileFormat, read_path, write_path, write_report
from datacleaner.logging import configure_logging

app = typer.Typer(
    help="Clean tabular data and prepare it for machine learning.",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log each step.")] = False,
) -> None:
    configure_logging("INFO" if verbose else "WARNING", json_logs=False)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def clean(
    input_file: Annotated[Path, typer.Argument(help="CSV, XLSX or Parquet file.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Where to write.")] = Path(
        "out"
    ),
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML/JSON pipeline config.")
    ] = None,
    output_format: Annotated[
        FileFormat, typer.Option("--format", "-f", help="Output file format.")
    ] = FileFormat.CSV,
    target: Annotated[str | None, typer.Option(help="Target column (overrides config).")] = None,
    feature_source: Annotated[
        str | None, typer.Option(help="Numeric column to bin (overrides config).")
    ] = None,
) -> None:
    """Run the pipeline; write cleaned data, ML-ready data and a JSON report."""
    try:
        config = PipelineConfig.from_file(config_file) if config_file else PipelineConfig()
        overrides = {
            k: v for k, v in (("target", target), ("feature_source", feature_source)) if v
        }
        config = config.model_copy(update=overrides)

        result = Pipeline(config).run(read_path(input_file))

        ext = output_format.value
        cleaned = write_path(result.cleaned, output_dir / f"cleaned.{ext}")
        ml = write_path(result.ml, output_dir / f"ml_ready.{ext}")
        report = write_report(result.report, output_dir / "report.json")
    except DataCleanerError as exc:
        raise _fail(str(exc)) from exc

    r = result.report
    typer.echo(f"Rows:               {r.before.rows} -> {r.after.rows}")
    typer.echo(f"Missing values:     {r.before.missing_values} -> {r.after.missing_values}")
    typer.echo(f"Duplicates removed: {r.duplicates_removed}")
    typer.echo(f"Outlier columns:    {', '.join(r.outliers) or 'none'}")
    typer.echo(f"Features created:   {', '.join(r.created_features) or 'none'}")

    for warning in r.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)

    typer.echo(f"\nWrote {cleaned}\n      {ml}\n      {report}")


@app.command()
def profile(
    input_file: Annotated[Path, typer.Argument(help="CSV, XLSX or Parquet file.")],
) -> None:
    """Print a quick summary of a raw dataset."""
    try:
        p = profile_dataframe(read_path(input_file))
    except DataCleanerError as exc:
        raise _fail(str(exc)) from exc

    typer.echo(
        f"{p.rows} rows x {p.columns} columns, "
        f"{p.missing_values} missing values, {p.duplicate_rows} duplicate rows"
    )
    typer.echo(f"ID-like columns: {', '.join(p.id_columns) or 'none'}\n")

    for col in p.column_profiles:
        typer.echo(
            f"  {col.name:<20} {col.kind:<8} missing={col.missing:<4} unique={col.unique}"
        )


@app.command("init-config")
def init_config(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path("config.yaml"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite if it exists.")] = False,
) -> None:
    """Write a starter YAML config containing every default."""
    import yaml

    if path.exists() and not force:
        raise _fail(f"{path} already exists (use --force to overwrite).")

    defaults = yaml.safe_dump(PipelineConfig().model_dump(), sort_keys=False)
    path.write_text(defaults, encoding="utf-8")
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
