"""CLI for creating gold-layer post-inference analysis artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from diffguard.analysis import write_analysis_artifacts

app = typer.Typer(add_completion=False)


@app.command()
def main(repo_root: Path = typer.Option(Path("."), help="Repository root.")) -> None:
    """Build merged CSVs, metrics, plots, and failure-case thumbnails."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = write_analysis_artifacts(repo_root.resolve())
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
