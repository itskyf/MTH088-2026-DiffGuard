"""CLI for streaming handcrafted feature extraction over bronze images."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from diffguard.handcrafted_features import iter_feature_rows

app = typer.Typer(add_completion=False)


def _configure_logging(log_path: Path) -> None:
    """Configure file and console logging for long-running feature jobs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _ntire_samples(repo_root: Path) -> list[tuple[Path, int]]:
    """Return NTIRE image paths with labels from ``test_labels.csv``."""
    labels = pd.read_csv(repo_root / "data/bronze/ntire/test_labels.csv")
    image_dir = repo_root / "data/bronze/ntire/test_images"
    return [(image_dir / row.image_name, int(row.label)) for row in labels.itertuples()]


def _z_image_turbo_samples(repo_root: Path) -> list[tuple[Path, int]]:
    """Return Z-Image-Turbo PNG paths with fake labels."""
    image_dir = repo_root / "data/bronze/z_image_turbo"
    paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() == ".png"
    )
    return [(path, 1) for path in paths]


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), help="Repository root."),
    output_dir: Path = typer.Option(
        Path("data/silver/handcrafted"), help="Silver output directory."
    ),
    progress_every: int = typer.Option(100, help="Log progress cadence."),
) -> None:
    """Compute handcrafted features for NTIRE and Z-Image-Turbo."""
    repo_root = repo_root.resolve()
    output_dir = (repo_root / output_dir).resolve()
    log_path = output_dir / "logs" / "compute_handcrafted_features.log"
    _configure_logging(log_path)
    logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting handcrafted feature extraction at %s", repo_root)

    datasets = [
        ("ntire", _ntire_samples(repo_root), output_dir / "ntire_features.csv"),
        (
            "z_image_turbo",
            _z_image_turbo_samples(repo_root),
            output_dir / "z_image_turbo_features.csv",
        ),
    ]
    for source, samples, output_path in datasets:
        logger.info("Processing %s with %d images", source, len(samples))
        rows = iter_feature_rows(samples, source=source, progress_every=progress_every)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Wrote %s", output_path)

    config = {
        "features": [
            "entropy",
            "laplacian_variance",
            "dct_high_frequency_ratio",
            "brightness_mean",
            "brightness_std",
            "edge_density",
        ],
        "max_side": 512,
        "edge_density": "cv2.Canny thresholds 100/200",
        "dct_high_frequency_ratio": "2D orthonormal DCT energy where normalized u+v >= 0.75",
        "outputs": [str(path) for _, _, path in datasets],
    }
    with (output_dir / "feature_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    logger.info("Feature extraction complete")


if __name__ == "__main__":
    app()
