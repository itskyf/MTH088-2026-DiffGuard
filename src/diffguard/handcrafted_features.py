"""Handcrafted image features for post-inference fake-image analysis.

This module computes lightweight, interpretable per-image descriptors for
streaming datasets.  The returned values are intended to complement detector
scores in analysis tables, not to replace AIDE/SPAI inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
from scipy.fftpack import dct

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def _load_grayscale(image_path: Path, max_side: int = 512) -> np.ndarray:
    """Load one image as a grayscale float array with bounded working size.

    Args:
        image_path: Image file to read.
        max_side: Maximum side length used for frequency/edge features.

    Returns:
        A two-dimensional float32 array in the range ``[0, 255]``.
    """
    with Image.open(image_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            resample = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
            gray = gray.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                resample=resample,
            )
        return np.asarray(gray, dtype=np.float32)


def _entropy(gray: np.ndarray) -> float:
    """Compute Shannon entropy over 8-bit grayscale intensities."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=False)
    prob = hist.astype(np.float64)
    prob = prob[prob > 0] / prob.sum()
    return float(-(prob * np.log2(prob)).sum())


def _dct_high_frequency_ratio(gray: np.ndarray) -> float:
    """Estimate the share of DCT energy in high-frequency coefficients."""
    centered = gray - float(gray.mean())
    coeff = dct(dct(centered, axis=0, norm="ortho"), axis=1, norm="ortho")
    energy = np.square(coeff, dtype=np.float64)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0

    rows, cols = energy.shape
    row_idx, col_idx = np.ogrid[:rows, :cols]
    high_mask = (row_idx / max(rows - 1, 1) + col_idx / max(cols - 1, 1)) >= 0.75
    return float(energy[high_mask].sum() / total)


def compute_image_features(
    image_path: Path, source: str, label: int
) -> dict[str, object]:
    """Compute one row of handcrafted descriptors for an image.

    Args:
        image_path: Image file to process.
        source: Dataset source name used by downstream merges.
        label: Binary fake-image label where ``1`` means fake.

    Returns:
        A dictionary suitable for CSV serialization.
    """
    gray = _load_grayscale(image_path)
    uint8_gray = np.clip(gray, 0, 255).astype(np.uint8)
    laplacian = cv2.Laplacian(uint8_gray, cv2.CV_64F)
    edges = cv2.Canny(uint8_gray, 100, 200)

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_stem": image_path.stem,
        "source": source,
        "label": int(label),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "entropy": _entropy(gray),
        "laplacian_variance": float(laplacian.var()),
        "dct_high_frequency_ratio": _dct_high_frequency_ratio(gray),
        "brightness_mean": float(gray.mean() / 255.0),
        "brightness_std": float(gray.std() / 255.0),
        "edge_density": float((edges > 0).mean()),
    }


def iter_feature_rows(
    samples: list[tuple[Path, int]],
    source: str,
    progress_every: int = 100,
) -> list[dict[str, object]]:
    """Compute handcrafted features for image samples without bulk image loads.

    Args:
        samples: Image paths paired with binary labels.
        source: Dataset source name for every produced row.
        progress_every: Log cadence in processed images.

    Returns:
        A list of per-image feature dictionaries.
    """
    rows: list[dict[str, object]] = []
    for index, (image_path, label) in enumerate(samples, start=1):
        try:
            rows.append(compute_image_features(image_path, source=source, label=label))
        except Exception:
            logger.exception("Failed to compute features for %s", image_path)
        if index % progress_every == 0:
            logger.info(
                "Computed handcrafted features for %d/%d images", index, len(samples)
            )
    return rows
