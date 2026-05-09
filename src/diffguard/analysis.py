"""Gold-layer analysis artifacts for AIDE/SPAI fake-image detection outputs.

The functions in this module merge detector scores, labels, and handcrafted
features into report-ready CSVs and figures.  They assume inference has already
finished and never run model scoring.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "entropy",
    "laplacian_variance",
    "dct_high_frequency_ratio",
    "brightness_mean",
    "brightness_std",
    "edge_density",
]


def normalize_image_key(value: object) -> str:
    """Normalize filenames/paths to an extension-free merge key."""
    return Path(str(value)).stem


def _read_score_csv(path: Path, score_name: str) -> pd.DataFrame:
    """Load detector scores and standardize key, label, and score columns."""
    frame = pd.read_csv(path)
    name_col = "image_name" if "image_name" in frame.columns else "image"
    if name_col not in frame.columns:
        raise ValueError(f"No image key column found in {path}")
    if "score" not in frame.columns:
        raise ValueError(f"No score column found in {path}")

    out = frame[
        [name_col, "score"] + (["label"] if "label" in frame.columns else [])
    ].copy()
    out["image_key"] = out[name_col].map(normalize_image_key)
    out = out.rename(columns={"score": score_name})
    keep = ["image_key", score_name]
    if "label" in out.columns:
        keep.append("label")
    out = out[keep].drop_duplicates("image_key")
    out[score_name] = out[score_name].astype(float).clip(0.0, 1.0)
    return out


def _classification_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    """Compute standard binary metrics at threshold ``0.5``."""
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def merge_ntire(repo_root: Path) -> pd.DataFrame:
    """Merge NTIRE labels, detector scores, and handcrafted features."""
    labels = pd.read_csv(repo_root / "data/bronze/ntire/test_labels.csv")
    labels = labels[["image_name", "label"]].copy()
    labels["image_key"] = labels["image_name"].map(normalize_image_key)
    labels["image_path"] = labels["image_name"].map(
        lambda name: str(repo_root / "data/bronze/ntire/test_images" / name)
    )
    labels["source"] = "ntire"

    aide = _read_score_csv(
        repo_root / "data/silver/aide/ntire_scores.csv", "aide_score"
    )
    spai = _read_score_csv(
        repo_root / "data/silver/spai/ntire_scores.csv", "spai_score"
    )
    features = pd.read_csv(repo_root / "data/silver/handcrafted/ntire_features.csv")
    features["image_key"] = features["image_name"].map(normalize_image_key)

    merged = labels.merge(aide[["image_key", "aide_score"]], on="image_key", how="left")
    merged = merged.merge(spai[["image_key", "spai_score"]], on="image_key", how="left")
    merged = merged.merge(
        features[["image_key", *FEATURE_COLUMNS, "width", "height"]],
        on="image_key",
        how="left",
    )
    return merged[
        [
            "image_path",
            "label",
            "source",
            "aide_score",
            "spai_score",
            *FEATURE_COLUMNS,
            "width",
            "height",
            "image_key",
        ]
    ]


def merge_z_image_turbo(repo_root: Path) -> pd.DataFrame:
    """Merge Z-Image-Turbo detector scores and handcrafted features."""
    features = pd.read_csv(
        repo_root / "data/silver/handcrafted/z_image_turbo_features.csv"
    )
    features["image_key"] = features["image_name"].map(normalize_image_key)
    features["label"] = 1
    features["source"] = "z_image_turbo"

    aide = _read_score_csv(
        repo_root / "data/silver/aide/z_image_turbo_scores.csv", "aide_score"
    )
    spai = _read_score_csv(
        repo_root / "data/silver/spai/z_image_turbo_scores.csv", "spai_score"
    )

    merged = features.merge(
        aide[["image_key", "aide_score"]], on="image_key", how="left"
    )
    merged = merged.merge(spai[["image_key", "spai_score"]], on="image_key", how="left")
    return merged[
        [
            "image_path",
            "label",
            "source",
            "aide_score",
            "spai_score",
            *FEATURE_COLUMNS,
            "width",
            "height",
            "image_key",
        ]
    ]


def build_metrics_comparison(ntire: pd.DataFrame, zit: pd.DataFrame) -> pd.DataFrame:
    """Build detector metric summary for NTIRE and Z-Image-Turbo."""
    rows: list[dict[str, object]] = []
    for detector, column in [("AIDE", "aide_score"), ("SPAI", "spai_score")]:
        metrics = _classification_metrics(ntire["label"], ntire[column])
        rows.append({"dataset": "ntire", "detector": detector, **metrics})
        zscores = zit[column]
        rows.append(
            {
                "dataset": "z_image_turbo",
                "detector": detector,
                "mean_score": float(zscores.mean()),
                "std_score": float(zscores.std()),
                "median_score": float(zscores.median()),
                "min_score": float(zscores.min()),
                "max_score": float(zscores.max()),
                "score_gt_0_5_rate": float((zscores > 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_score_distribution(ntire: pd.DataFrame, output_path: Path) -> None:
    """Save NTIRE score distributions split by detector and label."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, detector, column in zip(
        axes, ["AIDE", "SPAI"], ["aide_score", "spai_score"]
    ):
        for label, color, name in [(0, "#276FBF", "real"), (1, "#C84630", "fake")]:
            subset = ntire.loc[ntire["label"] == label, column]
            ax.hist(subset, bins=40, density=True, alpha=0.55, color=color, label=name)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(detector)
        ax.set_xlabel("Fake probability")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _save_feature_boxplots(
    ntire: pd.DataFrame, zit: pd.DataFrame, output_path: Path
) -> None:
    """Save handcrafted feature boxplots for NTIRE real/fake and ZIT fake."""
    plot_frame = pd.concat(
        [
            ntire.assign(
                group=np.where(ntire["label"] == 0, "NTIRE real", "NTIRE fake")
            ),
            zit.assign(group="Z-Image-Turbo"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    groups = ["NTIRE real", "NTIRE fake", "Z-Image-Turbo"]
    colors = ["#276FBF", "#C84630", "#2A9D8F"]
    for ax, feature in zip(axes.ravel(), FEATURE_COLUMNS):
        values = [
            plot_frame.loc[plot_frame["group"] == group, feature].dropna()
            for group in groups
        ]
        box = ax.boxplot(values, labels=groups, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _save_roc_comparison(ntire: pd.DataFrame, output_path: Path) -> None:
    """Save ROC curves for both detectors on NTIRE."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for detector, column, color in [
        ("AIDE", "aide_score", "#276FBF"),
        ("SPAI", "spai_score", "#C84630"),
    ]:
        fpr, tpr, _ = roc_curve(ntire["label"], ntire[column])
        auc = roc_auc_score(ntire["label"], ntire[column])
        ax.plot(fpr, tpr, label=f"{detector} AUC={auc:.3f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("NTIRE ROC comparison")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _save_uncertainty_proxy(ntire: pd.DataFrame, output_path: Path) -> None:
    """Save detector disagreement and near-threshold uncertainty proxy."""
    frame = ntire.copy()
    frame["disagreement"] = (frame["aide_score"] - frame["spai_score"]).abs()
    frame["mean_margin_to_0_5"] = (
        (frame["aide_score"] + frame["spai_score"]) / 2 - 0.5
    ).abs()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame["disagreement"], bins=40, color="#6A4C93", alpha=0.75)
    axes[0].set_title("|AIDE - SPAI|")
    axes[0].set_xlabel("Score disagreement")
    axes[0].set_ylabel("Images")
    axes[1].hist(frame["mean_margin_to_0_5"], bins=40, color="#2A9D8F", alpha=0.75)
    axes[1].set_title("Mean-score margin to 0.5")
    axes[1].set_xlabel("Absolute margin")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def select_failure_cases(ntire: pd.DataFrame) -> pd.DataFrame:
    """Select six report cases covering TP, FN, FP, and disagreement behavior."""
    frame = ntire.copy()
    frame["mean_score"] = (frame["aide_score"] + frame["spai_score"]) / 2
    frame["min_score"] = frame[["aide_score", "spai_score"]].min(axis=1)
    frame["max_score"] = frame[["aide_score", "spai_score"]].max(axis=1)
    frame["disagreement"] = (frame["aide_score"] - frame["spai_score"]).abs()
    frame["margin_to_0_5"] = (frame["mean_score"] - 0.5).abs()

    chosen: list[pd.Series] = []

    def add_rows(rows: pd.DataFrame, count: int, case_type: str, reason: str) -> None:
        for _, row in rows.iterrows():
            if len([x for x in chosen if x["case_type"] == case_type]) >= count:
                break
            if row["image_key"] in {x["image_key"] for x in chosen}:
                continue
            item = row.copy()
            item["case_type"] = case_type
            item["reason"] = reason
            chosen.append(item)

    add_rows(
        frame[frame["label"] == 1].sort_values(
            ["min_score", "mean_score"], ascending=False
        ),
        2,
        "true_positive_fake",
        "label=1 and both detector scores are among the highest available",
    )
    add_rows(
        frame[frame["label"] == 1].sort_values(
            ["min_score", "mean_score"], ascending=True
        ),
        2,
        "false_negative_fake",
        "label=1 but at least one detector assigns a low fake probability",
    )
    add_rows(
        frame[frame["label"] == 0].sort_values("max_score", ascending=False),
        1,
        "false_positive_real",
        "label=0 with the highest detector fake probability",
    )
    disagreement = frame.sort_values(
        ["disagreement", "margin_to_0_5"], ascending=[False, True]
    )
    add_rows(
        disagreement,
        1,
        "uncertain_disagreement",
        "large AIDE/SPAI score disagreement or near-threshold mean score",
    )

    result = pd.DataFrame(chosen).head(6).copy()
    result.insert(0, "case_id", [f"case_{i:02d}" for i in range(1, len(result) + 1)])
    return result


def copy_failure_case_images(cases: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Copy compact thumbnails for selected failure/interesting cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for _, row in cases.iterrows():
        source_path = Path(row["image_path"])
        suffix = source_path.suffix.lower() or ".png"
        target_path = output_dir / f"{row['case_id']}_{row['case_type']}{suffix}"
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((512, 512))
            image.save(target_path)
        copied_paths.append(str(target_path))
    cases = cases.copy()
    cases["copied_path"] = copied_paths
    return cases


def write_analysis_artifacts(repo_root: Path) -> dict[str, Path]:
    """Write all required gold-layer analysis artifacts and return their paths."""
    gold_dir = repo_root / "data/gold/analysis"
    plots_dir = gold_dir / "plots"
    cases_dir = gold_dir / "failure_cases"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ntire = merge_ntire(repo_root)
    zit = merge_z_image_turbo(repo_root)
    metrics = build_metrics_comparison(ntire, zit)

    ntire_path = gold_dir / "merged_ntire.csv"
    zit_path = gold_dir / "merged_z_image_turbo.csv"
    metrics_path = gold_dir / "metrics_comparison.csv"
    ntire.to_csv(ntire_path, index=False)
    zit.to_csv(zit_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    _save_score_distribution(ntire, plots_dir / "score_distribution.svg")
    _save_feature_boxplots(ntire, zit, plots_dir / "handcrafted_feature_boxplots.svg")
    _save_roc_comparison(ntire, plots_dir / "roc_comparison.svg")
    _save_uncertainty_proxy(
        ntire, plots_dir / "tta_or_uncertainty_proxy_if_available.svg"
    )

    cases = select_failure_cases(ntire)
    cases = copy_failure_case_images(cases, cases_dir)
    case_columns = [
        "case_id",
        "image_path",
        "copied_path",
        "source",
        "label",
        "aide_score",
        "spai_score",
        "entropy",
        "laplacian_variance",
        "dct_high_frequency_ratio",
        "case_type",
        "reason",
    ]
    cases[case_columns].to_csv(gold_dir / "failure_cases.csv", index=False)

    with (gold_dir / "analysis_config.json").open("w") as handle:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "threshold": 0.5,
                "failure_case_count": int(len(cases)),
                "score_columns": ["aide_score", "spai_score"],
            },
            handle,
            indent=2,
        )

    logger.info("Wrote analysis artifacts to %s", gold_dir)
    return {
        "merged_ntire": ntire_path,
        "merged_z_image_turbo": zit_path,
        "metrics_comparison": metrics_path,
        "failure_cases": gold_dir / "failure_cases.csv",
        "plots": plots_dir,
        "failure_case_images": cases_dir,
    }
