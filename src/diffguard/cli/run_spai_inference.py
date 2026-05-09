"""SPAI inference runner — executes NTIRE + ZIT inference and saves all artifacts.

Designed to run as a standalone script via `uv run` for long-running inference.
Outputs match what the notebook cells produce.
"""

import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import filetype
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from spai.config import get_config
from spai.data import data_finetune

# Albumentations 2.x removed ImageCompressionType; SPAI still references it
data_finetune.ImageCompressionType = type(
    "ImageCompressionType", (), {"JPEG": "jpeg", "WEBP": "webp"}
)
from spai.data.data_finetune import build_loader_test
from spai.models import build_cls_model
from spai.utils import load_pretrained

# --- Config ---
SEED = 42
BATCH_SIZE = 1
FEATURE_EXTRACTION_BATCH = 16
REPO_ROOT = Path(".")
CHECKPOINT = REPO_ROOT / "data" / "artifacts" / "checkpoints" / "spai.pth"
SPAI_CONFIG = REPO_ROOT / "data" / "submodules" / "spai" / "configs" / "spai.yaml"
NTIRE_DIR = REPO_ROOT / "data" / "bronze" / "ntire" / "test_images"
NTIRE_LABELS = REPO_ROOT / "data" / "bronze" / "ntire" / "test_labels.csv"
ZIT_DIR = REPO_ROOT / "data" / "bronze" / "z_image_turbo"
ZIT_METADATA = REPO_ROOT / "data" / "bronze" / "z_image_turbo" / "metadata.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "silver" / "spai"
NTIRE_MANIFEST = OUTPUT_DIR / "ntire_manifest.csv"
ZIT_MANIFEST = OUTPUT_DIR / "zit_manifest.csv"

# --- Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("spai_inference")


@torch.no_grad()
def run_spai_inference(model, data_loader, config, progress_every=100):
    """Run SPAI inference, returning {dataset_idx: score}."""
    model.eval()
    scores = {}
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            images = [img.cuda(non_blocking=True) for img in images]
            images = [img.squeeze(dim=1) for img in images]
            output = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH)
        else:
            images = images.cuda(non_blocking=True).squeeze(dim=1)
            output = model(images)

        output = torch.sigmoid(output)
        batch_scores = output.squeeze(dim=1).cpu().tolist()
        batch_indices = dataset_idx.cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
            batch_indices = [batch_indices]

        for i, s in zip(batch_indices, batch_scores):
            scores[i] = s

        if (idx + 1) % progress_every == 0:
            vram = torch.cuda.max_memory_allocated() / 1e6
            logger.info("Batch %d/%d | VRAM: %.0f MB", idx + 1, len(data_loader), vram)

    return scores


def scores_to_dataframe(scores, manifest):
    """Merge inference scores with manifest."""
    results = manifest.copy()
    score_values = [scores.get(i, float("nan")) for i in range(len(manifest))]
    results["score"] = score_values
    results["prediction"] = (results["score"] >= 0.5).astype(int)
    return results


def compute_metrics(y_true, y_score, threshold=0.5):
    """Binary classification metrics."""
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "threshold": threshold,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("Checkpoint: %s (%.0f MB)", CHECKPOINT, CHECKPOINT.stat().st_size / 1e6)
    logger.info("CUDA: %s", torch.cuda.is_available())

    # --- Build manifests ---
    labels = pd.read_csv(NTIRE_LABELS)
    manifest_ntire = pd.DataFrame(
        {
            "image": labels["image_name"],
            "split": "test",
            "class": labels["label"].astype(str),
            "label": labels["label"],
            "source": "ntire",
        }
    )
    manifest_ntire.to_csv(NTIRE_MANIFEST, index=False)
    logger.info("NTIRE manifest: %d images", len(manifest_ntire))

    image_files = sorted(
        [f.name for f in ZIT_DIR.iterdir() if f.is_file() and filetype.is_image(f)]
    )
    manifest_zit = pd.DataFrame(
        {
            "image": image_files,
            "split": "test",
            "class": "1",
            "label": 1,
            "source": "z_image_turbo",
        }
    )
    manifest_zit.to_csv(ZIT_MANIFEST, index=False)
    logger.info("ZIT manifest: %d images", len(manifest_zit))

    # --- Init model ---
    config = get_config(
        {
            "cfg": str(SPAI_CONFIG),
            "batch_size": BATCH_SIZE,
            "test_csv": [str(NTIRE_MANIFEST)],
            "test_csv_root": [str(NTIRE_DIR)],
            "pretrained": str(CHECKPOINT),
            "output": str(OUTPUT_DIR),
            "tag": "spai",
            "opts": [("MODEL.FEATURE_EXTRACTION_BATCH", str(FEATURE_EXTRACTION_BATCH))],
        }
    )

    logger.info("Building model...")
    model = build_cls_model(config)
    model.cuda()
    load_pretrained(config, model, logger, checkpoint_path=CHECKPOINT, verbose=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vram = torch.cuda.memory_allocated() / 1e6
    logger.info("Model ready: %d params, %.0f MB VRAM", n_params, vram)

    # --- NTIRE inference ---
    logger.info("=== NTIRE Inference ===")
    ntire_config = get_config(
        {
            "cfg": str(SPAI_CONFIG),
            "batch_size": BATCH_SIZE,
            "test_csv": [str(NTIRE_MANIFEST)],
            "test_csv_root": [str(NTIRE_DIR)],
            "pretrained": str(CHECKPOINT),
            "output": str(OUTPUT_DIR),
            "tag": "spai_ntire",
            "opts": [("MODEL.FEATURE_EXTRACTION_BATCH", str(FEATURE_EXTRACTION_BATCH))],
        }
    )
    _, ntire_datasets, ntire_loaders = build_loader_test(
        ntire_config, logger, split="test"
    )
    logger.info(
        "NTIRE: %d images in %d batches", len(ntire_datasets[0]), len(ntire_loaders[0])
    )

    t0 = time.time()
    ntire_scores = run_spai_inference(
        model, ntire_loaders[0], ntire_config, progress_every=100
    )
    elapsed = time.time() - t0
    logger.info("NTIRE done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    results_ntire = scores_to_dataframe(ntire_scores, manifest_ntire)
    results_ntire.to_csv(OUTPUT_DIR / "ntire_scores.csv", index=False)
    logger.info("Saved ntire_scores.csv (%d rows)", len(results_ntire))

    y_true = results_ntire["label"].values
    y_score = results_ntire["score"].values
    metrics = compute_metrics(y_true, y_score)
    with open(OUTPUT_DIR / "ntire_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "NTIRE metrics: ACC=%.4f AP=%.4f AUC=%.4f F1=%.4f",
        metrics["accuracy"],
        metrics["average_precision"],
        metrics["auc_roc"],
        metrics["f1"],
    )

    # Clear VRAM cache between datasets
    torch.cuda.empty_cache()
    del ntire_loaders, ntire_datasets
    logger.info("VRAM after cleanup: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    # --- ZIT inference ---
    logger.info("=== Z-Image-Turbo Inference ===")
    zit_config = get_config(
        {
            "cfg": str(SPAI_CONFIG),
            "batch_size": BATCH_SIZE,
            "test_csv": [str(ZIT_MANIFEST)],
            "test_csv_root": [str(ZIT_DIR)],
            "pretrained": str(CHECKPOINT),
            "output": str(OUTPUT_DIR),
            "tag": "spai_zit",
            "opts": [("MODEL.FEATURE_EXTRACTION_BATCH", str(FEATURE_EXTRACTION_BATCH))],
        }
    )
    _, zit_datasets, zit_loaders = build_loader_test(zit_config, logger, split="test")
    logger.info(
        "ZIT: %d images in %d batches", len(zit_datasets[0]), len(zit_loaders[0])
    )

    t0 = time.time()
    zit_scores = run_spai_inference(
        model, zit_loaders[0], zit_config, progress_every=25
    )
    elapsed = time.time() - t0
    logger.info("ZIT done in %.1fs", elapsed)

    results_zit = scores_to_dataframe(zit_scores, manifest_zit)
    results_zit.to_csv(OUTPUT_DIR / "z_image_turbo_scores.csv", index=False)
    logger.info("Saved z_image_turbo_scores.csv (%d rows)", len(results_zit))
    logger.info(
        "ZIT: mean=%.4f median=%.4f >0.5: %d/%d",
        results_zit["score"].mean(),
        results_zit["score"].median(),
        (results_zit["score"] > 0.5).sum(),
        len(results_zit),
    )

    # --- Reproducibility snapshot ---
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = git_branch = "unknown"

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_size_mb": round(CHECKPOINT.stat().st_size / 1e6, 1),
        "spai_config_path": str(SPAI_CONFIG),
        "spai_config": config.dump(),
        "model_type": config.MODEL.TYPE,
        "sid_approach": config.MODEL.SID_APPROACH,
        "resolution_mode": config.MODEL.RESOLUTION_MODE,
        "required_normalization": config.MODEL.REQUIRED_NORMALIZATION,
        "feature_extraction_batch": FEATURE_EXTRACTION_BATCH,
        "batch_size": BATCH_SIZE,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "ntire_metrics": metrics,
        "zit_mean_score": float(results_zit["score"].mean()),
        "input_paths": {
            "ntire_manifest": str(NTIRE_MANIFEST),
            "ntire_images": str(NTIRE_DIR),
            "zit_manifest": str(ZIT_MANIFEST),
            "zit_images": str(ZIT_DIR),
        },
        "output_path": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    logger.info("Saved experiment_config.json")

    logger.info("=== ALL DONE ===")


if __name__ == "__main__":
    main()
