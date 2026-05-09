"""Clean AIDE inference pipeline for AI-generated image detection.

Reproduces the evaluation logic of ``engine_finetune.evaluate()`` and
``main_finetune.py`` (eval-only mode) while stripping out all training
artifacts (DDP, Mixup, EMA, optimizer).  Outputs per-image fake-probability
scores suitable for downstream analysis.

Equivalent original command::

    python main_finetune.py \\
        --data_path dataset/progan/train \\
        --eval_data_path dataset/progan/eval \\
        --resume results/progan_train/GenImage_train.pth \\
        --eval True --output_dir results/progan_train
"""

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from aide import AIDE
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffguard.dct import DCTBaseRecModule

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

_TRANSFORM_TEST = transforms.Compose([transforms.ToTensor()])
_TRANSFORM_NORMALIZE = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AIDEDataset(Dataset):
    """Flat-folder image dataset compatible with AIDE's 5-stream input.

    Unlike the original AIDE which expects ``0_real/`` / ``1_fake/`` directory
    structure, this dataset accepts an explicit list of *(image_path, label)*
    tuples.  Each sample is preprocessed identically to ``TestDataset`` in the
    upstream codebase: DCT decomposition (4 frequency views) + normalised
    original image, stacked as ``[5, 3, 256, 256]``.
    """

    def __init__(
        self,
        samples: list[tuple[str, int | None]],
    ) -> None:
        self.samples = samples
        self.dct = DCTBaseRecModule()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            logger.warning("Failed to load image, skipping: %s", image_path)
            return self.__getitem__((index + 1) % len(self.samples))

        image = _TRANSFORM_TEST(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except Exception:
            logger.warning("DCT failed for image, skipping: %s", image_path)
            return self.__getitem__((index + 1) % len(self.samples))

        x_0 = _TRANSFORM_NORMALIZE(image)
        x_minmin = _TRANSFORM_NORMALIZE(x_minmin)
        x_maxmax = _TRANSFORM_NORMALIZE(x_maxmax)
        x_minmin1 = _TRANSFORM_NORMALIZE(x_minmin1)
        x_maxmax1 = _TRANSFORM_NORMALIZE(x_maxmax1)

        tensor_label = torch.tensor(label) if label is not None else torch.tensor(-1)
        img_name = Path(image_path).name
        return (
            torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0),
            tensor_label,
            img_name,
        )


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove ``module.`` prefix added by DistributedDataParallel."""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v
    return cleaned


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load a trained AIDE checkpoint, handling DDP-wrapped keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("model", ckpt.get("model_ema", ckpt))
    state_dict = _strip_module_prefix(raw)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("Missing keys (expected for frozen ConvNeXt): %d", len(missing))
    if unexpected:
        logger.warning("Unexpected checkpoint keys: %s", unexpected[:5])


class AIDEInferenceRunner:
    """Single-GPU inference runner for AIDE.

    Initialises the model once and exposes :meth:`run` to score any
    :class:`AIDEDataset`.  All VRAM-saving techniques are enabled by default:
    ``torch.inference_mode``, ``model.eval``, and optional ``bfloat16`` autocast.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 4,
        use_bf16: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size

        if use_bf16 is None:
            self.use_bf16 = (
                self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            )
        else:
            self.use_bf16 = use_bf16

        logger.info("Initialising AIDE model (bf16=%s)...", self.use_bf16)
        model = AIDE(resnet_path=None, convnext_path=None)
        _load_checkpoint(model, checkpoint_path)
        model.to(self.device).eval()
        self.model = model
        logger.info("Model ready on %s", self.device)

    @torch.inference_mode()
    def run(self, dataset: AIDEDataset) -> pd.DataFrame:
        """Score every image in *dataset* and return per-image results.

        Returns:
            DataFrame with columns ``image_name``, ``score``, ``label``.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        results: list[dict] = []
        for images, labels, names in loader:
            images = images.to(self.device, non_blocking=True)

            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(images)
            else:
                logits = self.model(images)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            labels_list = labels.tolist()
            names_list = list(names)

            for name, score, label in zip(names_list, probs, labels_list):
                results.append(
                    {"image_name": name, "score": score, "label": int(label)}
                )

            # Free GPU memory eagerly
            del images, logits
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return pd.DataFrame(results)
