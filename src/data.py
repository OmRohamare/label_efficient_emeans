"""
data.py
=======
HuggingFace zh-plus/tiny-imagenet dataset loader with DINOv2-compatible
preprocessing (64→224 bicubic resize + ImageNet normalisation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_dinov2_transform(image_size: int = 224) -> transforms.Compose:
    """
    Build the standard DINOv2 / ImageNet inference transform.

    Pipeline:
      1. Resize shortest side to ``image_size`` with bicubic interpolation.
      2. CenterCrop to ``(image_size, image_size)``.
      3. ToTensor → [0,1].
      4. Normalise with ImageNet mean/std.

    Parameters
    ----------
    image_size:
        Target square side length (default 224 for ViT-S/14).

    Returns
    -------
    ``torchvision.transforms.Compose``
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class TinyImageNetDataset(Dataset):
    """
    Thin PyTorch ``Dataset`` wrapping a HuggingFace ``datasets`` split.

    Parameters
    ----------
    hf_split:
        A HuggingFace ``datasets.Dataset`` object (train or valid split).
    transform:
        Optional image transform applied to each PIL image.
    image_key:
        Column name containing PIL images (default ``"image"``).
    label_key:
        Column name containing integer labels (default ``"label"``).
    """

    def __init__(
        self,
        hf_split: Any,
        transform: Callable | None = None,
        image_key: str = "image",
        label_key: str = "label",
    ) -> None:
        self.hf_split = hf_split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        sample = self.hf_split[idx]
        img: Image.Image = sample[self.image_key]

        # Tiny-ImageNet has some grayscale images – convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label: int = int(sample[self.label_key])
        return img, label


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_tiny_imagenet(
    cache_dir: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Download (or load from cache) zh-plus/tiny-imagenet and return
    ``(train_loader, valid_loader, num_classes)``.

    Parameters
    ----------
    cache_dir:
        HuggingFace cache directory. ``None`` uses the HF default.
    image_size:
        Target image resolution fed to DINOv2 (default 224).
    batch_size:
        DataLoader batch size.
    num_workers:
        DataLoader worker processes.
    pin_memory:
        Pin tensors to host memory (useful with CUDA).

    Returns
    -------
    train_loader, valid_loader, num_classes
    """
    # Lazy import so the module loads even without `datasets` installed at import time
    from datasets import load_dataset  # type: ignore[import]

    logger.info("Loading zh-plus/tiny-imagenet from HuggingFace …")
    ds = load_dataset(
        "zh-plus/tiny-imagenet",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    transform = build_dinov2_transform(image_size)
    train_ds = TinyImageNetDataset(ds["train"], transform=transform)
    valid_ds = TinyImageNetDataset(ds["valid"], transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,          # No shuffling – feature order must match label array
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    num_classes: int = len(set(ds["train"]["label"]))
    logger.info(
        "Dataset loaded: %d train, %d valid, %d classes",
        len(train_ds),
        len(valid_ds),
        num_classes,
    )
    return train_loader, valid_loader, num_classes


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(
    cache_dir: str | Path | None = None,
    expected_train: int = 100_000,
    expected_valid: int = 10_000,
    expected_classes: int = 200,
) -> dict[str, Any]:
    """
    Verify that the downloaded dataset has the expected shape.

    Parameters
    ----------
    cache_dir:
        HuggingFace cache directory.
    expected_train:
        Expected number of training examples.
    expected_valid:
        Expected number of validation examples.
    expected_classes:
        Expected number of classes.

    Returns
    -------
    Dict with verification results and ``"ok": bool``.

    Raises
    ------
    AssertionError
        If any check fails and ``strict=True``.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(
        "zh-plus/tiny-imagenet",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    n_train = len(ds["train"])
    n_valid = len(ds["valid"])
    unique_labels = set(ds["train"]["label"])
    n_classes = len(unique_labels)

    # Sample check: images are PIL, labels are int
    sample = ds["train"][0]
    img_ok = isinstance(sample["image"], Image.Image)
    label_ok = isinstance(sample["label"], int)

    # Check image mode / size
    img: Image.Image = sample["image"]
    native_size = img.size  # (W, H)

    checks = {
        "n_train": n_train,
        "n_valid": n_valid,
        "n_classes": n_classes,
        "native_size": native_size,
        "img_ok": img_ok,
        "label_ok": label_ok,
        "train_match": n_train == expected_train,
        "valid_match": n_valid == expected_valid,
        "classes_match": n_classes == expected_classes,
    }
    checks["ok"] = all(
        [
            checks["train_match"],
            checks["valid_match"],
            checks["classes_match"],
            checks["img_ok"],
            checks["label_ok"],
        ]
    )

    if checks["ok"]:
        logger.info("Dataset verification PASSED: %s", checks)
    else:
        logger.warning("Dataset verification FAILED: %s", checks)

    return checks


def get_labels_array(
    cache_dir: str | Path | None = None,
    split: str = "train",
) -> np.ndarray:
    """
    Return integer label array for a given split without loading images.

    Parameters
    ----------
    cache_dir:
        HuggingFace cache directory.
    split:
        ``"train"`` or ``"valid"``.

    Returns
    -------
    ``np.ndarray`` of shape ``(N,)`` dtype int64.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(
        "zh-plus/tiny-imagenet",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return np.array(ds[split]["label"], dtype=np.int64)
