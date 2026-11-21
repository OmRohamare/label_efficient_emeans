"""
features.py
===========
DINOv2 ViT-S/14 CLS-token feature extraction with:
  - L2 normalisation
  - .npy disk cache with resume support
  - linear_probe_ceiling() helper
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HuggingFace hub model ID for DINOv2 ViT-S/14
_DINOV2_MODEL_ID = "facebook/dinov2-small"
_FEATURE_DIM = 384  # CLS token dimensionality for ViT-S/14


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov2(device: torch.device | str = "cpu") -> nn.Module:
    """
    Load DINOv2 ViT-S/14 from HuggingFace Transformers and return in eval mode.

    Parameters
    ----------
    device:
        Target device.

    Returns
    -------
    ``transformers.Dinov2Model`` in eval mode on ``device``.
    """
    from transformers import Dinov2Model  # type: ignore[import]

    logger.info("Loading DINOv2 ViT-S/14 (%s) …", _DINOV2_MODEL_ID)
    model = Dinov2Model.from_pretrained(_DINOV2_MODEL_ID)
    model = model.to(device).eval()
    # Freeze all parameters – inference only
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device | str,
    l2_normalize: bool = True,
    desc: str = "Extracting features",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS-token features from ``model`` over ``loader``.

    Parameters
    ----------
    loader:
        DataLoader yielding ``(images, labels)`` batches.
    model:
        DINOv2 model (or any model whose forward returns a
        ``transformers.BaseModelOutputWithPooling``).
    device:
        Compute device.
    l2_normalize:
        If True, L2-normalise each feature vector to unit norm.
    desc:
        tqdm progress bar description.

    Returns
    -------
    feats: ``np.ndarray`` of shape ``(N, D)`` float32.
    labels: ``np.ndarray`` of shape ``(N,)`` int64.
    """
    device = torch.device(device)
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(pixel_values=images)
        # CLS token is at index 0 of the last hidden state
        cls_tokens: torch.Tensor = outputs.last_hidden_state[:, 0, :]  # (B, D)

        if l2_normalize:
            cls_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)

        all_feats.append(cls_tokens.cpu().float().numpy())
        all_labels.append(labels.numpy().astype(np.int64))

    feats = np.concatenate(all_feats, axis=0)    # (N, D)
    labels_arr = np.concatenate(all_labels, axis=0)  # (N,)
    return feats, labels_arr


# ---------------------------------------------------------------------------
# Cache-aware extraction
# ---------------------------------------------------------------------------

def extract_and_cache(
    loader: DataLoader,
    cache_path: Path,
    labels_cache_path: Path,
    model: nn.Module | None = None,
    device: torch.device | str = "cpu",
    l2_normalize: bool = True,
    force: bool = False,
    desc: str = "Extracting features",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and cache them to disk as ``.npy`` files.

    If ``cache_path`` and ``labels_cache_path`` already exist and
    ``force=False``, loads and returns cached arrays immediately
    (resume support).

    Parameters
    ----------
    loader:
        DataLoader yielding ``(images, labels)`` batches.
    cache_path:
        Path for the feature array ``(N, D)`` float32.
    labels_cache_path:
        Path for the labels array ``(N,)`` int64.
    model:
        DINOv2 model. Required unless cached files already exist.
    device:
        Compute device.
    l2_normalize:
        L2-normalise features after extraction.
    force:
        Recompute even if cache files exist.
    desc:
        tqdm description.

    Returns
    -------
    feats, labels: numpy arrays.
    """
    cache_path = Path(cache_path)
    labels_cache_path = Path(labels_cache_path)

    if not force and cache_path.exists() and labels_cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        feats = np.load(cache_path, mmap_mode="r")
        labels = np.load(labels_cache_path)
        logger.info("Cache hit: feats %s, labels %s", feats.shape, labels.shape)
        return feats, labels

    if model is None:
        raise ValueError("model must be provided when cache does not exist.")

    logger.info("Extracting features → %s", cache_path)
    feats, labels = extract_features(loader, model, device, l2_normalize=l2_normalize, desc=desc)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic save via temp file + rename
    _save_npy_atomic(feats, cache_path)
    _save_npy_atomic(labels, labels_cache_path)
    logger.info("Features saved: feats %s, labels %s", feats.shape, labels.shape)
    return feats, labels


def _save_npy_atomic(arr: np.ndarray, dst: Path) -> None:
    """Save numpy array atomically via sibling temp file."""
    import os
    import tempfile

    dst = Path(dst)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, arr)
        Path(tmp).replace(dst)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Linear probe ceiling
# ---------------------------------------------------------------------------

def linear_probe_ceiling(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    valid_feats: np.ndarray,
    valid_labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 0.1,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Fit a linear probe (logistic regression) on **all** training features
    to establish the accuracy ceiling for DINOv2 representations.

    Parameters
    ----------
    train_feats:
        Training features ``(N_train, D)`` – should be L2-normalised.
    train_labels:
        Integer labels ``(N_train,)``.
    valid_feats:
        Validation features ``(N_valid, D)``.
    valid_labels:
        Validation labels ``(N_valid,)``.
    max_iter:
        Maximum solver iterations.
    C:
        Inverse regularisation strength for logistic regression.
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``"train_acc"``, ``"valid_acc"``, ``"n_train"``, ``"n_valid"``.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore[import]
    from sklearn.preprocessing import normalize

    logger.info(
        "Fitting linear probe ceiling on %d samples, %d classes …",
        len(train_feats),
        len(np.unique(train_labels)),
    )
    # Ensure L2-normalised (idempotent if already unit norm)
    X_train = normalize(train_feats.astype(np.float64), norm="l2")
    X_valid = normalize(valid_feats.astype(np.float64), norm="l2")

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    train_acc = float(clf.score(X_train, train_labels))
    valid_acc = float(clf.score(X_valid, valid_labels))

    logger.info("Ceiling – train acc: %.4f, valid acc: %.4f", train_acc, valid_acc)
    return {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "n_train": len(train_feats),
        "n_valid": len(valid_feats),
    }
