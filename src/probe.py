"""
probe.py
========
Downstream evaluation probes for label-efficient learning.

Probes
------
LogisticProbe   – L-BFGS logistic regression (PyTorch).
CosineKNN       – Cosine-similarity k-NN classifier (k=20, batched).

Public API
----------
LogisticProbe      – class with fit(feats, labels) / predict(feats) / score(feats, labels)
CosineKNN          – class with fit(feats, labels) / predict(feats) / score(feats, labels)
evaluate_selection – convenience wrapper: fit on selected, evaluate on validation set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LogisticProbe (PyTorch L-BFGS)
# ---------------------------------------------------------------------------

class LogisticProbe:
    """
    Multinomial logistic regression trained with L-BFGS.

    Uses PyTorch so it benefits from GPU when available.

    Parameters
    ----------
    n_classes:
        Number of output classes.
    max_iter:
        Maximum L-BFGS iterations (outer).
    lr:
        L-BFGS step size.
    C:
        Inverse regularisation strength (L2 penalty = 1/(2*C)).
    device:
        Torch device for training (default: auto).
    """

    def __init__(
        self,
        n_classes: int,
        max_iter: int = 200,
        lr: float = 1.0,
        C: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.lr = lr
        self.C = C
        self._device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        self._model: nn.Linear | None = None

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> "LogisticProbe":
        """
        Fit the logistic regression probe.

        Parameters
        ----------
        feats:
            ``(N, D)`` float32 feature matrix (should be L2-normalised).
        labels:
            ``(N,)`` integer label array.

        Returns
        -------
        self
        """
        feats = np.asarray(feats, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        n, d = feats.shape

        X = torch.from_numpy(feats).to(self._device)
        y = torch.from_numpy(labels).to(self._device)

        model = nn.Linear(d, self.n_classes, bias=True).to(self._device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)

        criterion = nn.CrossEntropyLoss()
        weight_decay = 1.0 / (2.0 * self.C * n)

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            # L2 regularisation
            l2 = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + weight_decay * l2
            loss.backward()
            return loss

        optimizer.step(closure)
        self._model = model
        return self

    @torch.no_grad()
    def predict(self, feats: np.ndarray) -> np.ndarray:
        """
        Predict class labels for ``feats``.

        Parameters
        ----------
        feats:
            ``(M, D)`` float32 feature matrix.

        Returns
        -------
        ``(M,)`` int64 predicted labels.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        feats_t = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self._device)
        logits = self._model(feats_t)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def score(self, feats: np.ndarray, labels: np.ndarray) -> float:
        """Return accuracy on ``(feats, labels)``."""
        preds = self.predict(feats)
        return float(np.mean(preds == np.asarray(labels, dtype=np.int64)))


# ---------------------------------------------------------------------------
# CosineKNN
# ---------------------------------------------------------------------------

class CosineKNN:
    """
    Cosine-similarity k-nearest-neighbour classifier.

    Assumes features are L2-normalised; cosine similarity reduces to dot product.

    Parameters
    ----------
    k:
        Number of neighbours (default 20).
    block_size:
        Query batch size for memory-bounded dot-product computation.
    """

    def __init__(self, k: int = 20, block_size: int = 2048) -> None:
        self.k = k
        self.block_size = block_size
        self._train_feats: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> "CosineKNN":
        """
        Store the training reference set.

        Parameters
        ----------
        feats:
            ``(N, D)`` float32 L2-normalised features.
        labels:
            ``(N,)`` int64 labels.

        Returns
        -------
        self
        """
        self._train_feats = np.asarray(feats, dtype=np.float32)
        self._train_labels = np.asarray(labels, dtype=np.int64)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """
        Predict class labels via cosine-similarity k-NN.

        Parameters
        ----------
        feats:
            ``(M, D)`` float32 L2-normalised query features.

        Returns
        -------
        ``(M,)`` int64 predicted labels.
        """
        if self._train_feats is None:
            raise RuntimeError("Call fit() before predict().")

        feats = np.asarray(feats, dtype=np.float32)
        m = len(feats)
        preds = np.empty(m, dtype=np.int64)
        n_classes = int(self._train_labels.max()) + 1

        for start in range(0, m, self.block_size):
            end = min(start + self.block_size, m)
            q = feats[start:end]  # (B, D)
            # Cosine similarity ≡ dot product for L2-normalised vectors
            sims = q @ self._train_feats.T  # (B, N)
            # Top-k indices per query
            if self.k == 1:
                nn_idx = np.argmax(sims, axis=1)[:, None]
            else:
                # Partial sort: take top-k indices (not fully sorted, but enough)
                nn_idx = np.argpartition(sims, -self.k, axis=1)[:, -self.k:]  # (B, k)

            # Majority vote
            nn_labels = self._train_labels[nn_idx]  # (B, k)
            for b in range(end - start):
                votes = np.bincount(nn_labels[b], minlength=n_classes)
                preds[start + b] = int(np.argmax(votes))

        return preds

    def score(self, feats: np.ndarray, labels: np.ndarray) -> float:
        """Return accuracy on ``(feats, labels)``."""
        preds = self.predict(feats)
        return float(np.mean(preds == np.asarray(labels, dtype=np.int64)))


# ---------------------------------------------------------------------------
# Evaluation convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Results from :func:`evaluate_selection`."""

    logistic_train_acc: float
    logistic_valid_acc: float
    cosine_knn_valid_acc: float
    n_selected: int
    method: str


def evaluate_selection(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    valid_feats: np.ndarray,
    valid_labels: np.ndarray,
    selected_indices: np.ndarray,
    method: str = "unknown",
    n_classes: int = 200,
    knn_k: int = 20,
    lbfgs_max_iter: int = 200,
    device: torch.device | str | None = None,
) -> ProbeResult:
    """
    Fit both probes on the selected subset and evaluate on validation.

    Parameters
    ----------
    train_feats:
        Full training feature matrix ``(N_train, D)``.
    train_labels:
        Full training label array ``(N_train,)``.
    valid_feats:
        Validation feature matrix ``(N_valid, D)``.
    valid_labels:
        Validation label array ``(N_valid,)``.
    selected_indices:
        Indices into ``train_feats`` / ``train_labels`` to use as the
        labelled subset (budget B samples).
    method:
        Name of the selection strategy (for logging).
    n_classes:
        Number of output classes.
    knn_k:
        Number of nearest neighbours for :class:`CosineKNN`.
    lbfgs_max_iter:
        Max L-BFGS iterations for :class:`LogisticProbe`.
    device:
        Torch device for :class:`LogisticProbe`.

    Returns
    -------
    :class:`ProbeResult`
    """
    sel_feats = train_feats[selected_indices]
    sel_labels = train_labels[selected_indices]
    B = len(selected_indices)

    logger.info(
        "Evaluating selection (method=%s, B=%d) with LogisticProbe + CosineKNN …",
        method, B,
    )

    # Logistic probe
    logistic = LogisticProbe(n_classes=n_classes, max_iter=lbfgs_max_iter, device=device)
    logistic.fit(sel_feats, sel_labels)
    log_train_acc = logistic.score(sel_feats, sel_labels)
    log_valid_acc = logistic.score(valid_feats, valid_labels)

    # Cosine k-NN
    knn = CosineKNN(k=knn_k)
    knn.fit(sel_feats, sel_labels)
    knn_valid_acc = knn.score(valid_feats, valid_labels)

    logger.info(
        "LogisticProbe train=%.4f valid=%.4f | CosineKNN valid=%.4f",
        log_train_acc, log_valid_acc, knn_valid_acc,
    )

    return ProbeResult(
        logistic_train_acc=log_train_acc,
        logistic_valid_acc=log_valid_acc,
        cosine_knn_valid_acc=knn_valid_acc,
        n_selected=B,
        method=method,
    )
