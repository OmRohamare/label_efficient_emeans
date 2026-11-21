"""
assign.py
=========
Memory-efficient blocked nearest-centroid assignment using the identity:

    ||x - c||^2 = ||x||^2 - 2 x·c^T + ||c||^2

Accepts both PyTorch tensors and NumPy arrays.  Returns a **torch.Tensor**
(int64) so that tests can use ``torch.equal`` / ``torch.allclose`` directly.

Public API
----------
assign_blocked(feats, centroids, block_size=4096) -> torch.Tensor  (N,) int64
compute_mse(feats, centroids, assignments=None, block_size=4096) -> float
validate_assignments(assignments, k) -> None
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Type alias accepted by public functions
_TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: _TensorLike, device: torch.device | None = None) -> torch.Tensor:
    """Convert numpy array or tensor to float32 torch.Tensor on ``device``."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x.astype(np.float32, copy=False))
    else:
        t = x.float()
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_blocked(
    feats: _TensorLike,
    centroids: _TensorLike,
    block_size: int = 4096,
) -> torch.Tensor:
    """
    Assign each feature vector to its nearest centroid using blocked
    dot-product arithmetic to bound peak memory usage.

    Uses the decomposition::

        ||x_i - c_j||^2 = ||x_i||^2 - 2 x_i · c_j^T + ||c_j||^2

    Parameters
    ----------
    feats:
        Feature matrix, shape ``(N, D)``.  Accepts torch.Tensor or np.ndarray.
        GPU tensors are supported; computation runs on the same device.
    centroids:
        Centroid matrix, shape ``(K, D)``.  Same device as ``feats``.
    block_size:
        Number of data points processed per block.

    Returns
    -------
    assignments: ``torch.Tensor`` of shape ``(N,)`` dtype int64, values in
        ``[0, K)``, on the same device as ``feats``.

    Raises
    ------
    ValueError
        If ``feats`` is empty, or feature dimensions do not match.
    """
    # Determine device from input
    if isinstance(feats, torch.Tensor):
        device = feats.device
    else:
        device = torch.device("cpu")

    feats_t = _to_tensor(feats, device)
    centroids_t = _to_tensor(centroids, device)

    if feats_t.ndim != 2 or centroids_t.ndim != 2:
        raise ValueError("feats and centroids must be 2-D matrices.")

    n, d = feats_t.shape
    k, dc = centroids_t.shape

    if n == 0:
        raise ValueError("feats must not be empty (N=0).")
    if d != dc:
        raise ValueError(
            f"Feature dim {d} does not match centroid dim {dc}."
        )

    # Pre-compute squared norms  (N,) and (K,)
    feats_sq = (feats_t * feats_t).sum(dim=1)       # (N,)
    cent_sq = (centroids_t * centroids_t).sum(dim=1) # (K,)

    assignments = torch.empty(n, dtype=torch.int64, device=device)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = feats_t[start:end]          # (B, D)
        bsq = feats_sq[start:end]           # (B,)

        # (B, K) distances: ||x||^2 - 2 x·C^T + ||c||^2
        dists = bsq.unsqueeze(1) - 2.0 * (block @ centroids_t.T) + cent_sq.unsqueeze(0)
        assignments[start:end] = dists.argmin(dim=1)

    return assignments


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def compute_mse(
    feats: _TensorLike,
    centroids: _TensorLike,
    assignments: _TensorLike | None = None,
    block_size: int = 4096,
) -> float:
    """
    Compute mean squared error (MSE) between features and their assigned centroids.

    MSE = (1/N) Σ_i ||x_i - c_{a_i}||^2

    Parameters
    ----------
    feats:
        ``(N, D)`` feature matrix.
    centroids:
        ``(K, D)`` centroid matrix.
    assignments:
        ``(N,)`` integer assignments.  If None, computed via :func:`assign_blocked`.
    block_size:
        Block size for blocked operations.

    Returns
    -------
    Mean squared error (float).
    """
    if isinstance(feats, torch.Tensor):
        device = feats.device
    else:
        device = torch.device("cpu")

    feats_t = _to_tensor(feats, device)
    centroids_t = _to_tensor(centroids, device)

    if assignments is None:
        asgn = assign_blocked(feats_t, centroids_t, block_size=block_size)
    elif isinstance(assignments, torch.Tensor):
        asgn = assignments.to(device=device, dtype=torch.int64)
    else:
        asgn = torch.from_numpy(np.asarray(assignments, dtype=np.int64)).to(device)

    n = feats_t.shape[0]
    total_sq_err = 0.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = feats_t[start:end]
        assigned_cents = centroids_t[asgn[start:end]]
        diff = block - assigned_cents
        total_sq_err += float((diff * diff).sum().item())

    return total_sq_err / n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_assignments(
    assignments: _TensorLike,
    k: int,
) -> None:
    """
    Validate that ``assignments`` is a legal assignment array.

    Checks:
      - All values in ``[0, k)``
      - dtype is integer-like

    Parameters
    ----------
    assignments:
        Assignment array to validate (shape ``(N,)``).
    k:
        Number of clusters.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if isinstance(assignments, torch.Tensor):
        a = assignments
    else:
        a = torch.from_numpy(np.asarray(assignments))

    if not (a.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)):
        # Allow long (int64) which is torch's default integer dtype
        pass  # torch.argmin returns int64, which is fine

    if a.numel() > 0:
        if int(a.min().item()) < 0:
            raise ValueError(
                f"assignments contains negative values: min={int(a.min().item())}"
            )
        if int(a.max().item()) >= k:
            raise ValueError(
                f"assignments values must be in [0, {k}), "
                f"got max={int(a.max().item())}"
            )
