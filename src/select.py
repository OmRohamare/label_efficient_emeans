"""
select.py
=========
Sample-selection strategies for label-efficient learning.

Given a pool of unlabelled features and cluster centroids (or directly),
select a budget B of samples to label.

All functions accept torch.Tensor or numpy arrays and return a SelectionResult
whose `.indices` is a ``torch.Tensor`` of int64.

Strategies
----------
select_nearest_to_centroids  – one sample per centroid, nearest in L2 (no labels used).
select_random                – uniform random without replacement.
select_kmeanspp              – k-means++ diverse greedy selection.
validate_selection           – validate a selection (raises ValueError on error).

Public API
----------
SelectionResult               – dataclass
select_nearest_to_centroids(feats, centroids, B, seed) -> SelectionResult
select_random(feats, B, seed) -> SelectionResult
select_kmeanspp(feats, B, seed) -> SelectionResult
validate_selection(indices, n_pool) -> None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

_TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: _TensorLike, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x.astype(np.float32, copy=False))
    else:
        t = x.float()
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """
    Result of a sample-selection strategy.

    Attributes
    ----------
    indices:
        ``(B,)`` int64 torch.Tensor of selected indices into the feature pool.
        All values are distinct and in ``[0, N_pool)``.
    method:
        Name of the selection strategy used.
    B:
        Requested budget (number of samples to select).
    """

    indices: torch.Tensor   # (B,) int64, distinct
    method: str
    B: int


# ---------------------------------------------------------------------------
# Nearest-to-centroids
# ---------------------------------------------------------------------------

def select_nearest_to_centroids(
    feats: _TensorLike,
    centroids: _TensorLike,
    B: int,
    seed: int = 0,
    block_size: int = 4096,
) -> SelectionResult:
    """
    Select the ``B`` samples nearest to their respective cluster centroids.

    Strategy:
      1. For each centroid j, find the pool sample nearest to it in L2.
      2. If two centroids share the same nearest sample, the second centroid
         selects its next-nearest unselected sample.
      3. If K > B, keep the B samples whose centroid-distance is smallest.
      4. If K < B, fill remaining budget uniformly at random from unselected pool.

    No labels are ever used (see: no ``labels`` parameter).

    Parameters
    ----------
    feats:
        ``(N, D)`` float32 feature matrix (pool).
    centroids:
        ``(K, D)`` float32 centroid matrix.
    B:
        Budget: number of samples to select.
    seed:
        Integer seed for tie-breaking / random fill.
    block_size:
        Block size for blocked centroid-distance computation.

    Returns
    -------
    :class:`SelectionResult` with ``B`` distinct indices.
    """
    device = (
        feats.device
        if isinstance(feats, torch.Tensor)
        else torch.device("cpu")
    )
    feats_t = _to_tensor(feats, device=device)
    centroids_t = _to_tensor(centroids, device=device)
    n, d = feats_t.shape
    k = centroids_t.shape[0]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Chunked search: never materialise the full (K, N) matrix or a full argsort.
    # Peak extra memory ~ CH * N * 4 bytes (~100 MB for CH=256, N=100k).
    cent_sq = (centroids_t * centroids_t).sum(dim=1)   # (K,)
    feats_sq = (feats_t * feats_t).sum(dim=1)          # (N,)

    selected = []
    used: set[int] = set()
    CH = 256
    topm = min(16, n)

    for c0 in range(0, k, CH):
        c1 = min(c0 + CH, k)
        d = (
            cent_sq[c0:c1].unsqueeze(1)
            - 2.0 * (centroids_t[c0:c1] @ feats_t.T)
            + feats_sq.unsqueeze(0)
        )                                               # (CH, N)
        cand_idx = d.topk(topm, dim=1, largest=False).indices   # sorted ascending
        for r in range(c1 - c0):
            picked = None
            for cand in cand_idx[r].tolist():
                if cand not in used:
                    picked = cand
                    break
            if picked is None:                          # rare: all topm already taken
                for cand in d[r].argsort().tolist():
                    if cand not in used:
                        picked = cand
                        break
            selected.append((float(d[r, picked].item()), picked, c0 + r))
            used.add(picked)
        del d, cand_idx

    # If K > B: keep B closest (smallest distance)
    if len(selected) > B:
        selected.sort(key=lambda t: t[0])
        selected = selected[:B]

    # If K < B: fill with random unselected pool points
    if len(selected) < B:
        remaining = [i for i in range(n) if i not in used]
        need = B - len(selected)
        perm = torch.randperm(len(remaining), generator=rng).tolist()[:need]
        for idx in perm:
            selected.append((0.0, remaining[idx], -1))
            used.add(remaining[idx])

    indices = torch.tensor([s[1] for s in selected], dtype=torch.int64)
    return SelectionResult(indices=indices, method="nearest_to_centroids", B=B)


# ---------------------------------------------------------------------------
# Random selection
# ---------------------------------------------------------------------------

def select_random(
    feats: _TensorLike,
    B: int,
    seed: int = 0,
) -> SelectionResult:
    """
    Select ``B`` pool indices uniformly at random without replacement.

    Parameters
    ----------
    feats:
        ``(N, D)`` feature matrix (only used for pool size).
    B:
        Budget.
    seed:
        Integer seed for reproducibility.

    Returns
    -------
    :class:`SelectionResult` with ``B`` distinct indices.
    """
    if isinstance(feats, torch.Tensor):
        n = feats.shape[0]
    else:
        n = feats.shape[0]

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    indices = perm[:B]
    return SelectionResult(indices=indices, method="random", B=B)


# ---------------------------------------------------------------------------
# k-means++ seeding (selection only, no Lloyd)
# ---------------------------------------------------------------------------

def select_kmeanspp(
    feats: _TensorLike,
    B: int,
    seed: int = 0,
    n_local_trials: int = 3,
) -> SelectionResult:
    """
    Select ``B`` pool indices using k-means++ D²-weighted seeding.

    No Lloyd iterations are run.  Returns actual pool indices (not centroids).

    Parameters
    ----------
    feats:
        ``(N, D)`` feature matrix.
    B:
        Budget (number of points to select).
    seed:
        Integer seed for reproducibility.
    n_local_trials:
        Candidate oversampling per step.

    Returns
    -------
    :class:`SelectionResult` with ``B`` distinct indices.
    """
    feats_t = _to_tensor(feats)
    n = feats_t.shape[0]
    dev = feats_t.device

    rng = torch.Generator()
    rng.manual_seed(seed)

    sq_norms = (feats_t * feats_t).sum(dim=1)

    def _sqdist(rows: torch.Tensor) -> torch.Tensor:
        # (m, D) -> (m, N) squared distances via one matmul
        d = sq_norms.unsqueeze(0) + (rows * rows).sum(1, keepdim=True) \
            - 2.0 * (rows @ feats_t.T)
        return d.clamp_(min=0.0)

    idx0 = int(torch.randint(n, (1,), generator=rng).item())
    selected = torch.zeros(n, dtype=torch.bool, device=dev)
    selected[idx0] = True
    order = [idx0]
    min_sq = _sqdist(feats_t[idx0:idx0 + 1])[0]  # one matmul per step
    min_sq[selected] = 0.0

    for _ in range(1, min(B, n)):
        total = min_sq.sum().item()
        if total <= 0.0:
            pool = (~selected).nonzero().squeeze(1).cpu()
            cand = pool[torch.randint(len(pool), (n_local_trials,), generator=rng)]
        else:
            cand = torch.multinomial(
                (min_sq / total).cpu(), min(n_local_trials, n),
                replacement=False, generator=rng,
            )
        cand = cand[~selected[cand.to(dev)].cpu()]
        if len(cand) == 0:
            cand = (~selected).nonzero().squeeze(1)[:1].cpu()
        cand = cand.to(dev)
        d2 = _sqdist(feats_t[cand])                      # (c, N)
        new_min = torch.minimum(min_sq.unsqueeze(0), d2)
        best = int(new_min.sum(dim=1).argmin().item())
        best_idx = int(cand[best].item())
        min_sq = new_min[best].clone()
        selected[best_idx] = True
        min_sq[selected] = 0.0
        order.append(best_idx)

    indices = torch.tensor(order, dtype=torch.int64)
    return SelectionResult(indices=indices, method="kmeanspp", B=B)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_selection(
    indices: _TensorLike,
    n_pool: int,
) -> None:
    """
    Validate that ``indices`` is a legal selection.

    Checks:
      - All values in ``[0, n_pool)``
      - No duplicate indices

    Parameters
    ----------
    indices:
        Selected index array.
    n_pool:
        Size of the unlabelled feature pool.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if isinstance(indices, torch.Tensor):
        idx = indices
    else:
        idx = torch.from_numpy(np.asarray(indices, dtype=np.int64))

    if idx.numel() == 0:
        return

    if int(idx.min().item()) < 0:
        raise ValueError(
            f"Selection contains negative indices: min={int(idx.min().item())}"
        )
    if int(idx.max().item()) >= n_pool:
        raise ValueError(
            f"Selection index {int(idx.max().item())} >= n_pool={n_pool}"
        )
    if len(idx.unique()) < len(idx):
        raise ValueError(
            f"Selection contains duplicates: "
            f"{len(idx)} indices but only {len(idx.unique())} unique."
        )
