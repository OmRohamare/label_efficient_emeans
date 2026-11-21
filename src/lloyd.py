"""
lloyd.py
========
Full-batch Lloyd's k-means algorithm with:
  - k-means++ initialisation
  - Empty-cluster split policy (split largest cluster)
  - Per-iteration MSE tracking
  - PyTorch-native (CPU + GPU); accepts tensors or numpy arrays

Public API
----------
LloydKMeans   – class with fit(feats, n_iter, warm_start) -> List[float]
kmeanspp_init(feats, k, seed) -> torch.Tensor
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch

from .assign import assign_blocked, compute_mse

logger = logging.getLogger(__name__)

_TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: _TensorLike, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    return x.float().to(device)


# ---------------------------------------------------------------------------
# k-means++ initialisation
# ---------------------------------------------------------------------------

def kmeanspp_init(
    feats: torch.Tensor,
    k: int,
    seed: int = 42,
    n_local_trials: int = 5,
) -> torch.Tensor:
    """
    k-means++ seeding (Arthur & Vassilvitskii, 2007).

    Parameters
    ----------
    feats:
        ``(N, D)`` float32 tensor.
    k:
        Number of clusters.
    seed:
        Integer seed for reproducibility.
    n_local_trials:
        Candidate oversampling per step.

    Returns
    -------
    centroids: ``(K, D)`` float32 tensor on the same device as feats.
    """
    device = feats.device
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    n = feats.shape[0]
    # First centroid: uniform at random
    idx0 = int(torch.randint(n, (1,), generator=rng).item())
    centroids = [feats[idx0].clone()]

    min_sq_dist = torch.full((n,), float("inf"), device=device)

    for _ in range(1, k):
        last_c = centroids[-1]
        diff = feats - last_c.unsqueeze(0)
        sq = (diff * diff).sum(dim=1)
        torch.minimum(min_sq_dist, sq, out=min_sq_dist)

        total = min_sq_dist.sum().item()
        if total == 0.0:
            cand_idx = torch.randint(n, (n_local_trials,), generator=rng).tolist()
        else:
            probs_cpu = (min_sq_dist / total).cpu()
            cand_idx = torch.multinomial(
                probs_cpu, n_local_trials, replacement=False, generator=rng
            ).tolist()

        best_idx = cand_idx[0]
        best_pot = float("inf")
        for ci in cand_idx:
            c = feats[ci]
            d2 = feats - c.unsqueeze(0)
            sq_c = (d2 * d2).sum(dim=1)
            pot = torch.minimum(min_sq_dist, sq_c).sum().item()
            if pot < best_pot:
                best_pot = pot
                best_idx = ci

        centroids.append(feats[best_idx].clone())

    return torch.stack(centroids, dim=0)   # (K, D)


# ---------------------------------------------------------------------------
# Lloyd k-means
# ---------------------------------------------------------------------------

class LloydKMeans:
    """
    Full-batch Lloyd's k-means.

    Parameters
    ----------
    k:
        Number of clusters.
    seed:
        Integer RNG seed for initialisation and empty-cluster splitting.
    init:
        ``"kmeanspp"`` (default) or ``"random"``.
    n_local_trials:
        Candidate oversampling for k-means++ (ignored if init=="random").
    tol:
        Relative MSE improvement tolerance for early stopping (applied only
        after the second iteration).
    device:
        Torch device.  ``None`` → auto-detect (CUDA if available, else CPU).
    """

    def __init__(
        self,
        k: int,
        seed: int = 42,
        init: str = "kmeanspp",
        n_local_trials: int = 5,
        tol: float = 1e-9,
        device: torch.device | str | None = None,
    ) -> None:
        self.k = k
        self.seed = seed
        self.init = init
        self.n_local_trials = n_local_trials
        self.tol = tol

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # State
        self._rng_state: int = seed       # used for the split generator
        self.centroids: torch.Tensor | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        feats: _TensorLike,
        n_iter: int = 25,
        warm_start: bool = False,
        early_stop: bool = False,
    ) -> List[float]:
        """
        Run Lloyd's algorithm.

        Parameters
        ----------
        feats:
            ``(N, D)`` feature matrix (tensor or numpy array).
        n_iter:
            Number of Lloyd iterations to run.
        warm_start:
            If True and :attr:`centroids` is already set, skip initialisation
            and continue from the current centroids.
        early_stop:
            If True, stop once the relative MSE improvement falls below
            ``self.tol``. Default False: run exactly ``n_iter`` iterations,
            so the history has one entry per iteration.

        Returns
        -------
        mse_history: list of float, one entry per iteration.
        """
        feats_t = _to_tensor(feats, self.device)
        n, d = feats_t.shape

        if n < self.k:
            raise ValueError(f"n={n} < k={self.k}; cannot cluster.")

        # Initialise
        if not warm_start or self.centroids is None or not self._initialized:
            if self.init == "kmeanspp":
                self.centroids = kmeanspp_init(
                    feats_t, self.k, seed=self.seed,
                    n_local_trials=self.n_local_trials,
                )
            elif self.init == "random":
                g = torch.Generator()
                g.manual_seed(self.seed)
                idx = torch.randperm(n, generator=g)[:self.k]
                self.centroids = feats_t[idx].clone()
            else:
                raise ValueError(f"Unknown init: {self.init!r}")
            self._initialized = True

        mse_history: List[float] = []
        # Use a deterministic split generator seeded from self.seed
        split_gen = torch.Generator()
        split_gen.manual_seed(self.seed + 10000)

        for it in range(n_iter):
            # E-step: assign
            assignments = assign_blocked(feats_t, self.centroids)

            # Handle empty clusters
            self.centroids = self._handle_empty_clusters(
                feats_t, self.centroids, assignments, split_gen
            )
            # Recompute assignments after revival
            assignments = assign_blocked(feats_t, self.centroids)

            # M-step: recompute centroids
            self.centroids = self._recompute_centroids(feats_t, assignments, d)

            mse = compute_mse(feats_t, self.centroids, assignments)
            mse_history.append(float(mse))

            logger.debug("Lloyd iter %d/%d  MSE=%.6f", it + 1, n_iter, mse)

            if early_stop and len(mse_history) >= 2:
                rel_imp = abs(mse_history[-2] - mse_history[-1]) / (mse_history[-2] + 1e-12)
                if rel_imp < self.tol:
                    logger.info("Lloyd converged at iter %d (rel_imp=%.2e)", it + 1, rel_imp)
                    break

        return mse_history

    def predict(self, feats: _TensorLike) -> torch.Tensor:
        """
        Assign ``feats`` to nearest centroid.

        Returns
        -------
        assignments: ``torch.Tensor`` of shape ``(N,)`` int64.
        """
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
        feats_t = _to_tensor(feats, self.device)
        return assign_blocked(feats_t, self.centroids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recompute_centroids(
        self,
        feats: torch.Tensor,
        assignments: torch.Tensor,
        d: int,
    ) -> torch.Tensor:
        """M-step: compute mean of each cluster."""
        new_centroids = torch.zeros(self.k, d, dtype=torch.float64, device=feats.device)
        counts = torch.zeros(self.k, dtype=torch.int64, device=feats.device)

        new_centroids.scatter_add_(
            0,
            assignments.unsqueeze(1).expand(-1, d),
            feats.double(),
        )
        counts.scatter_add_(
            0,
            assignments,
            torch.ones(feats.shape[0], dtype=torch.int64, device=feats.device),
        )

        mask = counts > 0
        new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1).double()
        return new_centroids.float()

    def _handle_empty_clusters(
        self,
        feats: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Split the largest cluster to fill each empty cluster."""
        counts = torch.bincount(assignments.long(), minlength=self.k)
        dead = (counts == 0).nonzero(as_tuple=False).squeeze(1)

        if dead.numel() == 0:
            return centroids

        logger.debug("Empty clusters %s — applying split_largest", dead.tolist())
        centroids = centroids.clone()

        for dead_j in dead.tolist():
            largest_j = int(counts.argmax().item())
            noise = torch.randn(centroids.shape[1], generator=rng).to(centroids.device)
            noise *= 1e-4 * centroids[largest_j].norm().item()
            centroids[dead_j] = centroids[largest_j] + noise
            half = counts[largest_j].item() // 2
            counts[dead_j] = half
            counts[largest_j] -= half

        return centroids
