"""
emeans.py
=========
Full e-means implementation (arXiv:2611.09501, Meta FAIR).

EMA update rule:
    S_j ← (1 - α) * S_j + α * s_j
    N_j ← (1 - α) * N_j + α * n_j
    centroid_j = S_j / N_j

Alpha decays on a cosine schedule from alpha0 to alpha0/10 over all steps.
When use_schedule=False, alpha stays constant at alpha0.

PyTorch-native implementation. Accepts torch.Tensor or numpy arrays.

Public API
----------
EMeansConfig   – configuration dataclass
EMeansResult   – result dataclass
EMeans         – main class with fit()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import torch

from .assign import assign_blocked, compute_mse
from .laws import ScalingLawConfig, cosine_schedule
from .lloyd import kmeanspp_init
from .revive import RevivalConfig, maybe_revive

logger = logging.getLogger(__name__)

_TensorLike = torch.Tensor | np.ndarray


def _to_tensor(x: _TensorLike, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    return x.float().to(device)


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------

@dataclass
class EMeansConfig:
    """
    Configuration for the e-means algorithm.

    Parameters
    ----------
    k:
        Number of clusters.
    T:
        Number of passes (epoch multiplier). Controls total update steps.
    batch_size:
        Mini-batch size. If None, computed from verified scaling law.
    alpha0:
        Initial EMA learning rate. If None, computed from verified scaling law.
    use_schedule:
        If True, apply cosine decay to alpha (default True).
    use_revival:
        If True, enable dead-cluster revival.
    holdout_frac:
        Fraction of data held out purely for MSE tracking (not used in training).
        Default 0.0 for full data training.
    seed:
        Random seed for reproducibility.
    init:
        Initialization strategy: 'kmeanspp' (default in repo) or 'random' (paper default).
    finalize_last_epoch:
        If True, average the points assigned to each cluster during the last
        epoch (Algorithm 1, line 11; Section 3.2).
    init_usage:
        Initial EMA count N_j (Algorithm 1 line 2 uses 1e-10; Faiss uses 2**-16).
    batch_law:
        Scaling law variant: 'throughput' (default) or 'quality' (knee/8).
    alpha_min_factor:
        Cosine schedule end factor: alpha_min = alpha0 * factor. Default 0.1.
    revival:
        Dead-cluster revival configuration (used only when use_revival=True).
    n_local_trials:
        k-means++ oversampling.
    block_size:
        Block size for blocked assignment.
    device:
        Torch device. None → auto-detect.
    verbose:
        Log progress.
    """

    k: int
    T: int = 3
    batch_size: Optional[int] = None
    alpha0: Optional[float] = None
    use_schedule: bool = True
    use_revival: bool = False
    holdout_frac: float = 0.0
    seed: int = 42
    init: Literal["kmeanspp", "random"] = "kmeanspp"
    finalize_last_epoch: bool = True
    init_usage: float = 1e-10
    batch_law: Literal["throughput", "quality"] = "throughput"
    alpha_min_factor: float = 0.1
    revival: RevivalConfig = field(default_factory=RevivalConfig)
    n_local_trials: int = 5
    block_size: int = 4096
    device: Optional[str] = None
    verbose: bool = False


@dataclass
class EMeansResult:
    """
    Results from :meth:`EMeans.fit`.

    Fields
    ------
    centroids:
        ``(K, D)`` float32 final centroids (torch.Tensor).
    assignments:
        ``(N,)`` int64 cluster assignments for the full input (torch.Tensor).
    mse_history:
        MSE on the training set after each update step.
    holdout_mse_history:
        MSE on the holdout set after each update step (same as mse_history
        when holdout_frac=0).
    n_updates:
        Total number of EMA update steps executed.
    wall_seconds:
        Wall-clock seconds for the fit() call.
    config:
        Dict of actual hyperparameters used (batch_size, alpha0, num_updates).
    holdout_indices:
        ``(N_holdout,)`` int64 indices of holdout samples in the original feats.
    train_indices:
        ``(N_train,)`` int64 indices of training samples in the original feats.
    ema_counts:
        ``(K,)`` float64 EMA counts at the end of training.
    """

    centroids: torch.Tensor
    assignments: torch.Tensor
    mse_history: List[float]
    holdout_mse_history: List[float]
    n_updates: int
    wall_seconds: float
    config: dict
    holdout_indices: torch.Tensor
    train_indices: torch.Tensor
    ema_counts: torch.Tensor


# ---------------------------------------------------------------------------
# EMeans
# ---------------------------------------------------------------------------

class EMeans:
    """
    Mini-batch k-means with EMA updates (e-means).

    See :class:`EMeansConfig` for parameter documentation.
    """

    def __init__(self, cfg: EMeansConfig) -> None:
        self.cfg = cfg
        if cfg.device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(cfg.device)
        self.result_: Optional[EMeansResult] = None

    def fit(
        self,
        feats: _TensorLike,
        init_centroids: Optional[_TensorLike] = None,
    ) -> EMeansResult:
        """
        Run e-means on ``feats``.

        Parameters
        ----------
        feats:
            ``(N, D)`` float32 feature matrix (should be L2-normalised).
        init_centroids:
            Optional ``(K, D)`` pre-computed initial centroids for matched-init runs.

        Returns
        -------
        :class:`EMeansResult`
        """
        t0 = time.time()
        cfg = self.cfg
        feats_t = _to_tensor(feats, self._device)
        n, d = feats_t.shape

        if n < cfg.k:
            raise ValueError(f"n={n} < k={cfg.k}; cannot cluster.")

        # ----------------------------------------------------------------
        # Scaling-law config (verified formulas)
        # ----------------------------------------------------------------
        law = ScalingLawConfig(
            n=n, k=cfg.k, T=cfg.T,
            batch_size=cfg.batch_size,
            alpha0=cfg.alpha0,
            alpha_min_factor=cfg.alpha_min_factor,
            batch_law=cfg.batch_law,
        )
        batch_size = law.batch_size
        alpha0 = float(law.alpha0)
        num_updates = law.num_updates

        if cfg.verbose:
            logger.info(
                "EMeans fit: n=%d, k=%d, T=%d, batch_size=%d, alpha0=%.5f, num_updates=%d",
                n, cfg.k, cfg.T, batch_size, alpha0, num_updates,
            )

        # ----------------------------------------------------------------
        # Train / holdout split (by index), seeded
        # ----------------------------------------------------------------
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)

        all_idx = torch.arange(n, dtype=torch.int64, device=self._device)
        n_holdout = max(0, int(n * cfg.holdout_frac))

        if n_holdout > 0:
            perm = torch.randperm(n, generator=g).to(self._device)
            holdout_idx = perm[:n_holdout]
            train_idx = perm[n_holdout:]
        else:
            holdout_idx = torch.zeros(0, dtype=torch.int64, device=self._device)
            train_idx = all_idx

        feats_train = feats_t[train_idx]
        feats_holdout = feats_t[holdout_idx] if n_holdout > 0 else feats_train
        n_train = feats_train.shape[0]

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        if init_centroids is not None:
            centroids = _to_tensor(init_centroids, self._device)
        elif cfg.init == "random":
            perm_init = torch.randperm(n_train, generator=g)[:cfg.k]
            centroids = feats_train[perm_init].clone()
        else:
            # default: kmeanspp
            centroids = kmeanspp_init(
                feats_train, cfg.k, seed=cfg.seed, n_local_trials=cfg.n_local_trials
            )

        # ----------------------------------------------------------------
        # Initialise EMA accumulators (double precision)
        # S_j = N_j * c_j^(0), with N_j = init_usage (1e-10) per Algorithm 1
        # ----------------------------------------------------------------
        init_usage = cfg.init_usage
        S = centroids.double() * init_usage
        N = torch.full((cfg.k,), init_usage, dtype=torch.float64, device=self._device)

        mse_history: List[float] = []
        holdout_mse_history: List[float] = []

        # Last-epoch mean accumulators (Algorithm 1, line 11)
        pass_sums = torch.zeros(cfg.k, d, dtype=torch.float64, device=self._device)
        pass_counts = torch.zeros(cfg.k, dtype=torch.float64, device=self._device)
        last_epoch_start_step = max(0, num_updates - math.ceil(n_train / batch_size))

        # Pre-build shuffled order for mini-batches (seeded)
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(cfg.seed + 1)
        epoch_order = torch.randperm(n_train, generator=g2).to(self._device)
        epoch_pos = 0

        # ----------------------------------------------------------------
        # Main training loop
        # ----------------------------------------------------------------
        for step in range(num_updates):
            if cfg.use_schedule:
                alpha = cosine_schedule(
                    alpha0=alpha0,
                    step=step,
                    total_steps=num_updates,
                    alpha_min_factor=cfg.alpha_min_factor,
                )
            else:
                alpha = alpha0

            # Sample mini-batch (wrap around at epoch boundary)
            if epoch_pos + batch_size > n_train:
                g2.manual_seed(cfg.seed + 1 + (step // max(1, n_train // batch_size)))
                epoch_order = torch.randperm(n_train, generator=g2).to(self._device)
                epoch_pos = 0

            batch_local_idx = epoch_order[epoch_pos: epoch_pos + batch_size]
            epoch_pos += batch_size
            batch = feats_train[batch_local_idx]   # (B, D)

            # Assign batch to nearest centroid
            centroids_f32 = (S / N.unsqueeze(1)).float()
            batch_asgn = assign_blocked(batch, centroids_f32, block_size=cfg.block_size)

            # Mini-batch sums and counts
            s_j = torch.zeros(cfg.k, d, dtype=torch.float64, device=self._device)
            n_j = torch.zeros(cfg.k, dtype=torch.float64, device=self._device)

            s_j.scatter_add_(
                0,
                batch_asgn.unsqueeze(1).expand(-1, d),
                batch.double(),
            )
            n_j.scatter_add_(
                0,
                batch_asgn,
                torch.ones(batch.shape[0], dtype=torch.float64, device=self._device),
            )

            # Accumulate assignments during last epoch for finalization
            if cfg.finalize_last_epoch and step >= last_epoch_start_step:
                pass_sums += s_j
                pass_counts += n_j

            # EMA update
            S = (1.0 - alpha) * S + alpha * s_j
            N = (1.0 - alpha) * N + alpha * n_j

            # Recompute centroids (float32 for assignment)
            centroids_f32 = (S / N.unsqueeze(1)).float()

            # Dead-cluster revival (every 5 steps or configured check_every)
            if cfg.use_revival:
                rev_cfg = cfg.revival if cfg.use_revival else RevivalConfig(enabled=False)
                centroids_f32, N = maybe_revive(
                    centroids_f32, N, step, rev_cfg, rng_seed=cfg.seed + step
                )
                S = centroids_f32.double() * N.unsqueeze(1)

            # MSE tracking
            train_mse = compute_mse(feats_train, centroids_f32, block_size=cfg.block_size)
            holdout_mse = (
                compute_mse(feats_holdout, centroids_f32, block_size=cfg.block_size)
                if n_holdout > 0
                else train_mse
            )
            mse_history.append(float(train_mse))
            holdout_mse_history.append(float(holdout_mse))

            if cfg.verbose and step % max(1, num_updates // 10) == 0:
                logger.debug(
                    "Step %d/%d  alpha=%.5f  holdout_MSE=%.6f",
                    step, num_updates, alpha, holdout_mse,
                )

        # ----------------------------------------------------------------
        # Finalise centroids (Algorithm 1, line 11: last-epoch mean)
        # ----------------------------------------------------------------
        if cfg.finalize_last_epoch and pass_counts.sum() > 0:
            final_centroids = (S / N.unsqueeze(1)).float()
            nonempty = pass_counts > 0
            final_centroids[nonempty] = (
                pass_sums[nonempty] / pass_counts[nonempty].unsqueeze(1)
            ).float()
        else:
            final_centroids = (S / N.unsqueeze(1)).float()

        # Final assignments on full feats
        full_asgn = assign_blocked(feats_t, final_centroids, block_size=cfg.block_size)

        wall_seconds = time.time() - t0

        if cfg.verbose:
            logger.info(
                "EMeans done: %d steps, final holdout MSE=%.6f, wall=%.1fs",
                num_updates,
                holdout_mse_history[-1] if holdout_mse_history else 0.0,
                wall_seconds,
            )

        result = EMeansResult(
            centroids=final_centroids,
            assignments=full_asgn,
            mse_history=mse_history,
            holdout_mse_history=holdout_mse_history,
            n_updates=num_updates,
            wall_seconds=wall_seconds,
            config=dict(batch_size=batch_size, alpha0=alpha0, num_updates=num_updates),
            holdout_indices=holdout_idx,
            train_indices=train_idx,
            ema_counts=N.clone(),
        )
        self.result_ = result
        return result

    def predict(self, feats: _TensorLike) -> torch.Tensor:
        """
        Assign ``feats`` to nearest centroid using fitted centroids.

        Returns
        -------
        assignments: ``torch.Tensor`` (M,) int64.
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before predict().")
        feats_t = _to_tensor(feats, self._device)
        return assign_blocked(feats_t, self.result_.centroids, block_size=self.cfg.block_size)
