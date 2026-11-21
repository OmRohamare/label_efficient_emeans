"""
revive.py
=========
Dead-cluster detection and revival for the e-means algorithm.

A cluster j is "dead" if its EMA count N_j < tau * mean(N).
Revival uses a split-largest strategy: pick the cluster with the
highest EMA count, perturb its centroid, and assign it to the dead slot.

Accepts PyTorch tensors (CPU or GPU) for all array arguments.

Public API
----------
detect_dead_clusters(counts, tau) -> List[int]
revive_clusters(centroids, counts, dead_indices, seed) -> (centroids, counts)
RevivalConfig  (dataclass)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RevivalConfig:
    """
    Configuration for dead-cluster revival.

    Parameters
    ----------
    enabled:
        Whether to run revival during e-means training.
    tau:
        Dead-cluster threshold: cluster j is dead if N_j < tau * mean(N).
        Default 0.1.
    check_every:
        Run revival every this many update steps. Default 1.
    noise_scale:
        Scale of random perturbation applied to the split centroid
        (relative to its L2 norm). Default 1e-4.
    """

    enabled: bool = True
    tau: float = 0.1
    check_every: int = 1
    noise_scale: float = 1e-4


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_dead_clusters(
    counts: torch.Tensor | np.ndarray,
    tau: float = 0.1,
) -> List[int]:
    """
    Identify clusters whose EMA count is below ``tau * mean(counts)``.

    Parameters
    ----------
    counts:
        ``(K,)`` float tensor/array of EMA cluster counts (N_j values).
    tau:
        Threshold fraction of the mean count.

    Returns
    -------
    List of dead cluster indices (possibly empty).
    """
    if isinstance(counts, np.ndarray):
        counts = torch.from_numpy(counts.astype(np.float64))
    counts = counts.double()

    total = counts.sum().item()
    if total == 0.0:
        return list(range(len(counts)))

    mean_count = counts.mean().item()
    threshold = tau * mean_count
    dead = (counts < threshold).nonzero(as_tuple=False).squeeze(1).tolist()
    dead = [int(d) for d in dead]

    if dead:
        logger.debug(
            "Dead clusters detected (tau=%.3f, threshold=%.4f): %s",
            tau, threshold, dead,
        )
    return dead


# ---------------------------------------------------------------------------
# Revival
# ---------------------------------------------------------------------------

def revive_clusters(
    centroids: torch.Tensor | np.ndarray,
    counts: torch.Tensor | np.ndarray,
    dead_indices: List[int] | None = None,
    seed: int | None = None,
    tau: float = 0.1,
    noise_scale: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Revive dead clusters using a split-largest strategy.

    For each dead cluster j (sorted by ascending count so weakest first):
      1. Find the alive cluster with the highest count.
      2. Perturb its centroid by small Gaussian noise scaled by ``noise_scale``.
      3. Assign the perturbed centroid to cluster j.
      4. Split the EMA count equally between the two clusters.

    Parameters
    ----------
    centroids:
        ``(K, D)`` float32 centroid matrix (tensor or numpy array).
    counts:
        ``(K,)`` float64 EMA count array (tensor or numpy array).
    dead_indices:
        List of indices of dead clusters. If None, detected automatically
        using ``detect_dead_clusters(counts, tau=tau)``.
    seed:
        Integer seed for deterministic noise generation.  None → random.
    tau:
        Dead-cluster threshold (used only if dead_indices is None).
    noise_scale:
        Relative noise magnitude for split perturbation.

    Returns
    -------
    centroids: ``torch.Tensor`` (K, D) float32
    counts: ``torch.Tensor`` (K,) float64
    """
    # Convert inputs to tensors
    if isinstance(centroids, np.ndarray):
        centroids = torch.from_numpy(centroids.astype(np.float32))
    else:
        centroids = centroids.float()

    if isinstance(counts, np.ndarray):
        counts = torch.from_numpy(counts.astype(np.float64))
    else:
        counts = counts.double()

    device = centroids.device
    counts = counts.to(device)

    if dead_indices is None:
        dead_indices = detect_dead_clusters(counts, tau=tau)

    if not dead_indices:
        return centroids, counts

    centroids = centroids.clone()
    counts = counts.clone()

    # Deterministic RNG
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Process weakest dead clusters first
    dead_sorted = sorted(dead_indices, key=lambda j: counts[j].item())

    for dead_j in dead_sorted:
        # Find the cluster with the highest count (uses current counts)
        alive_j = int(counts.argmax().item())
        if alive_j == dead_j:
            # Edge case: pick any other
            temp = counts.clone()
            temp[dead_j] = -1.0
            alive_j = int(temp.argmax().item())

        norm = centroids[alive_j].norm().item()
        noise = torch.randn(centroids.shape[1], generator=rng)
        noise = noise * (noise_scale * norm if norm > 0 else 1e-6)
        noise = noise.to(device)

        centroids[dead_j] = centroids[alive_j] + noise
        half = counts[alive_j].item() / 2.0
        counts[alive_j] = half
        counts[dead_j] = half

        logger.debug(
            "Revived cluster %d from cluster %d (count split: %.4f each)",
            dead_j, alive_j, half,
        )

    return centroids, counts


# ---------------------------------------------------------------------------
# Convenience wrapper for e-means training loop
# ---------------------------------------------------------------------------

def maybe_revive(
    centroids: torch.Tensor,
    counts: torch.Tensor,
    step: int,
    cfg: RevivalConfig,
    rng_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditionally run revival based on ``cfg.check_every``.

    Parameters
    ----------
    centroids:
        ``(K, D)`` float32 centroid matrix.
    counts:
        ``(K,)`` float64 EMA count array.
    step:
        Current update step (0-indexed).
    cfg:
        :class:`RevivalConfig` instance.
    rng_seed:
        Seed for the revival noise generator (combine step + base seed).

    Returns
    -------
    (centroids, counts) – possibly updated.
    """
    if not cfg.enabled:
        return centroids, counts
    if step % cfg.check_every != 0:
        return centroids, counts

    dead = detect_dead_clusters(counts, tau=cfg.tau)
    if not dead:
        return centroids, counts

    return revive_clusters(
        centroids, counts,
        dead_indices=dead,
        seed=rng_seed,
        noise_scale=cfg.noise_scale,
    )
