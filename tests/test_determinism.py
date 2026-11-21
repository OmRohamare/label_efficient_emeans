"""
Tests for determinism guarantees across the entire codebase.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch
import numpy as np

from src.reproducibility import seed_everything
from src.emeans import EMeans, EMeansConfig
from src.lloyd import LloydKMeans
from src.select import select_nearest_to_centroids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_mixture(n=300, k=5, d=16, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    means = torch.randn(k, d, generator=rng)
    feats_list = []
    for i in range(k):
        pts = means[i] + 0.2 * torch.randn(n // k, d, generator=rng)
        feats_list.append(pts)
    feats = torch.cat(feats_list, dim=0).float()
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_seed_everything(self):
        """seed_everything(42) called twice → same RNG state."""
        seed_everything(42)
        r1_torch = torch.randn(10)
        r1_np = np.random.rand(10)

        seed_everything(42)
        r2_torch = torch.randn(10)
        r2_np = np.random.rand(10)

        assert torch.allclose(r1_torch, r2_torch), "torch RNG not reset by seed_everything"
        assert np.allclose(r1_np, r2_np), "numpy RNG not reset by seed_everything"

    def test_seed_everything_different_seeds(self):
        """Different seeds produce different random sequences."""
        seed_everything(1)
        r1 = torch.randn(10)

        seed_everything(2)
        r2 = torch.randn(10)

        assert not torch.allclose(r1, r2)


class TestEMeansDeterminism:
    def test_emeans_determinism(self):
        """Two runs with seed=42 produce identical centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=16, seed=0)
        cfg = EMeansConfig(
            k=5, T=3, batch_size=32, alpha0=0.5,
            use_schedule=True, use_revival=False,
            holdout_frac=0.1, seed=42,
        )

        m1 = EMeans(cfg)
        r1 = m1.fit(feats)

        m2 = EMeans(cfg)
        r2 = m2.fit(feats)

        assert torch.allclose(r1.centroids, r2.centroids, atol=1e-5), (
            "EMeans centroids not identical across two runs with same seed"
        )

    def test_emeans_different_seeds_differ(self):
        """Different seeds produce different centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=16, seed=0)

        cfg1 = EMeansConfig(k=5, T=2, batch_size=32, alpha0=0.5,
                            use_schedule=False, use_revival=False,
                            holdout_frac=0.0, seed=1)
        cfg2 = EMeansConfig(k=5, T=2, batch_size=32, alpha0=0.5,
                            use_schedule=False, use_revival=False,
                            holdout_frac=0.0, seed=2)

        r1 = EMeans(cfg1).fit(feats)
        r2 = EMeans(cfg2).fit(feats)

        assert not torch.allclose(r1.centroids, r2.centroids, atol=1e-3)


class TestLloydDeterminism:
    def test_lloyd_determinism(self):
        """LloydKMeans with same seed → same centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=16, seed=0)

        m1 = LloydKMeans(k=5, seed=99)
        m1.fit(feats, n_iter=10)

        m2 = LloydKMeans(k=5, seed=99)
        m2.fit(feats, n_iter=10)

        assert torch.allclose(m1.centroids, m2.centroids, atol=1e-6), (
            "LloydKMeans centroids not deterministic with same seed"
        )


class TestSelectionDeterminism:
    def test_selection_determinism(self):
        """select_nearest_to_centroids returns same result on repeated calls."""
        feats = _gaussian_mixture(n=300, k=5, d=16, seed=0)
        centroids = feats[:5].clone()  # pick first 5 as centroids
        B = 20

        seed_everything(0)
        r1 = select_nearest_to_centroids(feats, centroids, B=B, seed=0)

        seed_everything(0)
        r2 = select_nearest_to_centroids(feats, centroids, B=B, seed=0)

        assert torch.equal(r1.indices, r2.indices), (
            "select_nearest_to_centroids is not deterministic"
        )
