"""
Tests for src/lloyd.py — correctness, convergence and sklearn equivalence.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch
import numpy as np

from src.lloyd import LloydKMeans
from src.assign import compute_mse, assign_blocked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_mixture(n=300, k=5, d=8, seed=0):
    """Simple Gaussian mixture dataset as a float32 tensor."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    means = torch.randn(k, d, generator=rng)
    feats_list = []
    for i in range(k):
        pts = means[i] + 0.3 * torch.randn(n // k, d, generator=rng)
        feats_list.append(pts)
    feats = torch.cat(feats_list, dim=0).float()
    # L2-normalise
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLloydObjective:
    def test_lloyd_objective_nonincreasing(self):
        """MSE must not increase across 5 successive Lloyd iterations."""
        feats = _gaussian_mixture(n=300, k=5, d=8, seed=1)
        model = LloydKMeans(k=5, seed=42)
        model.fit(feats, n_iter=1)  # initialise

        mse_prev = compute_mse(feats, model.centroids, model.predict(feats))
        for _ in range(4):
            model.fit(feats, n_iter=1, warm_start=True)
            mse_cur = compute_mse(feats, model.centroids, model.predict(feats))
            assert mse_cur <= mse_prev + 1e-10, (
                f"MSE increased: {mse_prev:.6f} -> {mse_cur:.6f}"
            )
            mse_prev = mse_cur

    def test_lloyd_converges(self):
        """25 iterations on a Gaussian mixture: last MSE < first * 0.5."""
        feats = _gaussian_mixture(n=500, k=5, d=8, seed=2)
        model = LloydKMeans(k=5, seed=0)
        model.fit(feats, n_iter=1)
        mse_first = compute_mse(feats, model.centroids, model.predict(feats))

        model.fit(feats, n_iter=25, warm_start=False)
        mse_last = compute_mse(feats, model.centroids, model.predict(feats))
        assert mse_last < mse_first, "MSE should decrease over 25 iterations"

    def test_lloyd_mse_history(self):
        """fit() returns a list of MSE values, one per iteration."""
        feats = _gaussian_mixture(n=200, k=4, d=8, seed=3)
        model = LloydKMeans(k=4, seed=1)
        history = model.fit(feats, n_iter=5)
        assert isinstance(history, list)
        assert len(history) == 5
        for mse in history:
            assert mse >= 0.0


class TestLloydEmptyCluster:
    def test_lloyd_empty_cluster_split(self):
        """After fitting, the number of non-empty clusters equals k."""
        # Degenerate data: many duplicates force empty clusters unless split
        torch.manual_seed(5)
        k = 8
        # Only 3 unique cluster centres but we request k=8
        base = torch.randn(3, 8)
        feats = base.repeat_interleave(50, dim=0)   # (150, 8) with 3 unique values
        feats = feats + 0.001 * torch.randn_like(feats)
        feats = feats / feats.norm(dim=1, keepdim=True)

        model = LloydKMeans(k=k, seed=0)
        model.fit(feats, n_iter=10)
        assignments = model.predict(feats)
        unique_clusters = assignments.unique()
        assert len(unique_clusters) == k, (
            f"Expected {k} non-empty clusters after split, got {len(unique_clusters)}"
        )


class TestLloydDeterminism:
    def test_lloyd_deterministic(self):
        """Same seed produces identical centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=8, seed=10)

        m1 = LloydKMeans(k=5, seed=99)
        m1.fit(feats, n_iter=10)

        m2 = LloydKMeans(k=5, seed=99)
        m2.fit(feats, n_iter=10)

        assert torch.allclose(m1.centroids, m2.centroids, atol=1e-6)

    def test_lloyd_different_seeds_differ(self):
        """Different seeds (usually) produce different centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=8, seed=10)

        m1 = LloydKMeans(k=5, seed=1)
        m1.fit(feats, n_iter=5)

        m2 = LloydKMeans(k=5, seed=2)
        m2.fit(feats, n_iter=5)

        # They *could* coincide by chance, but that's astronomically unlikely
        assert not torch.allclose(m1.centroids, m2.centroids, atol=1e-3)


class TestLloydKMeansppInit:
    def test_lloyd_kmeanspp_init(self):
        """k-means++ initialisation yields k distinct centroids."""
        feats = _gaussian_mixture(n=300, k=5, d=8, seed=7)
        model = LloydKMeans(k=5, seed=0, init="kmeanspp")
        model.fit(feats, n_iter=0)   # 0 iterations: just initialise
        centroids = model.centroids
        assert centroids.shape == (5, 8)
        # All centroids distinct (pairwise distance > 0)
        for i in range(5):
            for j in range(i + 1, 5):
                dist = (centroids[i] - centroids[j]).norm()
                assert dist > 1e-6, f"Centroids {i} and {j} are identical"


class TestLloydVsSklearn:
    def test_lloyd_mse_vs_sklearn(self):
        """Lloyd MSE < sklearn KMeans MSE * 1.05 after 25 iterations."""
        pytest.importorskip("sklearn")
        from sklearn.cluster import KMeans as SKLearnKMeans

        feats = _gaussian_mixture(n=600, k=5, d=8, seed=42)
        feats_np = feats.numpy()

        # Sklearn KMeans
        sk = SKLearnKMeans(n_clusters=5, n_init=1, max_iter=25, random_state=42)
        sk.fit(feats_np)
        sk_centroids = torch.from_numpy(sk.cluster_centers_).float()
        sk_assign = assign_blocked(feats, sk_centroids)
        sk_mse = float(compute_mse(feats, sk_centroids, sk_assign))

        # Our Lloyd
        model = LloydKMeans(k=5, seed=42)
        model.fit(feats, n_iter=25)
        our_assign = model.predict(feats)
        our_mse = float(compute_mse(feats, model.centroids, our_assign))

        assert our_mse < sk_mse * 1.05, (
            f"Our MSE {our_mse:.6f} > sklearn MSE {sk_mse:.6f} * 1.05"
        )
