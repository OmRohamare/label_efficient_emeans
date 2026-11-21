"""
Tests for src/revive.py — dead-cluster detection and revival.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch

from src.revive import detect_dead_clusters, revive_clusters, RevivalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_counts(k=10, value=100):
    """All clusters equally populated."""
    return torch.full((k,), float(value))


def _skewed_counts(k=10, dead=3, live_value=100, dead_value=1):
    """First `dead` clusters have tiny counts, rest are healthy."""
    counts = torch.full((k,), float(live_value))
    counts[:dead] = float(dead_value)
    return counts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectDeadClusters:
    def test_detect_dead_clusters(self):
        """Clusters with count < tau * mean are detected as dead."""
        tau = 0.1
        counts = _skewed_counts(k=10, dead=3, live_value=100, dead_value=1)
        # mean ≈ (3*1 + 7*100)/10 = 70.3; tau*mean ≈ 7.03
        # Dead clusters have count=1 < 7.03 → should be detected
        dead = detect_dead_clusters(counts, tau=tau)
        assert len(dead) == 3, f"Expected 3 dead clusters, got {len(dead)}"
        assert set(dead) == {0, 1, 2}

    def test_detect_no_dead(self):
        """Uniform counts produce an empty dead list."""
        counts = _uniform_counts(k=10, value=100)
        dead = detect_dead_clusters(counts, tau=0.1)
        assert len(dead) == 0, f"Expected no dead clusters, got {dead}"

    def test_detect_respects_tau(self):
        """Higher tau catches more clusters as dead."""
        counts = torch.tensor([10.0, 50.0, 100.0, 100.0, 100.0])
        dead_low = detect_dead_clusters(counts, tau=0.05)   # mean=72, thresh=3.6 → only idx 0
        dead_high = detect_dead_clusters(counts, tau=0.5)   # thresh=36 → idx 0 and 1
        assert len(dead_low) <= len(dead_high)

    def test_detect_returns_indices(self):
        """Return value is a list (or tensor) of integer indices."""
        counts = _skewed_counts(k=8, dead=2)
        dead = detect_dead_clusters(counts, tau=0.1)
        for idx in dead:
            assert 0 <= int(idx) < 8


class TestReviveClusters:
    def test_revive_count_preserved(self):
        """Total counts are approximately preserved after revival."""
        torch.manual_seed(0)
        k = 10
        counts = _skewed_counts(k=k, dead=3, live_value=100, dead_value=1)
        centroids = torch.randn(k, 8)
        total_before = counts.sum().item()

        dead = detect_dead_clusters(counts, tau=0.1)
        new_centroids, new_counts = revive_clusters(
            centroids, counts, dead_indices=dead, seed=0
        )
        total_after = new_counts.sum().item()

        # Total mass is redistributed, not created or destroyed
        assert abs(total_after - total_before) / total_before < 0.05, (
            f"Total counts changed significantly: {total_before:.1f} → {total_after:.1f}"
        )

    def test_revive_no_dead_after(self):
        """After revival, re-detecting dead clusters returns empty."""
        torch.manual_seed(1)
        k = 10
        tau = 0.1
        counts = _skewed_counts(k=k, dead=3, live_value=100, dead_value=1)
        centroids = torch.randn(k, 8)

        dead = detect_dead_clusters(counts, tau=tau)
        new_centroids, new_counts = revive_clusters(
            centroids, counts, dead_indices=dead, seed=1
        )
        dead_after = detect_dead_clusters(new_counts, tau=tau)
        assert len(dead_after) == 0, (
            f"Dead clusters remain after revival: {dead_after}"
        )

    def test_revive_deterministic(self):
        """Same seed → identical perturbed centroids."""
        torch.manual_seed(2)
        k = 8
        counts = _skewed_counts(k=k, dead=2, live_value=100, dead_value=1)
        centroids = torch.randn(k, 8)
        dead = detect_dead_clusters(counts, tau=0.1)

        c1, n1 = revive_clusters(centroids, counts, dead_indices=dead, seed=99)
        c2, n2 = revive_clusters(centroids, counts, dead_indices=dead, seed=99)

        assert torch.allclose(c1, c2, atol=1e-6), "Revival not deterministic with same seed"
        assert torch.allclose(n1, n2, atol=1e-6), "Counts not deterministic with same seed"

    def test_revive_shape_preserved(self):
        """centroids shape is unchanged by revival."""
        torch.manual_seed(3)
        k, d = 10, 16
        counts = _skewed_counts(k=k, dead=2)
        centroids = torch.randn(k, d)
        dead = detect_dead_clusters(counts, tau=0.1)

        new_centroids, new_counts = revive_clusters(
            centroids, counts, dead_indices=dead, seed=0
        )
        assert new_centroids.shape == (k, d)
        assert new_counts.shape == (k,)


class TestRevivalConfig:
    def test_revival_config_defaults(self):
        """RevivalConfig has sensible defaults."""
        cfg = RevivalConfig()
        assert 0 < cfg.tau < 1
        assert cfg.tau <= 0.2  # typical range
