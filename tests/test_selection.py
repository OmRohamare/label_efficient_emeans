"""
Tests for src/select.py — sample selection strategies.
"""
import sys
import os
import inspect

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch

from src.select import (
    select_nearest_to_centroids,
    select_random,
    select_kmeanspp,
    validate_selection,
    SelectionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feats_and_centroids(n=200, k=10, d=16, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    feats = torch.randn(n, d, generator=rng)
    feats = feats / feats.norm(dim=1, keepdim=True)
    centroids = torch.randn(k, d, generator=rng)
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    return feats, centroids


# ---------------------------------------------------------------------------
# Tests for select_nearest_to_centroids
# ---------------------------------------------------------------------------

class TestSelectNearestToCentroids:
    def test_select_B_distinct(self):
        """Returns exactly B distinct indices."""
        feats, centroids = _feats_and_centroids(n=200, k=10, d=16, seed=1)
        B = 30
        result = select_nearest_to_centroids(feats, centroids, B=B, seed=0)
        assert len(result.indices) == B
        assert len(result.indices.unique()) == B

    def test_select_valid_indices(self):
        """All returned indices are in [0, n_pool)."""
        n = 200
        feats, centroids = _feats_and_centroids(n=n, k=10, d=16, seed=2)
        result = select_nearest_to_centroids(feats, centroids, B=25, seed=0)
        assert (result.indices >= 0).all()
        assert (result.indices < n).all()

    def test_select_no_labels_param(self):
        """Function signature must NOT have a 'labels' parameter."""
        sig = inspect.signature(select_nearest_to_centroids)
        assert "labels" not in sig.parameters, (
            "select_nearest_to_centroids must not accept a 'labels' parameter"
        )

    def test_select_handles_duplicates(self):
        """When two centroids share the same nearest point, next-nearest is used."""
        # Create a small pool where two centroids are very close to point 0
        d = 8
        feats = torch.zeros(10, d)
        feats[0] = torch.ones(d) / d ** 0.5   # point 0 is the target

        # Two centroids both pointing at point 0
        centroids = torch.zeros(2, d)
        centroids[0] = torch.ones(d) / d ** 0.5 * 1.01
        centroids[1] = torch.ones(d) / d ** 0.5 * 0.99

        # With B=2, must return 2 DISTINCT indices
        result = select_nearest_to_centroids(feats, centroids, B=2, seed=0)
        assert len(result.indices.unique()) == 2, (
            "Duplicate nearest neighbour not handled — B=2 should yield 2 distinct indices"
        )

    def test_select_result_type(self):
        """Result is a SelectionResult with an `indices` attribute."""
        feats, centroids = _feats_and_centroids(n=100, k=5, d=8, seed=3)
        result = select_nearest_to_centroids(feats, centroids, B=10, seed=0)
        assert isinstance(result, SelectionResult)
        assert hasattr(result, "indices")


# ---------------------------------------------------------------------------
# Tests for select_random
# ---------------------------------------------------------------------------

class TestSelectRandom:
    def test_select_random_distinct(self):
        """select_random returns exactly B distinct indices."""
        feats, _ = _feats_and_centroids(n=200, k=10, d=16, seed=4)
        B = 50
        result = select_random(feats, B=B, seed=0)
        assert len(result.indices) == B
        assert len(result.indices.unique()) == B

    def test_select_random_valid_indices(self):
        n = 150
        feats = torch.randn(n, 8)
        result = select_random(feats, B=30, seed=1)
        assert (result.indices >= 0).all()
        assert (result.indices < n).all()


# ---------------------------------------------------------------------------
# Tests for select_kmeanspp
# ---------------------------------------------------------------------------

class TestSelectKMeanspp:
    def test_select_kmeanspp_distinct(self):
        """select_kmeanspp returns exactly B distinct indices."""
        feats, _ = _feats_and_centroids(n=200, k=10, d=16, seed=5)
        B = 20
        result = select_kmeanspp(feats, B=B, seed=0)
        assert len(result.indices) == B
        assert len(result.indices.unique()) == B

    def test_select_kmeanspp_valid_indices(self):
        n = 100
        feats = torch.randn(n, 8)
        feats = feats / feats.norm(dim=1, keepdim=True)
        result = select_kmeanspp(feats, B=15, seed=2)
        assert (result.indices >= 0).all()
        assert (result.indices < n).all()


# ---------------------------------------------------------------------------
# Tests for validate_selection
# ---------------------------------------------------------------------------

class TestValidateSelection:
    def test_validate_selection_raises_out_of_range(self):
        """Indices >= n_pool raise ValueError."""
        indices = torch.tensor([0, 1, 200])  # 200 >= n_pool=100
        with pytest.raises(ValueError):
            validate_selection(indices, n_pool=100)

    def test_validate_selection_raises_negative(self):
        """Negative indices raise ValueError."""
        indices = torch.tensor([-1, 0, 1])
        with pytest.raises(ValueError):
            validate_selection(indices, n_pool=100)

    def test_validate_selection_raises_duplicates(self):
        """Duplicate indices raise ValueError."""
        indices = torch.tensor([0, 1, 1, 2])
        with pytest.raises(ValueError):
            validate_selection(indices, n_pool=100)

    def test_validate_selection_valid(self):
        """Valid distinct in-range indices pass without error."""
        indices = torch.tensor([0, 5, 10, 99])
        validate_selection(indices, n_pool=100)  # should not raise
