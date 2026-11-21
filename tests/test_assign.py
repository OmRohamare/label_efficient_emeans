"""
Tests for src/assign.py
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch

from src.assign import assign_blocked, compute_mse, validate_assignments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brute_force_assign(feats: torch.Tensor, centroids: torch.Tensor):
    """Reference assignment via torch.cdist."""
    dists = torch.cdist(feats, centroids, p=2)  # (n, k)
    return dists.argmin(dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignBlocked:
    def test_assign_small(self, small_feats):
        """50 points, 5 centroids — assign_blocked matches brute force."""
        feats, centroids = small_feats
        expected = _brute_force_assign(feats, centroids)
        result = assign_blocked(feats, centroids, block_size=16)
        assert torch.equal(result, expected), (
            f"assign_blocked mismatch: {(result != expected).sum()} / {len(result)} differ"
        )

    def test_assign_gpu(self, small_feats):
        """Same test on GPU if available, else skip."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        feats, centroids = small_feats
        feats_gpu = feats.cuda()
        centroids_gpu = centroids.cuda()
        expected = _brute_force_assign(feats_gpu, centroids_gpu)
        result = assign_blocked(feats_gpu, centroids_gpu, block_size=16)
        assert torch.equal(result, expected)

    def test_assign_empty_input(self):
        """Empty feature tensor raises ValueError."""
        feats = torch.zeros(0, 8)
        centroids = torch.randn(3, 8)
        with pytest.raises(ValueError):
            assign_blocked(feats, centroids)

    def test_assign_dimension_mismatch(self):
        """Dimension mismatch between feats and centroids raises ValueError."""
        feats = torch.randn(10, 8)
        centroids = torch.randn(3, 16)
        with pytest.raises(ValueError):
            assign_blocked(feats, centroids)

    def test_assign_single_centroid(self):
        """With a single centroid, all points are assigned to cluster 0."""
        torch.manual_seed(42)
        feats = torch.randn(20, 4)
        centroids = torch.randn(1, 4)
        result = assign_blocked(feats, centroids)
        assert result.shape == (20,)
        assert (result == 0).all(), "All assignments should be 0 with 1 centroid"

    def test_assign_deterministic(self, small_feats):
        """Same inputs produce identical outputs on repeated calls."""
        feats, centroids = small_feats
        r1 = assign_blocked(feats, centroids)
        r2 = assign_blocked(feats, centroids)
        assert torch.equal(r1, r2)

    def test_assign_known_centroids(self):
        """Points that ARE the centroids should be assigned to themselves."""
        torch.manual_seed(0)
        centroids = torch.eye(4)  # 4 orthogonal unit vectors
        # Each centroid is its own nearest neighbour
        result = assign_blocked(centroids, centroids)
        expected = torch.arange(4)
        assert torch.equal(result, expected)


class TestComputeMSE:
    def test_mse_nonnegative(self, small_feats):
        """MSE is always >= 0."""
        feats, centroids = small_feats
        assignments = assign_blocked(feats, centroids)
        mse = compute_mse(feats, centroids, assignments)
        assert mse >= 0.0, f"MSE should be non-negative, got {mse}"

    def test_mse_zero(self):
        """When every point equals its assigned centroid, MSE == 0."""
        torch.manual_seed(1)
        k, d = 5, 8
        centroids = torch.randn(k, d)
        # Build feats as exact copies of centroids (each repeated 10 times)
        feats = centroids.repeat_interleave(10, dim=0)   # (50, 8)
        assignments = torch.arange(k).repeat_interleave(10)  # (50,)
        mse = compute_mse(feats, centroids, assignments)
        assert mse < 1e-6, f"MSE should be ~0 when points == centroids, got {mse}"

    def test_mse_scalar(self, small_feats):
        """compute_mse returns a Python float or 0-d tensor."""
        feats, centroids = small_feats
        assignments = assign_blocked(feats, centroids)
        mse = compute_mse(feats, centroids, assignments)
        # Should be convertible to float
        float(mse)


class TestValidateAssignments:
    def test_validate_valid(self):
        """Valid assignments pass without error."""
        assignments = torch.tensor([0, 1, 2, 0, 1])
        k = 3
        validate_assignments(assignments, k)  # should not raise

    def test_validate_raises_out_of_range(self):
        """Assignment index >= k raises ValueError."""
        assignments = torch.tensor([0, 1, 5])  # 5 >= k=3
        with pytest.raises(ValueError):
            validate_assignments(assignments, k=3)

    def test_validate_raises_negative(self):
        """Negative assignment index raises ValueError."""
        assignments = torch.tensor([-1, 0, 1])
        with pytest.raises(ValueError):
            validate_assignments(assignments, k=3)
