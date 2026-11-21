"""
Shared pytest fixtures for label_efficient_emeans tests.
"""
import sys
import os

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `import src.*` works regardless
# of how pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_feats():
    """
    Gaussian mixture with n=500 points, k=10 clusters, d=16 dimensions.
    Returns (feats, labels, centroids_gt) all as torch.Tensors (float32).
    feats are L2-normalised.
    """
    torch.manual_seed(0)
    n, k, d = 500, 10, 16
    n_per_cluster = n // k

    means = torch.randn(k, d)
    means = means / means.norm(dim=1, keepdim=True)  # unit-sphere means

    feats_list, labels_list = [], []
    for i in range(k):
        pts = means[i] + 0.1 * torch.randn(n_per_cluster, d)
        feats_list.append(pts)
        labels_list.append(torch.full((n_per_cluster,), i, dtype=torch.long))

    feats = torch.cat(feats_list, dim=0)  # (500, 16)
    labels = torch.cat(labels_list, dim=0)  # (500,)

    # L2-normalise
    feats = feats / feats.norm(dim=1, keepdim=True)

    return feats, labels, means


@pytest.fixture(scope="session")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA not available")


@pytest.fixture
def small_feats():
    """50 points, 5 centroids, d=8 — fast for unit tests."""
    torch.manual_seed(7)
    n, k, d = 50, 5, 8
    feats = torch.randn(n, d)
    feats = feats / feats.norm(dim=1, keepdim=True)
    centroids = torch.randn(k, d)
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    return feats, centroids
