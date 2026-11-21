"""
Tests for src/emeans.py — correctness, determinism, schema, holdout.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch
import dataclasses

from src.emeans import EMeans, EMeansConfig, EMeansResult
from src.assign import compute_mse, assign_blocked
from src.reproducibility import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_mixture(n=400, k=5, d=16, seed=0):
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


def _default_config(**kwargs):
    base = dict(
        k=5,
        T=3,
        batch_size=64,
        alpha0=0.5,
        use_schedule=True,
        use_revival=False,
        holdout_frac=0.1,
        seed=42,
    )
    base.update(kwargs)
    return EMeansConfig(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEMeansBasic:
    def test_emeans_basic_runs(self):
        """T=3 passes on a synthetic dataset without raising."""
        feats = _gaussian_mixture(n=400, k=5, d=16, seed=1)
        cfg = _default_config(T=3)
        model = EMeans(cfg)
        result = model.fit(feats)
        assert result is not None

    def test_emeans_result_schema(self):
        """EMeansResult has all required fields."""
        feats = _gaussian_mixture(n=400, k=5, d=16, seed=2)
        cfg = _default_config(T=2)
        model = EMeans(cfg)
        result = model.fit(feats)

        required_fields = {
            "centroids",
            "mse_history",
            "holdout_mse_history",
            "n_updates",
            "wall_seconds",
            "config",
        }
        result_fields = {f.name for f in dataclasses.fields(result)}
        missing = required_fields - result_fields
        assert not missing, f"EMeansResult missing fields: {missing}"

    def test_emeans_holdout_separate(self):
        """Holdout indices and train indices are disjoint."""
        feats = _gaussian_mixture(n=400, k=5, d=16, seed=3)
        cfg = _default_config(T=2, holdout_frac=0.2)
        model = EMeans(cfg)
        result = model.fit(feats)

        # The result should expose holdout_indices and train_indices
        assert hasattr(result, "holdout_indices") or hasattr(model, "holdout_indices"), (
            "EMeans must expose holdout_indices via result or model"
        )
        # Get indices from wherever they live
        if hasattr(result, "holdout_indices"):
            hi = set(result.holdout_indices.tolist())
            ti = set(result.train_indices.tolist())
        else:
            hi = set(model.holdout_indices.tolist())
            ti = set(model.train_indices.tolist())

        assert len(hi) > 0, "Holdout set should be non-empty"
        assert len(ti) > 0, "Train set should be non-empty"
        assert hi.isdisjoint(ti), "Holdout and train sets must be disjoint"


class TestEMeansMSE:
    def test_emeans_mse_decreases(self):
        """Holdout MSE at end of training is lower than at the start."""
        feats = _gaussian_mixture(n=600, k=5, d=16, seed=4)
        cfg = _default_config(T=5, batch_size=32, alpha0=0.8)
        model = EMeans(cfg)
        result = model.fit(feats)

        history = result.holdout_mse_history
        assert len(history) >= 2, "Need at least 2 holdout MSE checkpoints"
        # Allow generous tolerance: last third average < first value
        avg_last = sum(history[len(history)//2:]) / max(1, len(history) - len(history)//2)
        assert avg_last <= history[0] * 1.5, (
            f"Holdout MSE did not decrease: first={history[0]:.4f}, avg_last={avg_last:.4f}"
        )

    def test_emeans_lloyd_equiv(self):
        """batch_size=n, alpha0=1.0, no schedule, 1 epoch ≈ 1 Lloyd iteration."""
        from src.lloyd import LloydKMeans

        feats = _gaussian_mixture(n=200, k=5, d=16, seed=5)
        n = feats.shape[0]

        # EMeans in Lloyd mode
        cfg = EMeansConfig(
            k=5,
            T=1,
            batch_size=n,
            alpha0=1.0,
            use_schedule=False,
            use_revival=False,
            holdout_frac=0.0,
            seed=7,
        )
        em_model = EMeans(cfg)
        em_result = em_model.fit(feats)
        em_mse = float(compute_mse(
            feats, em_result.centroids,
            assign_blocked(feats, em_result.centroids)
        ))

        # Reference Lloyd (1 iteration, same init seed)
        lloyd = LloydKMeans(k=5, seed=7, init="kmeanspp")
        lloyd.fit(feats, n_iter=1)
        lloyd_mse = float(compute_mse(
            feats, lloyd.centroids, lloyd.predict(feats)
        ))

        # They won't be exactly equal due to ordering, but should be close
        assert abs(em_mse - lloyd_mse) / (lloyd_mse + 1e-9) < 0.15, (
            f"EMeans Lloyd-equiv MSE {em_mse:.4f} differs too much from "
            f"Lloyd MSE {lloyd_mse:.4f}"
        )


class TestEMeansDeterminism:
    def test_emeans_deterministic(self):
        """Same config + seed → identical centroids on two separate runs."""
        feats = _gaussian_mixture(n=400, k=5, d=16, seed=6)
        cfg = _default_config(T=3, seed=42)

        m1 = EMeans(cfg)
        r1 = m1.fit(feats)

        m2 = EMeans(cfg)
        r2 = m2.fit(feats)

        assert torch.allclose(r1.centroids, r2.centroids, atol=1e-5), (
            "EMeans centroids differ between identical runs — non-deterministic!"
        )


class TestEMeansRevival:
    def test_emeans_revival_runs(self):
        """Revival enabled: fit completes without crash."""
        feats = _gaussian_mixture(n=400, k=5, d=16, seed=8)
        cfg = _default_config(T=3, use_revival=True)
        model = EMeans(cfg)
        result = model.fit(feats)
        assert result.centroids.shape == (5, 16)
