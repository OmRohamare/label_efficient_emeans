"""
tests/test_laws.py
==================
Unit tests for the e-means scaling laws (src/laws.py).
Verifies:
  - Batch size strictly increases with k (raw) and non-decreasingly (power-of-2).
  - Alpha0 decreases as k increases (given fixed batch size).
  - Batch size and alpha0 match expected references from Meta's Faiss OLS fits.
  - Rounding to power of two and clamping behavior.
"""

import math
import pytest
from src.laws import (
    compute_batch_size,
    compute_alpha0,
    compute_num_updates,
    round_power_of_two,
    ScalingLawConfig,
    _BATCH_LAWS,
    _ALPHA_LAW,
)


class TestLawsBatchSize:
    def test_batch_size_increases_with_k(self):
        """Verify that batch size is NOT constant across k (regression test for k-cancellation bug)."""
        n, T = 100_000, 3
        k_values = [50, 200, 1024, 1265, 5000, 10000]
        batch_sizes = [compute_batch_size(n=n, k=k, T=T) for k in k_values]

        # Verify batch sizes are not all identical
        assert len(set(batch_sizes)) > 1, f"Batch size must vary with k, got {batch_sizes}"

        # Verify monotonic non-decreasing
        for i in range(len(batch_sizes) - 1):
            assert batch_sizes[i] <= batch_sizes[i + 1]

        # Specifically, k=5000 should require a larger batch than k=200
        assert compute_batch_size(n=n, k=5000, T=T) > compute_batch_size(n=n, k=200, T=T)

    def test_raw_batch_size_strictly_increases_with_k(self):
        """Verify raw power law calculation strictly increases with k."""
        b_c, b_k, b_n, b_T = _BATCH_LAWS["throughput"]
        n, T = 100_000, 3
        raw_b200 = b_c * (200 ** b_k) * (n ** b_n) * (T ** b_T)
        raw_b1265 = b_c * (1265 ** b_k) * (n ** b_n) * (T ** b_T)
        raw_b5000 = b_c * (5000 ** b_k) * (n ** b_n) * (T ** b_T)

        assert raw_b200 < raw_b1265 < raw_b5000
        assert math.isclose(raw_b200, 4229.2, rel_tol=1e-2)
        assert math.isclose(raw_b1265, 5422.3, rel_tol=1e-2)

    def test_batch_size_power_of_two(self):
        """Verify computed batch sizes are powers of two."""
        for k in [100, 500, 1265, 2000]:
            b = compute_batch_size(n=50_000, k=k, T=3)
            assert (b & (b - 1)) == 0, f"Batch size {b} is not a power of 2"

    def test_batch_size_quality_law(self):
        """Verify quality law gives smaller batch sizes (knee / 8)."""
        b_throughput = compute_batch_size(n=100_000, k=1000, T=3, law="throughput")
        b_quality = compute_batch_size(n=100_000, k=1000, T=3, law="quality")
        assert b_quality < b_throughput


class TestLawsAlpha0:
    def test_alpha0_decreases_with_k(self):
        """For fixed batch size, alpha0 should decrease as k increases."""
        n, T, B = 100_000, 3, 4096
        a200 = compute_alpha0(n=n, k=200, T=T, batch_size=B)
        a1265 = compute_alpha0(n=n, k=1265, T=T, batch_size=B)
        a5000 = compute_alpha0(n=n, k=5000, T=T, batch_size=B)

        assert a200 > a1265 > a5000
        assert 0.0 < a1265 < 1.0

    def test_alpha0_clamping(self):
        """Alpha0 must be clamped to [1e-4, 1.0]."""
        # Very small k might produce raw alpha > 1.0
        a_small = compute_alpha0(n=100_000, k=10, T=3, batch_size=4096)
        assert a_small <= 1.0
        assert a_small >= 1e-4


class TestScalingLawConfig:
    def test_scaling_law_config_auto(self):
        cfg = ScalingLawConfig(n=100_000, k=1265, T=3)
        assert cfg.batch_size == 4096
        assert cfg.num_updates == 74
        assert 0.25 < cfg.alpha0 < 0.30
        assert cfg.alpha_min == pytest.approx(cfg.alpha0 * 0.1)

    def test_scaling_law_config_manual_override(self):
        cfg = ScalingLawConfig(n=100_000, k=200, T=3, batch_size=1024, alpha0=0.215)
        assert cfg.batch_size == 1024
        assert cfg.alpha0 == pytest.approx(0.215)
        assert cfg.num_updates == 293
