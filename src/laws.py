"""
laws.py
=======
Scaling laws for the e-means algorithm (arXiv:2611.09501, Meta FAIR).

Verified against the e-means paper PDF (arXiv:2611.09501v1, Table 2 and Section 3.3)
and the official Faiss implementation (faiss.contrib.e_means.base).

================================================================================
EXPLANATION OF THE PREVIOUS laws.py BUG
================================================================================
In earlier unverified versions of this file, the mini-batch size was estimated as:

    B ≈ C_B * (k ** 0.5) * ((n / k) ** 0.5) * (T ** -0.5)

Notice the algebraic identity:
    (k ** 0.5) * ((n / k) ** 0.5) = k^0.5 * (n^0.5 / k^0.5) = n^0.5 = sqrt(n)

Because the exponent on k (0.5) and the exponent on the ratio (n/k) (0.5) were
identical, k cancelled out completely! As a result, for any dataset size n and epoch
budget T, the predicted batch size was strictly constant across all values of k:
    B = 50 * sqrt(100,000) * (3 ** -0.5) ≈ 9129  (identical for k=200, 500, 1024, etc.)
An unexplained identical batch size across varying k values is an algebraic artifact
from an incorrect formulation.

================================================================================
VERIFIED SCALING LAWS (arXiv:2611.09501, Section 3.3 & Table 2)
================================================================================
1. Batch Size Law (Equation 7):
   B = b_c * (k ** b_k) * (n ** b_n) * (T ** b_T)
   rounded to the nearest power of two and capped at n.
   
   - Throughput law (knee at +0.2% excess validation MSE):
       b_c = 0.035134, b_k = 0.134749, b_n = 0.876473, b_T = 0.813344
       (Paper Table 2: b_c=0.035, b_k=0.13, b_n=0.88, b_T=0.81)
   - Quality law (used for low dimension d <= 16 or large scales n > 2M; knee / 8):
       b_c = 0.00439175, b_k = 0.134749, b_n = 0.876473, b_T = 0.813344
       (Paper Table 2: b_c=0.0044, b_k=0.13, b_n=0.88, b_T=0.81)

   Crucially, b_k = +0.13 > 0, so batch size increases sublinearly with k!

2. Decay Rate Law (Equation 6):
   alpha0 = min(a_c * (k ** a_k) * (B ** a_B) * (n ** a_n) * (T ** a_T), 1.0)
   
   - Fitted coefficients (Table 2 & Faiss):
       a_c = 0.8244, a_k = -0.4617, a_B = 0.9003, a_n = -0.4498, a_T = -0.1057
       (Paper Table 2: a_c=0.824, a_k=-0.46, a_B=0.90, a_n=-0.45, a_T=-0.11)
   The initial EMA learning rate alpha0 directly couples to the batch size B
   (a_B = +0.90), compensating for larger mini-batches by moving further per step.
   Clamped to [1e-4, 1.0].

3. Cosine Annealing Schedule (Equation 5):
   alpha(t) = alpha0 * (eps + 0.5 * (1 - eps) * (1 + cos(pi * t / M)))
   where eps = 0.1 (decays to 10% of initial alpha0 over M total steps).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verified Scaling-Law Coefficients (arXiv:2611.09501, Table 2 & Faiss)
# ---------------------------------------------------------------------------

# Batch law coefficients: (b_c, b_k, b_n, b_T)
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Alpha0 step law coefficients: (a_c, a_k, a_B, a_n, a_T)
_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)

# Floor factor for cosine annealing (Equation 5, eps = 0.1)
_ALPHA_MIN_FACTOR = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_power_of_two(value: float) -> int:
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


# ---------------------------------------------------------------------------
# ScalingLawConfig
# ---------------------------------------------------------------------------

@dataclass
class ScalingLawConfig:
    """
    Container for verified scaling-law parameters.

    Parameters
    ----------
    n:
        Dataset size.
    k:
        Number of clusters.
    T:
        Number of passes (epoch multiplier).
    batch_size:
        Mini-batch size (computed if None).
    alpha0:
        Initial EMA learning rate (computed if None).
    num_updates:
        Total number of EMA update steps (computed if None).
    alpha_min_factor:
        Cosine schedule end factor: alpha_min = alpha0 * alpha_min_factor. Default 0.1.
    batch_law:
        'throughput' (default) or 'quality' (knee/8).
    """

    n: int
    k: int
    T: int
    batch_size: int | None = None
    alpha0: float | None = None
    num_updates: int | None = None
    alpha_min_factor: float = _ALPHA_MIN_FACTOR
    batch_law: Literal["throughput", "quality"] = "throughput"

    def __post_init__(self) -> None:
        if self.batch_size is None:
            self.batch_size = compute_batch_size(self.n, self.k, self.T, law=self.batch_law)
        if self.num_updates is None:
            self.num_updates = compute_num_updates(self.n, self.k, self.T, self.batch_size)
        if self.alpha0 is None:
            self.alpha0 = compute_alpha0(self.n, self.k, self.T, batch_size=self.batch_size)
        self.alpha0 = float(min(1.0, max(1e-4, self.alpha0)))

    @property
    def alpha_min(self) -> float:
        """Minimum alpha on the cosine schedule."""
        return self.alpha0 * self.alpha_min_factor  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Scaling-law functions
# ---------------------------------------------------------------------------

def compute_batch_size(
    n: int,
    k: int,
    T: int,
    law: Literal["throughput", "quality"] = "throughput",
) -> int:
    """
    Compute recommended mini-batch size B from the verified scaling law (Eq. 7).

    B = b_c * (k ** b_k) * (n ** b_n) * (T ** b_T)
    rounded to the nearest power of two and capped at n.

    Parameters
    ----------
    n:
        Dataset size.
    k:
        Number of clusters.
    T:
        Epoch multiplier (number of passes).
    law:
        'throughput' (knee at 0.2%) or 'quality' (knee / 8).

    Returns
    -------
    Batch size (int), rounded to nearest power of 2, clipped to [1, n].
    """
    if law not in _BATCH_LAWS:
        raise ValueError(f"Unknown law '{law}'. Choose 'throughput' or 'quality'.")
    b_c, b_k, b_n, b_T = _BATCH_LAWS[law]
    raw_B = b_c * (k ** b_k) * (n ** b_n) * (T ** b_T)
    B = round_power_of_two(raw_B)
    B = max(1, min(B, n))
    logger.debug(
        "compute_batch_size(n=%d, k=%d, T=%d, law=%s) -> raw=%.1f, power_of_2=%d",
        n, k, T, law, raw_B, B,
    )
    return B


def compute_alpha0(
    n: int,
    k: int,
    T: int,
    batch_size: int | None = None,
    law: Literal["throughput", "quality"] = "throughput",
) -> float:
    """
    Compute recommended initial EMA learning rate alpha0 from verified scaling law (Eq. 6).

    alpha0 = min(a_c * (k ** a_k) * (B ** a_B) * (n ** a_n) * (T ** a_T), 1.0)

    Parameters
    ----------
    n:
        Dataset size.
    k:
        Number of clusters.
    T:
        Epoch multiplier.
    batch_size:
        Mini-batch size. If None, computed from compute_batch_size(n, k, T, law=law).
    law:
        Batch law used if batch_size is None.

    Returns
    -------
    alpha0 (float), clipped to [1e-4, 1.0].
    """
    if batch_size is None:
        batch_size = compute_batch_size(n, k, T, law=law)

    a_c, a_k, a_B, a_n, a_T = _ALPHA_LAW
    raw_alpha0 = a_c * (k ** a_k) * (batch_size ** a_B) * (n ** a_n) * (T ** a_T)
    alpha0 = float(min(1.0, max(1e-4, raw_alpha0)))
    logger.debug(
        "compute_alpha0(n=%d, k=%d, T=%d, B=%d) -> raw=%.6f, clipped=%.6f",
        n, k, T, batch_size, raw_alpha0, alpha0,
    )
    return alpha0


def compute_num_updates(n: int, k: int, T: int, batch_size: int) -> int:
    """
    Compute total number of EMA mini-batch update steps.

    num_updates = ceil(T * n / batch_size)
    """
    num_updates = math.ceil(T * n / batch_size)
    return max(1, num_updates)


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------

def cosine_schedule(
    alpha0: float,
    step: int,
    total_steps: int,
    alpha_min_factor: float = _ALPHA_MIN_FACTOR,
) -> float:
    """
    Cosine learning-rate schedule (Equation 5):
    alpha(t) = alpha0 * (alpha_min_factor + (1 - alpha_min_factor) * 0.5 * (1 + cos(pi * t / (M-1))))
    """
    if total_steps <= 1:
        return alpha0
    t = max(0, min(step, total_steps - 1))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t / (total_steps - 1)))
    return alpha0 * (alpha_min_factor + (1.0 - alpha_min_factor) * cosine_factor)
