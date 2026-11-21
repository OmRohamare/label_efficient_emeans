"""
reproducibility.py
==================
Determinism utilities: seed_everything() and get_env_info().
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Set all random seeds to ``seed`` for reproducibility.

    Sets:
      - Python built-in ``random``
      - ``numpy``
      - ``torch`` (CPU + CUDA)
      - PYTHONHASHSEED environment variable
      - CuDNN deterministic / benchmark flags

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # CuDNN: deterministic ops, disable autotuner
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_hash() -> str:
    """Return current git HEAD hash, or 'N/A' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "N/A"


def get_env_info() -> dict[str, Any]:
    """
    Collect environment metadata for reproducibility logging.

    Returns
    -------
    dict with keys:
        python_version, platform, torch_version, cuda_available,
        cuda_version, cudnn_version, numpy_version, git_hash,
        cpu_count, gpu_names.
    """
    gpu_names: list[str] = []
    cuda_version = "N/A"
    cudnn_version = "N/A"

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda or "N/A"
        cudnn_version = str(torch.backends.cudnn.version())
        gpu_names = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "numpy_version": np.__version__,
        "git_hash": _git_hash(),
        "cpu_count": os.cpu_count(),
        "gpu_names": gpu_names,
    }
