"""
metrics.py
==========
Clustering and selection quality metrics, plus result persistence via Parquet.

Public API
----------
compute_nmi(true_labels, pred_labels) -> float
compute_class_coverage(true_labels, selected_indices, n_classes) -> float
ResultRecord                – full result schema dataclass
save_results(records, path) -> None   (atomic append to Parquet)
load_results(path) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def compute_nmi(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """
    Compute Normalised Mutual Information (NMI) between ground-truth
    class labels and predicted cluster assignments.

    Parameters
    ----------
    true_labels:
        ``(N,)`` integer ground-truth class labels.
    pred_labels:
        ``(N,)`` integer predicted cluster assignments.
    average_method:
        Averaging method passed to scikit-learn
        (``"arithmetic"``, ``"geometric"``, or ``"min"``).

    Returns
    -------
    NMI in [0, 1].
    """
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"true_labels length {len(true_labels)} != pred_labels length {len(pred_labels)}."
        )
    nmi = float(
        normalized_mutual_info_score(true_labels, pred_labels, average_method=average_method)
    )
    return nmi


def compute_class_coverage(
    true_labels: np.ndarray,
    selected_indices: np.ndarray,
    n_classes: int | None = None,
) -> float:
    """
    Compute the fraction of unique classes represented in the selected subset.

    ``coverage = |unique classes in selected| / n_classes``

    Parameters
    ----------
    true_labels:
        ``(N,)`` integer labels for the full pool.
    selected_indices:
        ``(B,)`` integer indices of selected samples into ``true_labels``.
    n_classes:
        Total number of classes. If None, inferred as ``len(unique(true_labels))``.

    Returns
    -------
    Class coverage in [0, 1].
    """
    true_labels = np.asarray(true_labels, dtype=np.int64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    if n_classes is None:
        n_classes = len(np.unique(true_labels))

    selected_labels = true_labels[selected_indices]
    n_covered = len(np.unique(selected_labels))
    return float(n_covered) / float(n_classes)


# ---------------------------------------------------------------------------
# ResultRecord
# ---------------------------------------------------------------------------

@dataclass
class ResultRecord:
    """
    Full result schema for one experimental run.

    All fields are stored as Parquet-compatible types (Python scalars, lists,
    or None). List fields are serialised to JSON strings in Parquet.

    Attributes
    ----------
    experiment:
        Experiment name (e.g. ``"cluster_bench"``).
    method:
        Method name (e.g. ``"EMeans-T3"``).
    seed:
        Random seed used.
    k:
        Number of clusters.
    n_train:
        Training pool size.
    n_valid:
        Validation set size.
    B:
        Selection budget (number of labelled samples). None if not a selection exp.
    T:
        e-means epoch multiplier. None if not applicable.
    batch_size:
        Mini-batch size. None if not applicable.
    alpha0:
        Initial EMA learning rate. None if not applicable.
    n_updates:
        Number of update steps. None if not applicable.
    revival_enabled:
        Whether dead-cluster revival was enabled.
    nmi:
        Normalised Mutual Information (clustering vs. true labels).
    class_coverage:
        Fraction of classes covered by the selected subset.
    logistic_valid_acc:
        Validation accuracy of the logistic probe.
    cosine_knn_valid_acc:
        Validation accuracy of cosine k-NN.
    final_mse_train:
        Final training MSE.
    final_mse_holdout:
        Final holdout MSE.
    mse_history:
        List of per-step holdout MSEs. Stored as JSON string in Parquet.
    wall_time_s:
        Wall-clock time in seconds for the full method.
    gpu_name:
        GPU name, or ``"cpu"`` if CPU-only.
    extra:
        Arbitrary extra metadata (dict). Stored as JSON string in Parquet.
    """

    experiment: str
    method: str
    seed: int
    k: int
    n_train: int
    n_valid: int
    B: int | None = None
    T: int | None = None
    batch_size: int | None = None
    alpha0: float | None = None
    n_updates: int | None = None
    revival_enabled: bool | None = None
    nmi: float | None = None
    class_coverage: float | None = None
    logistic_valid_acc: float | None = None
    cosine_knn_valid_acc: float | None = None
    final_mse_train: float | None = None
    final_mse_holdout: float | None = None
    mse_history: list[float] | None = None
    wall_time_s: float | None = None
    gpu_name: str | None = None
    extra: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for a Parquet row (lists → JSON strings)."""
        import json
        d = asdict(self)
        # Serialise list / dict fields to JSON strings for Parquet compatibility
        for key in ("mse_history", "extra"):
            if d[key] is not None:
                d[key] = json.dumps(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResultRecord":
        """Reconstruct a ``ResultRecord`` from a Parquet row dict."""
        import json
        d = dict(d)
        for key in ("mse_history", "extra"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = None
        # Remove any extra columns not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------

def save_results(
    records: list[ResultRecord],
    path: str | Path,
    append: bool = True,
) -> None:
    """
    Save a list of :class:`ResultRecord` to a Parquet file.

    If ``append=True`` and the file exists, existing rows are loaded,
    new rows are appended, and the whole table is written atomically.

    Parameters
    ----------
    records:
        Results to save.
    path:
        Destination ``.parquet`` file path.
    append:
        If True, append to existing file. If False, overwrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([r.to_dict() for r in records])

    if append and path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    # Atomic write via temp file
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Saved %d records to %s (total %d rows)", len(records), path, len(df))


def load_results(path: str | Path) -> pd.DataFrame:
    """
    Load results from a Parquet file.

    Parameters
    ----------
    path:
        Path to ``.parquet`` file.

    Returns
    -------
    ``pd.DataFrame`` with one row per result.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
