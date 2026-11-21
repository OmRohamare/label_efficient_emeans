#!/usr/bin/env python3
"""
experiments/02_selection_sweep.py
===================================
Selection budget sweep: compare label selection strategies across budget sizes.

Usage
-----
    python experiments/02_selection_sweep.py --smoke-test
    python experiments/02_selection_sweep.py --config configs/experiments.yaml \
        --cache-dir cache --results-dir results --seeds 0 1 2 3 4

Arguments
---------
--config        Path to experiments YAML config
--cache-dir     Directory containing train_feats.npy, train_labels.npy, val_feats.npy, val_labels.npy
--results-dir   Where to save selection_sweep.parquet
--seeds         List of integer seeds (default: 0–9)
--dry-run       Print combos without running
--smoke-test    B=[200,500], 1 seed, fast
"""

import sys
import os
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pandas as pd

from src.utils import load_config, get_device, save_atomic, setup_logging
from src.assign import assign_blocked, compute_mse
from src.lloyd import LloydKMeans
from src.emeans import EMeans, EMeansConfig
from src.select import (
    select_nearest_to_centroids,
    select_random,
    select_kmeanspp,
    SelectionResult,
)
from src.probe import CosineKNN, evaluate_selection
from src.metrics import ResultRecord, save_results
from src.reproducibility import seed_everything

logger = setup_logging("selection_sweep")

FULL_BUDGETS = [200, 500, 1000, 2000, 5000]
SMOKE_BUDGETS = [200, 500]
FULL_SEEDS = list(range(10))
SMOKE_SEEDS = [0]


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def _lloyd_centroids(feats: torch.Tensor, k: int, n_iter: int, seed: int) -> torch.Tensor:
    model = LloydKMeans(k=k, seed=seed)
    model.fit(feats, n_iter=n_iter)
    return model.centroids


def _emeans_centroids(feats: torch.Tensor, k: int, T: int, seed: int,
                      revival: bool = True) -> torch.Tensor:
    cfg = EMeansConfig(
        k=k, T=T,
        alpha0=None, batch_size=None,
        use_schedule=True, use_revival=revival,
        holdout_frac=0.0, seed=seed,
    )
    result = EMeans(cfg).fit(feats)
    return result.centroids


def _official_emeans_centroids(feats: torch.Tensor, k: int, T: int, seed: int) -> torch.Tensor:
    from faiss.contrib.e_means import NumpyEmeans
    x = feats.detach().cpu().numpy().astype("float32")
    n, d = x.shape
    em = NumpyEmeans(d, k, nepochs=T, seed=seed)
    em.train(x)
    return torch.from_numpy(em.centroids).to(device=feats.device, dtype=feats.dtype)


def select_with_method(
    method: str,
    feats: torch.Tensor,
    k: int,
    B: int,
    seed: int,
) -> SelectionResult:
    seed_everything(seed)
    if method == "Random":
        return select_random(feats, B=B, seed=seed)
    elif method == "KMeanspp":
        return select_kmeanspp(feats, B=B, seed=seed)
    elif method == "Lloyd-3":
        centroids = _lloyd_centroids(feats, k=k, n_iter=3, seed=seed)
        return select_nearest_to_centroids(feats, centroids, B=B, seed=seed)
    elif method == "Lloyd-25":
        centroids = _lloyd_centroids(feats, k=k, n_iter=25, seed=seed)
        return select_nearest_to_centroids(feats, centroids, B=B, seed=seed)
    elif method == "EMeans-T3":
        centroids = _emeans_centroids(feats, k=k, T=3, seed=seed, revival=True)
        return select_nearest_to_centroids(feats, centroids, B=B, seed=seed)
    elif method == "EMeans-T3-NoRevival":
        centroids = _emeans_centroids(feats, k=k, T=3, seed=seed, revival=False)
        return select_nearest_to_centroids(feats, centroids, B=B, seed=seed)
    elif method == "Official-EMeans-T3":
        centroids = _official_emeans_centroids(feats, k=k, T=3, seed=seed)
        return select_nearest_to_centroids(feats, centroids, B=B, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")


METHODS = [
    "Random",
    "KMeanspp",
    "Lloyd-3",
    "Lloyd-25",
    "EMeans-T3",
    "EMeans-T3-NoRevival",
    "Official-EMeans-T3",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cache_dir: Path, smoke_test: bool, device: torch.device):
    def _load(name):
        path = cache_dir / name
        if not path.exists() and name.startswith("val_"):
            path = cache_dir / name.replace("val_", "valid_", 1)
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        return torch.from_numpy(np.load(path)).float()

    train_feats = _load("train_feats.npy")
    train_labels = _load("train_labels.npy").long()
    val_feats = _load("val_feats.npy")
    val_labels = _load("val_labels.npy").long()

    if smoke_test:
        # Sample across the entire dataset, not only the first classes.
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(
            len(train_feats), generator=generator
        )[:5000]
        train_feats = train_feats[indices]
        train_labels = train_labels[indices]

    return (
        train_feats.to(device),
        train_labels.to(device),
        val_feats.to(device),
        val_labels.to(device),
    )


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    df = pd.read_parquet(results_path)
    return set(zip(df["method"], df["budget"], df["seed"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def flush_rows(rows: list, results_path: Path) -> None:
    """Append rows to the parquet atomically, then clear the buffer."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if results_path.exists():
        new_df = pd.concat([pd.read_parquet(results_path), new_df],
                           ignore_index=True)
    tmp = results_path.with_name(f"{results_path.stem}.tmp.parquet")
    new_df.to_parquet(tmp, index=False)
    save_atomic(tmp, results_path)
    rows.clear()


def parse_args():
    p = argparse.ArgumentParser(description="Selection strategy budget sweep")
    p.add_argument("--config", default="configs/experiments.yaml")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "selection_sweep.parquet"

    device = get_device()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    k = cfg.get("k", 200)

    budgets = SMOKE_BUDGETS if args.smoke_test else FULL_BUDGETS
    seeds = SMOKE_SEEDS if args.smoke_test else (args.seeds or FULL_SEEDS)
    methods = args.methods if args.methods is not None else METHODS

    try:
        train_feats, train_labels, val_feats, val_labels = load_data(
            cache_dir, args.smoke_test, device=torch.device("cpu")
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Train: %s  Val: %s  k=%d", train_feats.shape, val_feats.shape, k)

    completed = load_completed(results_path)
    rows = []
    total = len(methods) * len(budgets) * len(seeds)
    done = 0

    for method in methods:
        for B in budgets:
            for seed in seeds:
                done += 1
                key = (method, B, seed)
                if key in completed:
                    logger.info("[%d/%d] SKIP %s B=%d seed=%d", done, total,
                                method, B, seed)
                    continue
                if args.dry_run:
                    logger.info("[%d/%d] DRY-RUN %s B=%d seed=%d", done, total,
                                method, B, seed)
                    continue

                logger.info("[%d/%d] %s B=%d seed=%d ...", done, total,
                            method, B, seed)
                t0 = time.perf_counter()
                try:
                    result = select_with_method(
                        method, train_feats, k=B, B=B, seed=seed
                    )
                    # evaluate_selection expects full training arrays
                    # and indices into those arrays.
                    import numpy as np

                    train_feats_np = (
                        train_feats.detach().cpu().numpy()
                        if hasattr(train_feats, "detach")
                        else np.asarray(train_feats)
                    )
                    train_labels_np = (
                        train_labels.detach().cpu().numpy()
                        if hasattr(train_labels, "detach")
                        else np.asarray(train_labels)
                    )
                    val_feats_np = (
                        val_feats.detach().cpu().numpy()
                        if hasattr(val_feats, "detach")
                        else np.asarray(val_feats)
                    )
                    val_labels_np = (
                        val_labels.detach().cpu().numpy()
                        if hasattr(val_labels, "detach")
                        else np.asarray(val_labels)
                    )
                    sel_idx = result.indices.detach().cpu().numpy().astype(np.int64)

                    eval_result = evaluate_selection(
                        train_feats_np,
                        train_labels_np,
                        val_feats_np,
                        val_labels_np,
                        selected_indices=sel_idx,
                        method=method,
                        n_classes=int(train_labels_np.max()) + 1,
                    )
                    wall = time.perf_counter() - t0

                    row = {
                        "method": method,
                        "budget": B,
                        "seed": seed,
                        "k": B,
                        "knn_acc": eval_result.cosine_knn_valid_acc,
                        "logistic_valid_acc": eval_result.logistic_valid_acc,
                        "class_coverage": float(
                            len(set(train_labels_np[sel_idx].tolist()))
                            / max(1, int(train_labels_np.max()) + 1)
                        ),
                        "wall_seconds": wall,
                    }
                    rows.append(row)
                    flush_rows(rows, results_path)
                    logger.info("  knn_acc=%.3f  wall=%.1fs",
                                row["knn_acc"], wall)
                except Exception as e:
                    logger.error("  FAILED %s B=%d seed=%d: %s",
                                 method, B, seed, e)

    flush_rows(rows, results_path)

    # Print summary table
    if results_path.exists():
        df = pd.read_parquet(results_path)
        summary = (
            df.groupby(["method", "budget"])["knn_acc"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "KNN_acc_mean", "std": "KNN_acc_std"})
            .sort_values(["budget", "KNN_acc_mean"], ascending=[True, False])
        )
        print("\n=== Selection Sweep Summary ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
