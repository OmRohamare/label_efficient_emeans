#!/usr/bin/env python3
"""
experiments/01_cluster_bench.py
================================
Clustering benchmark: compare Random, Lloyd, and E-Means on the
label_efficient_emeans dataset using MSE as the metric.

Supports:
- Verified scaling laws and grid-best configurations
- Matched initialization (k-means++ and random)
- Sweep across arbitrary k (e.g. k=200, k=1265 in the paper's valid range)

Usage
-----
    python experiments/01_cluster_bench.py --smoke-test
    python experiments/01_cluster_bench.py --cache-dir cache --results-dir results_full --seeds 0 1 2 3 4
    python experiments/01_cluster_bench.py --cache-dir cache --results-dir results_full --k 1265 --seeds 0 1 2 3 4
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
from src.reproducibility import seed_everything

logger = setup_logging("cluster_bench")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def run_lloyd(feats: torch.Tensor, k: int, n_iter: int, seed: int, init: str = "kmeanspp") -> dict:
    t0 = time.perf_counter()
    model = LloydKMeans(k=k, seed=seed, init=init)
    model.fit(feats, n_iter=n_iter)
    wall = time.perf_counter() - t0
    assignments = model.predict(feats)
    mse = float(compute_mse(feats, model.centroids, assignments))
    return {"mse": mse, "wall_seconds": wall, "centroids": model.centroids}


def run_emeans(feats: torch.Tensor, k: int, T: int, seed: int, init: str = "kmeanspp") -> dict:
    t0 = time.perf_counter()
    cfg = EMeansConfig(
        k=k, T=T,
        alpha0=None,       # use verified scaling law
        batch_size=None,   # use verified scaling law
        use_schedule=True,
        use_revival=True,
        holdout_frac=0.0,  # evaluate on full training data, matching Lloyd
        seed=seed,
        init=init,
        finalize_last_epoch=True,
    )
    model = EMeans(cfg)
    result = model.fit(feats)
    wall = time.perf_counter() - t0
    mse = float(compute_mse(
        feats, result.centroids,
        assign_blocked(feats, result.centroids)
    ))
    return {"mse": mse, "wall_seconds": wall, "centroids": result.centroids,
            "n_updates": result.n_updates}


def run_emeans_gridbest(feats: torch.Tensor, k: int, seed: int, init: str = "kmeanspp") -> dict:
    t0 = time.perf_counter()
    # Grid-best config for k=200, T=3 from 03_law_check.py (alpha0 ≈ 0.215443, batch 1024)
    cfg = EMeansConfig(
        k=k, T=3,
        alpha0=0.215443,
        batch_size=1024,
        use_schedule=True,
        use_revival=True,
        holdout_frac=0.0,
        seed=seed,
        init=init,
        finalize_last_epoch=True,
    )
    model = EMeans(cfg)
    result = model.fit(feats)
    wall = time.perf_counter() - t0
    mse = float(compute_mse(
        feats, result.centroids,
        assign_blocked(feats, result.centroids)
    ))
    return {"mse": mse, "wall_seconds": wall, "centroids": result.centroids,
            "n_updates": result.n_updates}


def run_random_baseline(feats: torch.Tensor, k: int, seed: int) -> dict:
    t0 = time.perf_counter()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    idx = torch.randperm(feats.shape[0], generator=rng)[:k]
    centroids = feats[idx]
    wall = time.perf_counter() - t0
    assignments = assign_blocked(feats, centroids)
    mse = float(compute_mse(feats, centroids, assignments))
    return {"mse": mse, "wall_seconds": wall, "centroids": centroids}


METHODS = {
    "Random": lambda feats, k, seed: run_random_baseline(feats, k, seed),
    "Lloyd-3": lambda feats, k, seed: run_lloyd(feats, k, 3, seed),
    "Lloyd-25": lambda feats, k, seed: run_lloyd(feats, k, 25, seed),
    "EMeans-T1": lambda feats, k, seed: run_emeans(feats, k, 1, seed),
    "EMeans-T2": lambda feats, k, seed: run_emeans(feats, k, 2, seed),
    "EMeans-T3": lambda feats, k, seed: run_emeans(feats, k, 3, seed),
    "EMeans-T3-GridBest": lambda feats, k, seed: run_emeans_gridbest(feats, k, seed),
    "EMeans-T5": lambda feats, k, seed: run_emeans(feats, k, 5, seed),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Clustering benchmark")
    p.add_argument("--config", default="configs/clustering.yaml")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--k", type=int, default=None, help="Cluster count override (e.g. 200, 1265)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", nargs="+", default=None, help="Subset of methods to run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use n=5000, k=50 for speed")
    return p.parse_args()


def load_features(cache_dir: Path, smoke_test: bool, device: torch.device) -> torch.Tensor:
    npy = cache_dir / "train_feats.npy"
    if not npy.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {npy}\n"
            "Run the feature extraction pipeline first."
        )
    arr = np.load(npy)
    feats = torch.from_numpy(arr).float()
    if smoke_test:
        n = min(5000, feats.shape[0])
        feats = feats[:n]
    return feats.to(device)


def load_existing(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    df = pd.read_parquet(results_path)
    if "k" in df.columns:
        return set(zip(df["method"], df["seed"], df["k"]))
    return set(zip(df["method"], df["seed"]))


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "cluster_bench.parquet"

    device = get_device()
    logger.info("Device: %s", device)

    # Load config
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    if args.k is not None:
        k = args.k
    elif args.smoke_test:
        k = 50
    else:
        k = cfg.get("k", 200)

    # Load features
    try:
        feats = load_features(cache_dir, args.smoke_test, device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Features: %s, k=%d", feats.shape, k)

    methods_to_run = {m: METHODS[m] for m in args.methods} if args.methods else METHODS
    completed = load_existing(results_path)
    rows = []
    total = len(methods_to_run) * len(args.seeds)
    done = 0

    for method_name, method_fn in methods_to_run.items():
        for seed in args.seeds:
            done += 1
            if (method_name, seed, k) in completed:
                logger.info("[%d/%d] SKIP %s seed=%d k=%d (already done)",
                            done, total, method_name, seed, k)
                continue

            if args.dry_run:
                logger.info("[%d/%d] DRY-RUN %s seed=%d k=%d",
                            done, total, method_name, seed, k)
                continue

            logger.info("[%d/%d] Running %s seed=%d k=%d ...",
                        done, total, method_name, seed, k)
            seed_everything(seed)
            try:
                out = method_fn(feats, k, seed)
                row = {
                    "method": method_name,
                    "seed": seed,
                    "k": k,
                    "n": feats.shape[0],
                    "mse": out["mse"],
                    "wall_seconds": out["wall_seconds"],
                    "n_updates": out.get("n_updates", None),
                }
                rows.append(row)
                logger.info("  MSE=%.6f  wall=%.1fs", row["mse"], row["wall_seconds"])
            except Exception as e:
                logger.error("  FAILED: %s", e)

    if rows:
        new_df = pd.DataFrame(rows)
        if results_path.exists():
            existing = pd.read_parquet(results_path)
            # Remove any conflicting duplicate keys if overwriting
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["method", "seed", "k"], keep="last")
        else:
            combined = new_df
        tmp_path = results_path.with_name(f"{results_path.stem}.tmp.parquet")
        combined.to_parquet(tmp_path, index=False)
        save_atomic(tmp_path, results_path)
        logger.info("Saved %d rows -> %s", len(rows), results_path)

    # Print summary table
    if results_path.exists():
        df = pd.read_parquet(results_path)
        for curr_k in df["k"].unique():
            sub_df = df[df["k"] == curr_k]
            summary = (
                sub_df.groupby("method")["mse"]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "MSE_mean", "std": "MSE_std", "count": "n_runs"})
                .sort_values("MSE_mean")
            )
            print(f"\n=== Clustering Benchmark Summary (k={curr_k}) ===")
            print(summary.to_string())


if __name__ == "__main__":
    main()
