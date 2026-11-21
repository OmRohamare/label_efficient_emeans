#!/usr/bin/env python3
"""
experiments/03_law_check.py
============================
Scaling law verification sweep: checks whether the UNVERIFIED power-law
formulas for alpha0 and batch_size actually minimise MSE.

Sweep grid
----------
- k ∈ {200, 1024}        (k=200 is EXTRAPOLATION — flagged in output)
- alpha0 ∈ 10 values geom-spaced in [1e-3, 1.0]
- batch_size ∈ {256, 1024}
- seeds: 0–4

For each (k, alpha0, batch_size, seed) we also run the law-predicted params
and compute regret = (MSE_config - MSE_best) / MSE_best.

Results saved to results/law_check.parquet.

Usage
-----
    python experiments/03_law_check.py --smoke-test
    python experiments/03_law_check.py --cache-dir cache --results-dir results
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

from src.utils import get_device, save_atomic, setup_logging, load_config
from src.assign import assign_blocked, compute_mse
from src.emeans import EMeans, EMeansConfig
from src.laws import compute_alpha0, compute_batch_size
from src.reproducibility import seed_everything

logger = setup_logging("law_check")

K_VALUES = [200, 1024]
N_ALPHA = 10
ALPHA_RANGE = (1e-3, 1.0)
BATCH_SIZES = [256, 1024]
SEEDS = [0, 1, 2, 3, 4]
T = 3  # fixed number of passes for all runs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(cache_dir: Path, smoke_test: bool) -> torch.Tensor:
    npy = cache_dir / "train_feats.npy"
    if not npy.exists():
        raise FileNotFoundError(f"Missing: {npy}")
    arr = np.load(npy)
    feats = torch.from_numpy(arr).float()
    if smoke_test:
        feats = feats[:5000]
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scaling law check")
    p.add_argument("--config", default="configs/clustering.yaml")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_emeans_with_params(
    feats: torch.Tensor,
    k: int,
    T: int,
    alpha0: float,
    batch_size: int,
    seed: int,
) -> float:
    """Run e-means with explicit override params; return final MSE."""
    cfg = EMeansConfig(
        k=k, T=T,
        alpha0=alpha0,
        batch_size=batch_size,
        use_schedule=True,
        use_revival=False,
        holdout_frac=0.0,
        seed=seed,
    )
    result = EMeans(cfg).fit(feats)
    assignments = assign_blocked(feats, result.centroids)
    return float(compute_mse(feats, result.centroids, assignments))


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "law_check.parquet"

    try:
        feats = load_features(cache_dir, args.smoke_test)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    n = feats.shape[0]
    logger.info("Features: %s", feats.shape)

    alpha_values = np.geomspace(ALPHA_RANGE[0], ALPHA_RANGE[1], N_ALPHA).tolist()
    k_list = [200] if args.smoke_test else K_VALUES
    bs_list = [256] if args.smoke_test else BATCH_SIZES
    seeds = [0] if args.smoke_test else SEEDS

    rows = []

    for k in k_list:
        extrapolation = k == 200   # outside the paper's validated range
        flag = "EXTRAPOLATION" if extrapolation else "OK"

        # Compute law-predicted params
        law_alpha0 = compute_alpha0(n=n, k=k, T=T)
        law_batch_size = compute_batch_size(n=n, k=k, T=T)

        # For each seed, find best MSE across the full grid
        best_mse_per_seed: dict[int, float] = {}
        grid_mse: dict[tuple, float] = {}   # (alpha0, bs, seed) -> mse, reused below
        total_grid = len(seeds) * len(alpha_values) * len(bs_list)
        done_grid = 0
        t_start = time.time()

        for seed in seeds:
            best = float("inf")
            for alpha0 in alpha_values:
                for bs in bs_list:
                    if args.dry_run:
                        logger.info("DRY-RUN k=%d alpha0=%.4f bs=%d seed=%d",
                                    k, alpha0, bs, seed)
                        continue
                    seed_everything(seed)
                    try:
                        mse = run_emeans_with_params(feats, k, T, alpha0, bs, seed)
                        best = min(best, mse)
                        grid_mse[(alpha0, bs, seed)] = mse
                    except Exception as e:
                        logger.warning("k=%d a0=%.4f bs=%d seed=%d FAIL: %s",
                                       k, alpha0, bs, seed, e)
                    done_grid += 1
                    if done_grid % 5 == 0 or done_grid == total_grid:
                        el = time.time() - t_start
                        eta = el / done_grid * (total_grid - done_grid)
                        logger.info("k=%d grid %d/%d  elapsed=%.0fs  eta(this k)=%.0fs",
                                    k, done_grid, total_grid, el, eta)
            best_mse_per_seed[seed] = best

        # Now run law-predicted params
        for seed in seeds:
            if args.dry_run:
                logger.info("DRY-RUN law k=%d seed=%d", k, seed)
                continue
            seed_everything(seed)
            try:
                mse_law = run_emeans_with_params(
                    feats, k, T,
                    alpha0=law_alpha0,
                    batch_size=int(law_batch_size),
                    seed=seed,
                )
                mse_best = best_mse_per_seed.get(seed, float("nan"))
                regret = (mse_law - mse_best) / (mse_best + 1e-9)
                rows.append({
                    "k": k,
                    "n": n,
                    "T": T,
                    "seed": seed,
                    "mse_law": mse_law,
                    "mse_best": mse_best,
                    "regret": regret,
                    "law_alpha0": law_alpha0,
                    "law_batch_size": law_batch_size,
                    "flag": flag,
                })
                logger.info("k=%d seed=%d MSE_law=%.5f MSE_best=%.5f regret=%.3f [%s]",
                            k, seed, mse_law, mse_best, regret, flag)
            except Exception as e:
                logger.error("k=%d seed=%d law FAIL: %s", k, seed, e)

        # Also record the grid sweep results
        for alpha0 in alpha_values:
            for bs in bs_list:
                for seed in seeds:
                    if args.dry_run:
                        continue
                    key = (alpha0, bs, seed)
                    if key not in grid_mse:     # run failed in the first pass
                        continue
                    try:
                        mse = grid_mse[key]
                        rows.append({
                            "k": k,
                            "n": n,
                            "T": T,
                            "seed": seed,
                            "alpha0": alpha0,
                            "batch_size": bs,
                            "mse": mse,
                            "flag": flag,
                            "row_type": "grid",
                        })
                    except Exception:
                        pass

    if rows and not args.dry_run:
        new_df = pd.DataFrame(rows)
        if results_path.exists():
            existing = pd.read_parquet(results_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        tmp = results_path.with_name(f"{results_path.stem}.tmp.parquet")
        combined.to_parquet(tmp, index=False)
        save_atomic(tmp, results_path)
        logger.info("Saved %d rows → %s", len(rows), results_path)

    # Summary
    if results_path.exists():
        df = pd.read_parquet(results_path)
        law_rows = df[df.get("row_type", pd.Series(["law"] * len(df))) != "grid"]
        if "regret" in law_rows.columns:
            summary = law_rows.groupby(["k", "flag"])["regret"].agg(["mean", "std"])
            print("\n=== Scaling Law Regret Summary ===")
            print(summary.to_string())


if __name__ == "__main__":
    main()