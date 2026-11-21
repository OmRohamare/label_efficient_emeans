#!/usr/bin/env python3
"""
experiments/05_official_emeans_check.py
=======================================
Reference check: run Meta's official e-means (faiss.contrib.e_means) and
faiss's own Lloyd k-means on the SAME features and k as 01_cluster_bench.py,
and measure MSE the same way: (1/N) * sum_i ||x_i - c_{a(i)}||^2 against the
final centroids with exact nearest-centroid assignment.

CPU only (NumPy path), so it works on Windows with `pip install faiss-cpu`.
Wall-clock numbers are CPU numbers and are NOT comparable to the GPU
benchmark; compare MSE.

Usage:
    python experiments/05_official_emeans_check.py --smoke-test
    python experiments/05_official_emeans_check.py --cache-dir cache \
        --results-dir results_full --seeds 0 1 2 3 4
"""
import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from faiss.contrib.e_means import NumpyEmeans


def exact_mse(x: np.ndarray, centroids: np.ndarray) -> float:
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype="float32"))
    d2, _ = index.search(x, 1)
    return float(d2.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--results-dir", default="results_full")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--T", type=float, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()

    x = np.load(Path(a.cache_dir) / "train_feats.npy").astype("float32")
    k = a.k
    if a.smoke_test:
        x, k = x[:5000], 50
    n, d = x.shape
    print(f"faiss {faiss.__version__}  n={n} d={d} k={k}")

    rows = []
    for seed in a.seeds:
        for T in a.T:
            t0 = time.perf_counter()
            em = NumpyEmeans(d, k, nepochs=T, seed=seed)
            em.train(x)
            wall = time.perf_counter() - t0
            mse = exact_mse(x, em.centroids)
            rows.append(dict(method=f"Official-EMeans-T{T:g}", seed=seed, k=k,
                             n=n, mse=mse, wall_seconds=wall))
            print(rows[-1])
        for niter in (3, 25):
            t0 = time.perf_counter()
            # max_points_per_centroid is raised so faiss does NOT subsample
            # the training set (its default is 256*k points).
            km = faiss.Kmeans(d, k, niter=niter, seed=seed, verbose=False,
                              max_points_per_centroid=10**9)
            km.train(x)
            wall = time.perf_counter() - t0
            mse = exact_mse(x, km.centroids)
            rows.append(dict(method=f"FaissLloyd-{niter}", seed=seed, k=k,
                             n=n, mse=mse, wall_seconds=wall))
            print(rows[-1])

    df = pd.DataFrame(rows)
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "official_check_smoke.parquet" if a.smoke_test else "official_check.parquet"
    df.to_parquet(out / name, index=False)
    print("\n=== Official / faiss reference ===")
    print(df.groupby("method")[["mse", "wall_seconds"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()