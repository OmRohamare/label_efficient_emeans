#!/usr/bin/env python3
"""
experiments/04_make_figures.py
================================
Generate all 9 paper figures from experiment result Parquet files.

Figures generated
-----------------
fig1_accuracy_vs_budget.png       — KNN accuracy vs label budget (line, CI)
fig2_accuracy_diffs_ci.png        — Bootstrap 95% CI on paired accuracy differences
fig3_holdout_mse_vs_updates.png   — Holdout MSE vs update step (EMeans vs Lloyd)
fig4_mse_vs_wallclock.png         — Train MSE vs wall-clock time
fig5_epochs_to_match.png          — Epochs for EMeans to match Lloyd-25 MSE
fig6_class_coverage.png           — Class coverage (%) vs budget
fig7_revival_ablation.png         — EMeans-T3 with/without revival
fig8_scaling_law_regret.png       — Regret heatmap (alpha0 × batch_size)
fig9_ceiling_reference.png        — Best clustering MSE vs linear-probe ceiling

Usage
-----
    python experiments/04_make_figures.py --results-dir results --output-dir results/figures
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.utils import setup_logging

logger = setup_logging("make_figures")

PALETTE = {
    "Random": "#aaaaaa",
    "KMeanspp": "#4c72b0",
    "Lloyd-3": "#dd8452",
    "Lloyd-25": "#55a868",
    "EMeans-T1": "#c44e52",
    "EMeans-T2": "#8172b2",
    "EMeans-T3": "#937860",
    "EMeans-T5": "#da8bc3",
    "EMeans-T3-NoRevival": "#ccb974",
    "EMeans-T3-GridBest": "#17becf",
    "Official-EMeans-T3": "#d62728",
}


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _load_df(path: Path, name: str):
    if not path.exists():
        logger.warning("Missing %s (%s) — skipping related figures", name, path)
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs Budget
# ---------------------------------------------------------------------------

def fig1_accuracy_vs_budget(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    pref_order = ["Random", "KMeanspp", "Lloyd-3", "Lloyd-25", "EMeans-T3", "EMeans-T3-NoRevival", "Official-EMeans-T3"]
    methods = [m for m in pref_order if m in df["method"].unique()] + [m for m in df["method"].unique() if m not in pref_order]
    for method in methods:
        grp = df[df["method"] == method]
        summary = grp.groupby("budget")["knn_acc"].agg(["mean", "std"]).reset_index()
        color = PALETTE.get(method, None)
        ax.plot(summary["budget"], summary["mean"], marker="o", label=method,
                color=color, linewidth=1.8)
        ax.fill_between(
            summary["budget"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.15, color=color,
        )
    ax.set_xlabel("Label Budget B")
    ax.set_ylabel("KNN Accuracy")
    ax.set_title("Label Selection: Accuracy vs Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "fig1_accuracy_vs_budget.png")


# ---------------------------------------------------------------------------
# Figure 2: Pairwise accuracy differences with bootstrap CI
# ---------------------------------------------------------------------------

def fig2_accuracy_diffs_ci(df: pd.DataFrame, out: Path):
    baseline = "Random"
    methods = [m for m in df["method"].unique() if m != baseline]
    B_ref = df["budget"].median()
    sub = df[df["budget"] == B_ref]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_ticks, y_labels = [], []

    for i, method in enumerate(methods):
        base_vals = sub[sub["method"] == baseline]["knn_acc"].values
        meth_vals = sub[sub["method"] == method]["knn_acc"].values
        min_len = min(len(base_vals), len(meth_vals))
        if min_len == 0:
            continue
        diffs = meth_vals[:min_len] - base_vals[:min_len]
        mean_diff = diffs.mean()
        lo, hi = bootstrap_ci(diffs)
        ax.barh(i, mean_diff, xerr=[[mean_diff - lo], [hi - mean_diff]],
                color=PALETTE.get(method, "steelblue"), capsize=4, alpha=0.8)
        y_ticks.append(i)
        y_labels.append(method)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(f"Accuracy − Random (B={int(B_ref)}, bootstrap 95% CI)")
    ax.set_title("Paired Accuracy Differences vs Random Baseline")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out / "fig2_accuracy_diffs_ci.png")


# ---------------------------------------------------------------------------
# Figure 3: Holdout MSE vs updates
# ---------------------------------------------------------------------------

def fig3_holdout_mse_vs_updates(df: pd.DataFrame, out: Path):
    # df should have columns: method, update_step, holdout_mse, seed
    if "update_step" not in df.columns or "holdout_mse" not in df.columns:
        logger.warning("fig3: required columns missing")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, grp in df.groupby("method"):
        summary = grp.groupby("update_step")["holdout_mse"].agg(["mean", "std"]).reset_index()
        color = PALETTE.get(method)
        ax.plot(summary["update_step"], summary["mean"], label=method,
                color=color, linewidth=1.8)
        ax.fill_between(
            summary["update_step"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.15, color=color,
        )
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Holdout MSE")
    ax.set_title("Holdout MSE vs Update Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "fig3_holdout_mse_vs_updates.png")


# ---------------------------------------------------------------------------
# Figure 4: MSE vs Wall-clock
# ---------------------------------------------------------------------------

def fig4_mse_vs_wallclock(df: pd.DataFrame, out: Path):
    if "wall_seconds" not in df.columns or "mse" not in df.columns:
        logger.warning("fig4: required columns missing")
        return

    k_vals = sorted(df["k"].unique()) if "k" in df.columns else [None]
    if len(k_vals) > 1:
        fig, axes = plt.subplots(1, len(k_vals), figsize=(6 * len(k_vals), 4), sharey=False)
        for ax, k_val in zip(axes, k_vals):
            sub = df[df["k"] == k_val]
            for method, grp in sub.groupby("method"):
                ax.scatter(grp["wall_seconds"], grp["mse"],
                           label=method, color=PALETTE.get(method),
                           alpha=0.8, s=40)
            ax.set_xlabel("Wall-clock Time (s)")
            ax.set_ylabel("Train MSE")
            ax.set_title(f"MSE vs Wall-clock Time (k={k_val})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, grp in df.groupby("method"):
            ax.scatter(grp["wall_seconds"], grp["mse"],
                       label=method, color=PALETTE.get(method),
                       alpha=0.8, s=40)
        ax.set_xlabel("Wall-clock Time (s)")
        ax.set_ylabel("Train MSE")
        k_str = f" (k={k_vals[0]})" if k_vals[0] else ""
        ax.set_title(f"MSE vs Wall-clock Time{k_str}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    _save(fig, out / "fig4_mse_vs_wallclock.png")


# ---------------------------------------------------------------------------
# Figure 5: Epochs to match Lloyd-25
# ---------------------------------------------------------------------------

def fig5_epochs_to_match(df: pd.DataFrame, out: Path):
    if "mse" not in df.columns:
        logger.warning("fig5: required columns missing")
        return

    lloyd25_mse = df[df["method"] == "Lloyd-25"]["mse"].mean()
    emeans_methods = [m for m in df["method"].unique() if "EMeans" in m]

    epochs, mean_mse = [], []
    for m in sorted(emeans_methods):
        if "-NoRevival" in m or "-GridBest" in m or "Official" in m:
            continue
        try:
            T = int(m.split("-T")[-1])
        except (ValueError, IndexError):
            continue
        mse = df[df["method"] == m]["mse"].mean()
        epochs.append(T)
        mean_mse.append(mse)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, mean_mse, marker="o", color="#c44e52", linewidth=1.8,
            label="EMeans (mean MSE)")
    ax.axhline(lloyd25_mse, color="#55a868", linestyle="--", linewidth=1.5,
               label="Lloyd-25 MSE")
    ax.set_xlabel("Number of Passes T")
    ax.set_ylabel("Train MSE")
    ax.set_title("Passes to Match Lloyd-25 MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "fig5_epochs_to_match.png")


# ---------------------------------------------------------------------------
# Figure 6: Class coverage vs budget
# ---------------------------------------------------------------------------

def fig6_class_coverage(df: pd.DataFrame, out: Path):
    if "class_coverage" not in df.columns:
        logger.warning("fig6: class_coverage column missing")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    pref_order = ["Random", "KMeanspp", "Lloyd-3", "Lloyd-25", "EMeans-T3", "EMeans-T3-NoRevival", "Official-EMeans-T3"]
    methods = [m for m in pref_order if m in df["method"].unique()] + [m for m in df["method"].unique() if m not in pref_order]
    for method in methods:
        grp = df[df["method"] == method]
        summary = grp.groupby("budget")["class_coverage"].agg(["mean", "std"]).reset_index()
        color = PALETTE.get(method)
        ax.plot(summary["budget"], summary["mean"] * 100, marker="s",
                label=method, color=color, linewidth=1.8)
    ax.set_xlabel("Label Budget B")
    ax.set_ylabel("Class Coverage (%)")
    ax.set_title("Class Coverage vs Label Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    _save(fig, out / "fig6_class_coverage.png")


# ---------------------------------------------------------------------------
# Figure 7: Revival ablation
# ---------------------------------------------------------------------------

def fig7_revival_ablation(df: pd.DataFrame, out: Path):
    methods = ["EMeans-T3", "EMeans-T3-NoRevival"]
    sub = df[df["method"].isin(methods)]
    if sub.empty:
        logger.warning("fig7: no revival ablation data")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in sub.groupby("method"):
        summary = grp.groupby("budget")["knn_acc"].agg(["mean", "std"]).reset_index()
        color = PALETTE.get(method)
        ax.plot(summary["budget"], summary["mean"], marker="o",
                label=method, color=color, linewidth=1.8)
        ax.fill_between(
            summary["budget"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.15, color=color,
        )
    ax.set_xlabel("Label Budget B")
    ax.set_ylabel("KNN Accuracy")
    ax.set_title("Revival Ablation: EMeans-T3 ± Revival")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "fig7_revival_ablation.png")


# ---------------------------------------------------------------------------
# Figure 8: Scaling law regret heatmap
# ---------------------------------------------------------------------------

def fig8_scaling_law_regret(df: pd.DataFrame, out: Path):
    grid = df[df.get("row_type", pd.Series(["grid"] * len(df))) == "grid"]
    if grid.empty or "alpha0" not in grid.columns or "batch_size" not in grid.columns:
        logger.warning("fig8: no grid data for regret heatmap")
        return

    pivot = grid.groupby(["alpha0", "batch_size"])["mse"].mean().reset_index()
    alpha_vals = sorted(pivot["alpha0"].unique())
    bs_vals = sorted(pivot["batch_size"].unique())
    Z = np.zeros((len(alpha_vals), len(bs_vals)))
    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(bs_vals):
            match = pivot[(pivot["alpha0"] == a) & (pivot["batch_size"] == b)]["mse"]
            Z[i, j] = match.values[0] if len(match) else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(bs_vals)))
    ax.set_xticklabels([str(b) for b in bs_vals])
    ax.set_yticks(range(len(alpha_vals)))
    ax.set_yticklabels([f"{a:.3f}" for a in alpha_vals])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("alpha0")
    ax.set_title("MSE Heatmap (alpha0 × batch_size)\n[lower = better]")
    fig.colorbar(im, ax=ax, label="Mean MSE")
    _save(fig, out / "fig8_scaling_law_regret.png")


# ---------------------------------------------------------------------------
# Figure 9: Ceiling reference
# ---------------------------------------------------------------------------

def fig9_ceiling_reference(cluster_df: pd.DataFrame, out: Path, ceiling: float = None):
    if ceiling is None:
        logger.warning("fig9: no ceiling value — skipping")
        return

    methods = ["Lloyd-25", "EMeans-T3", "EMeans-T5"]
    sub = cluster_df[cluster_df["method"].isin(methods)]

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in sub.groupby("method"):
        mse_mean = grp["mse"].mean()
        ax.bar(method, mse_mean, color=PALETTE.get(method, "steelblue"), alpha=0.8)
    ax.axhline(ceiling, color="red", linestyle="--", linewidth=1.5,
               label=f"Linear probe ceiling (MSE={ceiling:.4f})")
    ax.set_ylabel("Train MSE")
    ax.set_title("Best Clustering MSE vs Linear-Probe Ceiling")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "fig9_ceiling_reference.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate paper figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="results/figures")
    return p.parse_args()


def main():
    if not HAS_MPL:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load available Parquet files
    sel_df = _load_df(results_dir / "selection_sweep.parquet", "selection_sweep")
    clust_df = _load_df(results_dir / "cluster_bench.parquet", "cluster_bench")
    law_df = _load_df(results_dir / "law_check.parquet", "law_check")

    # --- Figure 1 ---
    if sel_df is not None:
        try:
            fig1_accuracy_vs_budget(sel_df, out_dir)
        except Exception as e:
            logger.error("fig1 failed: %s", e)

    # --- Figure 2 ---
    if sel_df is not None:
        try:
            fig2_accuracy_diffs_ci(sel_df, out_dir)
        except Exception as e:
            logger.error("fig2 failed: %s", e)

    # --- Figure 3 (needs holdout MSE traces) ---
    if clust_df is not None and "update_step" in (clust_df.columns if clust_df is not None else []):
        try:
            fig3_holdout_mse_vs_updates(clust_df, out_dir)
        except Exception as e:
            logger.error("fig3 failed: %s", e)
    else:
        logger.warning("fig3: no update_step column — skipping")
        # Create placeholder
        if HAS_MPL:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Holdout MSE vs Updates")
            _save(fig, out_dir / "fig3_holdout_mse_vs_updates.png")

    # --- Figure 4 ---
    if clust_df is not None and "mse" in clust_df.columns:
        try:
            fig4_mse_vs_wallclock(clust_df, out_dir)
        except Exception as e:
            logger.error("fig4 failed: %s", e)

    # --- Figure 5 ---
    if clust_df is not None:
        try:
            fig5_epochs_to_match(clust_df, out_dir)
        except Exception as e:
            logger.error("fig5 failed: %s", e)

    # --- Figure 6 ---
    if sel_df is not None:
        try:
            fig6_class_coverage(sel_df, out_dir)
        except Exception as e:
            logger.error("fig6 failed: %s", e)

    # --- Figure 7 ---
    if sel_df is not None:
        try:
            fig7_revival_ablation(sel_df, out_dir)
        except Exception as e:
            logger.error("fig7 failed: %s", e)

    # --- Figure 8 ---
    if law_df is not None:
        try:
            fig8_scaling_law_regret(law_df, out_dir)
        except Exception as e:
            logger.error("fig8 failed: %s", e)

    # --- Figure 9 ---
    # Try to read ceiling from cluster_bench if available
    if clust_df is not None:
        ceiling_col = "linear_probe_ceiling"
        if ceiling_col in clust_df.columns:
            ceiling = clust_df[ceiling_col].dropna().mean()
            try:
                fig9_ceiling_reference(clust_df, out_dir, ceiling=ceiling)
            except Exception as e:
                logger.error("fig9 failed: %s", e)
        else:
            logger.warning("fig9: no linear_probe_ceiling column — skipping")

    logger.info("Figure generation complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
