# Experimental Protocol

This document describes the full experimental protocol for the `label_efficient_emeans` project, covering data preparation, feature extraction, clustering evaluation, label selection, and figure generation.

---

## 1. Environment Setup

```bash
git clone <repo_url> label_efficient_emeans
cd label_efficient_emeans
pip install -e ".[dev]"
```

Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Dataset

**Dataset:** `zh-plus/tiny-imagenet` from HuggingFace Hub.

- **Train split:** 100,000 images × 200 classes (500 per class)
- **Val split:** 10,000 images × 200 classes (50 per class)
- **Resolution:** 64×64 RGB (resized to 224×224 for DINOv2)

Download and verify:
```bash
python -c "from src.data import verify_dataset; verify_dataset()"
```

---

## 3. Feature Extraction

**Model:** DINOv2 ViT-S/14, CLS token, L2-normalised.

```bash
python -m src.features --split train --output-dir cache/
python -m src.features --split val   --output-dir cache/
```

Output files:
- `cache/train_feats.npy` — shape `(100000, 384)`, float32
- `cache/train_labels.npy` — shape `(100000,)`, int64
- `cache/val_feats.npy` — shape `(10000, 384)`, float32
- `cache/val_labels.npy` — shape `(10000,)`, int64

Verify L2-normalisation:
```python
import numpy as np
feats = np.load("cache/train_feats.npy")
norms = np.linalg.norm(feats, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5)
```

---

## 4. Linear Probe Ceiling

Establishes the upper-bound accuracy using the **full** training set.

```bash
python -c "from src.features import linear_probe_ceiling; linear_probe_ceiling('cache/')"
```

Saves `cache/ceiling.json` with keys `{"knn_acc": float, "logistic_acc": float}`.

> **Note:** This ceiling is computed with **all** labels — not a fair comparison to the budget-limited methods, but serves as a reference.

---

## 5. Clustering Benchmark (Experiment 1)

Runs all 7 methods (Random, Lloyd-3, Lloyd-25, EMeans-T1 through T5) with seeds 0–4.

```bash
python experiments/01_cluster_bench.py \
    --config configs/clustering.yaml \
    --cache-dir cache \
    --results-dir results \
    --seeds 0 1 2 3 4
```

**Smoke test** (5000 points, k=50, fast):
```bash
python experiments/01_cluster_bench.py --smoke-test
```

Output: `results/cluster_bench.parquet`

---

## 6. Selection Budget Sweep (Experiment 2)

Evaluates 6 selection strategies across 5 budget levels with 10 seeds each.

```bash
python experiments/02_selection_sweep.py \
    --config configs/experiments.yaml \
    --cache-dir cache \
    --results-dir results
```

**Smoke test** (B=[200,500], 1 seed):
```bash
python experiments/02_selection_sweep.py --smoke-test
```

Output: `results/selection_sweep.parquet`

| Strategy | Description |
|---|---|
| Random | Uniform random sampling |
| KMeanspp | K-means++ seeding as selector |
| Lloyd-3 | Nearest-to-centroids after 3 Lloyd iters |
| Lloyd-25 | Nearest-to-centroids after 25 Lloyd iters |
| EMeans-T3 | Nearest-to-centroids after e-means (T=3, with revival) |
| EMeans-T3-NoRevival | Same without dead-cluster revival |

---

## 7. Scaling Law Check (Experiment 3)

Verifies whether law-predicted hyperparameters $(B^*, \alpha_0^*)$ achieve near-optimal MSE.

```bash
python experiments/03_law_check.py \
    --cache-dir cache \
    --results-dir results
```

Output: `results/law_check.parquet`

> ⚠️ k=200 results are flagged as **EXTRAPOLATION** in the output.

---

## 8. Figure Generation

```bash
python experiments/04_make_figures.py \
    --results-dir results \
    --output-dir results/figures
```

Generates 9 figures in `results/figures/`. Missing Parquet files are skipped gracefully.

---

## 9. Running All Tests

```bash
pytest tests/ -v --tb=short
```

For GPU tests only:
```bash
pytest tests/ -v -k "gpu"
```

For fast unit tests only (excludes sklearn integration):
```bash
pytest tests/ -v -k "not sklearn"
```

---

## 10. Reproducibility Checklist

- [ ] All seeds set via `src/reproducibility.py::seed_everything(seed)`
- [ ] DINOv2 model loaded in `eval()` mode with `torch.no_grad()`
- [ ] Features are L2-normalised (verified by norm check)
- [ ] Results saved atomically via `src/utils.py::save_atomic`
- [ ] Completed (method, seed) combos skipped on re-run (resume support)
- [ ] All UNVERIFIED scaling law coefficients are flagged in code comments

---

## 11. Expected Runtimes

| Step | GPU (A100) | CPU (16-core) |
|---|---|---|
| Feature extraction (train) | ~8 min | ~90 min |
| Clustering benchmark | ~5 min | ~25 min |
| Selection sweep | ~20 min | ~120 min |
| Scaling law check | ~30 min | ~180 min |

> Times are approximate and depend on hardware and batch size.
