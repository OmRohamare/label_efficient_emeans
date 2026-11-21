# Limitations and Reproducibility Notes

This document honestly describes the limitations of this project and the conditions under which results should be interpreted.

---

## 1. Scaling Law Exponents: Verified Against Meta's Faiss Implementation

**Status:** Verified and updated from Meta FAIR's official Faiss implementation (`faiss.contrib.e_means`).

**Background & Fix:** An earlier draft of `src/laws.py` used an unverified algebraic heuristic where $k$ canceled out. This was identified and replaced with the verified multi-variable OLS scaling laws from Meta FAIR / Faiss:
- Throughput batch law: $b_c = 0.035134, b_k = 0.134749, b_n = 0.876473, b_T = 0.813344$
- Initial learning rate law: $a_c = 0.8244, a_k = -0.4617, a_B = 0.9003, a_n = -0.4498, a_T = -0.1057$

**Residual Consideration:** The paper's scaling laws were fitted on raw-scale (unnormalized) embeddings across large $k \ge 4\sqrt{n}$. On $L_2$-normalized DINOv2 embeddings at low $k$ ($k=200$), the power law prescribes $B=4096, \alpha_0=0.6405$. An empirical grid sweep (`experiments/03_law_check.py`) showed that a slightly smaller batch size ($B=1024, \alpha_0 \approx 0.215$) yields lower MSE (0.65179 vs 0.65190). At $k=1265$ (inside the paper's valid range), the published scaling law ($B=4096, \alpha_0=0.2733$) outperforms Lloyd-25 directly without manual tuning.

---

## 2. L2-Normalized Features vs. Raw-Scale Embeddings

The paper's scaling laws were fitted on raw-scale embedding distributions: DEEP1B (96-dim), SIFT1M (128-dim), SPACEV1B (100-dim), VIBE (64-dim). These are **not** L2-normalized.

This project uses L2-normalized DINOv2 features (384-dim). After normalization:
- All feature vectors lie on the unit hypersphere.
- The effective magnitude of distances is bounded: `||x - c||² ∈ [0, 4]` for unit vectors.
- The optimal step sizes and batch sizes for normalized data may differ substantially from those derived for raw embeddings.

**Consequence:** The scaling laws should be treated as a starting point, not as a validated prescription for this specific setup. The empirical sweep in `experiments/03_law_check.py` measures actual regret for this dataset.

---

## 3. Input Resolution: 64×64 → 224×224 Upsampling

Tiny-ImageNet images are 64×64 pixels. DINOv2 ViT-S/14 was designed for 224×224 inputs. This project upsamples images using bilinear interpolation before passing them to DINOv2.

**Consequences:**
- Feature quality may be lower than on native-resolution images.
- The linear-probe ceiling is a **representation-specific** upper bound, not a theoretical upper bound on all possible classifiers.
- Results cannot be directly compared to experiments on CIFAR-10/100 or full-resolution ImageNet.

---

## 4. k=200 Is Outside the Paper's Valid Range

For n=100,000, the paper states a guideline of k ≥ 4√n ≈ 1,265. The budget B=200 corresponds to k=200, which falls below this threshold.

**Consequence:** Results for B=200 are flagged as **extrapolation experiments** in all experiment scripts. The scaling laws may produce extreme or nonsensical values at k=200. The algorithm still runs, but performance guarantees do not apply.

---

## 5. GPU Nondeterminism

Certain GPU operations (e.g., `torch.cumsum` on CUDA) may produce slightly different floating-point results across runs even with the same seed. This project:
- Sets `torch.backends.cudnn.deterministic = True` where possible.
- Uses `torch.use_deterministic_algorithms(True)` where available.
- Documents that some GPU operations remain nondeterministic due to hardware parallelism.

For fully reproducible numerical results, run on CPU (slow but deterministic).

---

## 6. Validation Set Independence

The validation split is used **only** for final evaluation. It is never used to:
- Select clustering hyperparameters.
- Tune classifier regularization.
- Choose between methods or seeds.

Regularization strength for the logistic probe is selected using a held-out subset of the **training pool** (not the validation set).

---

## 7. Statistical Interpretation

**Bootstrap confidence intervals** (95%) are reported for pairwise accuracy differences (e-means vs. others). These intervals:
- Are computed by resampling **seeds** (the natural unit of independent repetition).
- Measure uncertainty about the **mean** accuracy difference across seeds.
- Do **not** tell you whether the difference is statistically significant in a frequentist hypothesis-testing sense.
- Overlapping intervals do **not** imply equivalence; non-overlapping intervals do **not** guarantee significance at any specific α level.

Differences that are not distinguishable from zero within these intervals are reported as **inconclusive**, not as proof that the methods are equivalent.

---

## 8. NMI Is a Diagnostic Only

Normalized Mutual Information (NMI) is computed only for the k=200 diagnostic. It is:
- Not used to select methods or hyperparameters.
- Not the primary metric (downstream classification accuracy is the primary metric).
- Computed **after** selection, using ground-truth training labels (never during clustering).

A high NMI does not imply better downstream accuracy; this is treated as an empirical question.

---

## 9. Missing Classes in Selected Subsets

At small budgets (B=200, B=500), no selection method is guaranteed to cover all 200 classes. When the selected set misses some classes:
- The logistic probe is trained only on seen classes. Missing classes cannot be predicted.
- This is **documented** in `src/probe.py`, not silently ignored.
- Class coverage (number of distinct classes in selected set) is recorded as a metric.

---

## 10. Colab Session Interruptions

Google Colab sessions can disconnect. All experiment scripts support resuming:
- Results are saved to Parquet after each (method, B, seed) combination.
- Already-completed combinations are skipped on restart.
- Feature caches are saved to disk and reused across sessions.

Feature extraction and experiments should be run with Google Drive mounted to persist results across Colab sessions.

---

## Summary Table

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| Unverified scaling law exponents | Medium | Empirical sweep + manual override |
| L2-normalized vs. raw features | Medium | Noted in all comparisons |
| 64→224 upsampling | Low–Medium | Use only as relative comparison |
| k=200 out of valid range | Low | Flagged as extrapolation |
| GPU nondeterminism | Low | Documented; use CPU for exact repro |
| Small n_seeds for CI | Medium | 10 seeds; CIs reported with width |
| NMI as diagnostic only | Low | Not used for selection |
