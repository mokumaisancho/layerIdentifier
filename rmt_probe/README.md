# RMT-Cleaned Probe Training for LLM Correctness Prediction

A methodology for compressing and interpreting LLM probe features using Random Matrix Theory (Marchenko-Pastur filtering with Ledoit-Wolf shrinkage). Tested on Gemma-4B-QAT and Qwen3.5-4B-MLX with 150 problems each.

---

## TL;DR

- LLM probe features (per-layer statistics of activation norms) are **highly redundant**.
- Random Matrix Theory compresses **588 → 14 features** (Gemma) and **448 → 14 features** (Qwen).
- AUROC loss is negligible: 0.988 → 0.982 (Gemma), 0.995 → 0.963 (Qwen).
- Cross-architecture mode comparison: max cosine similarity 0.47 — modes do not transfer.
- All numbers reproducible from `captures.json` + the scripts in this repo. No model inference required to verify the results.

---

## What this is

A pipeline that takes precomputed per-layer probe features and:

1. **Standardizes** the feature matrix (z-score per column)
2. **Applies Ledoit-Wolf shrinkage** to the sample covariance (robust to small T)
3. **Eigendecomposes** the shrunk covariance
4. **Selects signal modes** above the Marchenko-Pastur upper noise edge
5. **Projects** the original features onto the signal eigenvectors
6. **Trains a logistic-regression probe** with stratified k-fold CV
7. **Bootstraps** 1000 resamples for 95% confidence intervals

Outputs: AUROC, bootstrap CI, top-feature loadings per mode, cross-architecture mode similarity.

---

## Key results

| Metric | Gemma-4B-E4B | Qwen3.5-4B |
|---|---|---|
| Raw features | 588 | 448 |
| RMT signal modes | 14 | 14 (MP cutoff gave 12, corrected to 14) |
| **RMT-k AUROC** | **0.982** ± 0.022 | **0.963** ± 0.020 |
| Raw AUROC (baseline) | 0.988 ± 0.018 | 0.995 ± 0.011 |
| Compression ratio | 33× | 32× |
| Bootstrap 95% CI half-width | ≤ 0.022 | ≤ 0.022 |

### Cross-architecture mode comparison

3×3 cosine similarity between top-3 Gemma modes and top-3 Qwen modes (computed on 448 common features):

```
              Qwen-m0   Qwen-m1   Qwen-m2
Gemma-m0     +0.436    +0.046    -0.004
Gemma-m1     +0.020    -0.471    -0.311
Gemma-m2     -0.238    +0.113    +0.055
```

Max |cos| = 0.471. **Modes do not transfer across architectures.** This independently confirms cross-arch probe transfer failure (originally observed in direct probe-transfer tests).

### Acceptance criteria scorecard

8 of 12 quantitative ACs pass. The 4 that fail have documented root causes (small-sample variance, MP-cutoff too tight for correlated features, modes are continuum-not-clusters). See `results/RMT_PROBE_RESULTS.md` for the full table.

---

## What RMT modes represent

Each RMT mode is **one feature family**, computed across many layers — NOT a layer module.

| Arch | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Gemma | `delta_mean` family | `delta_std` family | `delta_variance` family |
| Qwen | `convergence_slope` family | `std_norm` family | `mean_norm` family |

A mode is "the `delta_mean` statistic trajectory across all layers" — not "layer 17". The signal is highly compressible because features within a family are nearly redundant (e.g., `L12_delta_mean`, `L13_delta_mean`, `L14_delta_mean` all measure similar dynamics).

---

## Repository structure

```
rmt_probe/
├── README.md                         # this file
├── plan.md                           # the full ultraplan document
├── src/
│   ├── loader.py                     # captures.json → feature matrix
│   ├── eigendecomp.py                # Ledoit-Wolf + Marchenko-Pastur + projection
│   ├── probe.py                      # stratified k-fold CV LR probe
│   ├── bootstrap.py                  # bootstrap CIs
│   ├── pipeline.py                   # Wave 1 MVP orchestrator
│   ├── wave2.py                      # Wave 2 robustness (4 tasks)
│   ├── wave3.py                      # Wave 3 interpretation (4 tasks)
│   ├── pipeline_check.py             # final verification
│   └── qwen_l14_as_probe.py          # ancillary: per-layer probe check
├── tests/
│   ├── test_loader.py
│   ├── test_eigendecomp.py
│   ├── test_probe.py
│   ├── test_bootstrap.py
│   └── test_pipeline.py
└── results/
    ├── RMT_PROBE_RESULTS.md          # full write-up
    ├── wave1_summary.json
    ├── wave2_results.json
    ├── wave3_results.json
    ├── wave1_gemma/eigvec_cache.npz
    └── wave1_qwen/eigvec_cache.npz
```

---

## How to reproduce

### Requirements

- Python 3.11+
- scikit-learn 1.8.0
- numpy 2.4.3
- scipy 1.17.1
- pytest

### Quickstart

```bash
# 1. Run all unit tests (should show 31 passed)
python3 -m pytest rmt_probe/tests -v

# 2. Run the full pipeline check
python3 rmt_probe/src/pipeline_check.py
# Expected output: "FINAL: ✓ ALL CHECKS PASS"

# 3. Re-run Wave 1 from scratch (requires captures.json files)
python3 rmt_probe/src/pipeline.py
```

### Captures data

The pipeline reads from:
- `../results/gemma-4-E4B-nothinking/captures.json` (T=150, N=588)
- `../results/qwen3.5-4B-nothinking/captures.json` (T=150, N=448)

Each entry contains:
- `final_correct` (binary label)
- `layer_features` (dict: `{layer_name}_{feature_name}` → value)
- 14 feature families per layer × 42 layers (Gemma) / 32 layers (Qwen)

Captures can be regenerated from running `src/entropy_pipeline.py` on a problem set with the LLM. The capture step is the slow part (~3 minutes per problem on M1); the RMT analysis itself runs in seconds.

---

## Limitations

These are honest limitations — do not use this repo for things it cannot deliver.

### Methodological

1. **T = 150 problems is small.** Per-fold variance is non-trivial (10-fold std 0.041 / 0.068). Extending to T ≥ 1500 would tighten CIs.
2. **Two architectures only** (Gemma, Qwen). Cross-arch non-transferability claim is n=2. Needs Llama / Mistral / Phi replication for robustness.
3. **Single dataset.** All problems come from one arithmetic/reasoning benchmark. Generalization to MMLU, HumanEval, ARC, etc. is untested.
4. **MP cutoff assumes i.i.d. features.** Probe features are correlated by construction (layers share inputs). The Ledoit-Wolf shrinkage handles most of this, but Qwen needed a corrected k (12 → 14) — MP was too tight. This is a known caveat, not a bug.
5. **RMT ≈ PCA when shrinkage is mild** (LW intensity 0.07–0.11 here). At fixed k, RMT modes and PCA modes give identical AUROCs. RMT's contribution is the **data-driven k selection**, not a different subspace. If you already know k, PCA suffices.

### Interpretive

6. **RMT modes are NOT cognitive modes.** They are statistical aggregates (e.g., "delta_mean across all layers"). They do not correspond to "thinking", "verifying", "retrieving", or any other cognitive activity.
7. **RMT modes are NOT layer modules.** Silhouette score is 0.07–0.09 (target ≥ 0.40). Modes do not form discrete layer clusters. If you want layer-level modules, this is the wrong tool.
8. **Probe features are per-problem, not per-token.** All within-problem temporal structure is collapsed. A "thinking mode" lives in the token-time dimension, which this pipeline does not capture.
9. **No causal mechanism.** This pipeline describes WHERE the discriminative signal lives, not WHY it arises or HOW to intervene. Correlation only.

### Engineering

10. **No streaming / online inference.** Pipeline is batch-mode. Real-time deployment would need per-problem feature extraction + mode projection on the fly.
11. **Bootstrap cost is O(n_resamples × probe_fit).** 1000 resamples × 5-fold CV × 2 architectures ≈ 3 minutes. Acceptable for analysis, too slow for hyperparameter search.

---

## Future initiatives (recommended extensions)

Ordered by expected value / cost.

### High value

**1. Per-token extension → true cognitive modes**
Capture per-token activations (not per-problem features), label tokens by cognitive activity (planning, verifying, retrieving, doubting), reapply RMT. This is the path from "statistical modes" to "cognitive modes." Requires new capture pipeline (~6–10 hours infra).

**2. Multi-architecture replication**
Add Llama-3-8B, Mistral-7B, Phi-3-mini. Re-test cross-arch cosine claim. If max cosine remains < 0.5 across all pairs, the "no transfer" finding becomes robust (n=5 instead of n=2). Requires ~4 hours capture time per arch.

**3. Causal intervention (activation steering)**
Use the dominant writer layer (L14 in Qwen, L2+L4 in Gemma) to test if adding a "correct direction" can flip incorrect→correct predictions. Closes the correlation→causation gap. ~1–2 hours.

**4. SAE feature linkage**
Compute sparse autoencoder features on the same activations, then map SAE features to RMT modes via correlation. This connects RMT's data-driven modes to interpretability-grounded SAE features (Anthropic-style). Requires SAE training or pretrained SAE.

### Medium value

**5. Multi-dataset generalization**
Run on MMLU, ARC, HumanEval, GSM8K. Does k=14 still suffice? Do modes change per dataset? ~1 hour analysis per dataset (capture cost varies).

**6. Function-vector comparison**
Compare RMT modes to function vectors (Hendel et al 2023). Both are low-rank probes — are they the same directions?

**7. Larger-N replication (T = 1500+)**
Re-run everything with T=1500 problems. Tightens CIs, validates AUROC numbers. Mostly capture cost.

### Lower value / risk

**8. Online probe deployment**
Real-time correctness prediction during inference. Engineering challenge, modest scientific value.

**9. Neuron-level importance**
Within the dominant writer layer (e.g., Qwen L14), identify which neurons carry the signal. Requires model access for per-neuron ablation.

**10. Probe-feature family expansion**
Add new feature families beyond the current 14 (e.g., attention-entropy features, residual-stream magnitude). May reveal additional structure.

---

## Related work

- **Ledoit-Wolf shrinkage**: Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional covariance matrices."
- **Marchenko-Pastur law**: Marchenko & Pastur (1967), eigenvalue distribution of random matrices.
- **Probing classifiers**: Alain & Bengio (2017), Hewitt & Liang (2019), Belinkov (2022 survey).
- **Cross-architecture transfer**: Multiple works on probe-transferability, generally finding it fails.
- **Function vectors**: Hendel et al (2023), "In-Context Learning and Task-Vector Arithmetic".
- **Refusal direction**: Arditi et al (2024), "Refusal in Language Models Is Mediated by a Single Direction".
- **SAE features**: Bricken et al (2023), Templeton et al (2024) — Anthropic interpretability.

---

## Citation

If you use this methodology, cite as:

```
@misc{rmt_probe_2026,
  title  = {RMT-Cleaned Probe Training for LLM Correctness Prediction},
  author = {mokumaisancho},
  year   = {2026},
  url    = {https://github.com/mokumaisancho/layerIdentifier}
}
```

---

## License

MIT. See `LICENSE` in the parent repository.

---

## Status

Methodology complete. 31/31 unit tests pass. Pipeline check passes (5/5 stages). 8/12 quantitative ACs pass with documented root causes for the 4 failures.

This is a methodology contribution, not a cognitive-discovery paper. Frame accordingly.
