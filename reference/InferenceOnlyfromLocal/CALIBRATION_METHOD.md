# Scorer Calibration: Mathematical Framework & Optimization

## Problem Statement

The `score_fix_scope()` function classifies model output as GLOBAL or LOCAL fix.
It was scoring the **full model output** including `<think〉...〈/think〉` blocks,
causing inflated scores because thinking text naturally contains GLOBAL keywords
("refactor", "architecture", "shared state") that are irrelevant to the actual fix.

## Mathematical Framework

The problem decomposes into four resolved mathematical results:

### 1. Sufficient Statistics (Fisher 1922, Neyman-Fisher Factorization)

**Theorem**: A statistic S(X) is sufficient for parameter θ iff
```
P(X|θ) = g(S(X), θ) · h(X)
```

**Application**: The scorer should operate on the answer text only (sufficient
statistic), not the thinking+answer concatenation. The thinking block is h(X) —
it carries no information about fix scope but contaminates the measurement.

**Fix**: Strip `<think〉...〈/think〉` blocks before scoring. This constructs a
sufficient statistic by projecting onto the relevant subspace.

### 2. Neyman-Pearson Lemma (1933) — Optimal Threshold

**Theorem**: For binary classification at fixed false-positive rate α, the
optimal decision rule is the likelihood ratio test:
```
Reject H₀ if P(X|H₁) / P(X|H₀) > k
```

**Application**: Our scorer uses arbitrary weights (base=0.30, ±0.15 per keyword,
threshold=0.50). The NP lemma says the optimal threshold depends on the
class-conditional distributions P(score|GLOBAL) and P(score|LOCAL), which we
don't know yet.

**Fix**: After collecting labeled data, set threshold at the NP-optimal point
for the desired false-positive rate.

### 3. Calibration Theory (Platt 1999, Zadrozny & Elkan 2002)

**Problem**: Raw scorer output (0.65, 0.80, etc.) is not a calibrated probability.

**Solution**: Map raw scores → true probabilities:
- **Platt scaling**: `P(GLOBAL|s) = 1 / (1 + exp(As + B))` — 2 parameters
- **Isotonic regression**: Monotone step function (more flexible, needs more data)
- **Venn-ABERS predictors** (Mervin et al. 2020): Probability intervals, best
  for small calibration sets

**Key result from Yousef et al. (2021, IEEE TDSC)**: Platt scaling is "very
stable and acceptable" across all experiments, even at small sample sizes.

### 4. Active Learning — Sample Efficiency

**Problem**: 150 benchmark records × 60s/record = 2.5h GPU time.

**Solution**: Active learning reduces required samples by 3-5x (multiple papers).
Query-by-uncertainty: run only the records where the scorer is most uncertain,
maximizing information per GPU minute.

## Optimized Pipeline

### Phase 0: Structural Fix (0 GPU min)
- Strip thinking blocks from scorer input
- Save full outputs (not 200-char previews)
- **Result**: Sufficient statistic constructed

### Phase 1: Balanced Data Collection (~34 min GPU)
- Re-run 14 GLOBAL records with full output capture (~14 min)
- Run 20 LOCAL records (balanced sampling from 114 LOCAL in benchmark) (~20 min)
- Total: 34 labeled samples, 42% GLOBAL / 58% LOCAL

### Phase 2: Calibration (0 GPU min)
- Fit Platt scaling on 34 samples → calibrated P(GLOBAL|score)
- Parameter ratio: 34:2 = 17:1 (comfortable)
- Alternative: Venn-ABERS if Platt underfits

### Phase 3: Classifier Training (0 GPU min)
- Extract features: raw score, n_tokens, thinking_ratio, code_indicators
- Train LogisticRegression with L1 regularization on 34 samples
- Feature ratio: 34:5 ≈ 7:1 (workable with regularization)
- Evaluate with leave-one-out CV

### Phase 4: NP Threshold (0 GPU min)
- Plot ROC curve on 34 calibrated samples
- Set threshold at NP-optimal point for desired false-positive rate
- Report: AUROC, ECE, confusion matrix at optimal threshold

### Phase 5 (Optional): Active Learning (~10 min GPU)
- Score remaining 116 records with calibrated scorer
- Run only the 10-15 most uncertain samples
- Refine calibration

**Total GPU time: 34-45 min (vs 2.5h naïve = 3-4x reduction)**

## Data Requirements

| Method | Parameters | Min Samples | Ratio | Our N |
|--------|-----------|-------------|-------|-------|
| Platt scaling | 2 (A, B) | 10-20 | 5:1+ | 34 ✅ |
| Isotonic regression | ~10 steps | 50+ | 5:1 | 34 ⚠️ |
| LR classifier (5 features) | 5+1 | 30-60 | 6:1+ | 34 ✅ |
| LR classifier (30 features) | 30+1 | 300+ | 10:1 | 34 ❌ |

## Current Scorer: Known Issues

| Component | Current | Correct | Status |
|-----------|---------|---------|--------|
| Input | Full output (thinking+answer) | Answer only | ✅ Fixed |
| Scorer function | Regex keyword count | Trained classifier | ❌ Placeholder |
| Weights | Hand-tuned (0.15, 0.30) | Learned from data | ❌ Pending |
| Threshold | Arbitrary 0.50 | NP-optimal | ❌ Pending |
| Score → probability | Raw score | Platt-scaled | ❌ Pending |

## References

- Fisher, R.A. (1922). "On the mathematical foundations of theoretical statistics"
- Neyman, J. & Pearson, E.S. (1933). "On the problem of the most efficient tests of statistical hypotheses"
- Platt, J. (1999). "Probabilistic outputs for support vector machines"
- Zadrozny, B. & Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates"
- Yousef, W. et al. (2021). "Classifier Calibration: With Application to Threat Scores in Cybersecurity" IEEE TDSC. arXiv:2102.05143
- Mervin, L. et al. (2020). "Comparison of Scaling Methods to Obtain Calibrated Probabilities" JCIM. doi:10.1021/acs.jcim.0c00476
- Zhang, J. et al. (2023). "Transferable Post-hoc Calibration on Pretrained Transformers in Noisy Text Classification" AAAI
