# Mechanistic Investigation: What is Gemma L6 Reading?

**Date**: 2026-06-06
**Status**: ✅ N=150 partial results in — N=30 "L4 specific writer" story REFUTED. L5 + L0_L5_all conditions still running.

## 🔬 N=150 update (2026-06-06, partial — 6/8 conditions complete)

Re-ran at proper statistical power. N=150 = 50 problems × 3 seeds. Baseline AUROC 0.826 (vs 0.702 at N=30 — both well above chance). The original N=30 run was killed before L5 + L0_L5_all completed; a completion run is in progress to fill those gaps.

### N=150 ablation results (so far)

| Ablation | L6 AUROC | Δ vs baseline | Accuracy | Interpretation |
|---|---|---|---|---|
| Baseline | **0.826** | — | 93/150 (62.0%) | Solid headline AUROC |
| L0 | 0.500 | -0.326 | 0/150 (0%) | Catastrophic — model broken, AUROC undefined |
| L1 | 0.624 | -0.203 | 29/150 (19%) | Model mostly broken |
| **L2** | **0.504** | **-0.322** | 91/150 (61%) | 🎯 **CLEAN — accuracy preserved, L6 signal gone** |
| L3 | 0.682 | -0.144 | 84/150 (56%) | Moderate drop — partial contribution |
| **L4** | **0.536** | **-0.290** | 88/150 (59%) | 🎯 **CLEAN — accuracy preserved, L6 signal gone** |
| L5 | ⏳ in progress | | | re-running |
| L0-L5 all | ⏳ in progress | | | re-running |

### 🔥 Headline finding: DISTRIBUTED, not localized

**The "L4 is the specific writer" story is REFUTED at N=150.** L2 and L4 each independently destroy the L6 probe (AUROC → 0.50 / 0.54) while preserving accuracy (61% / 59% vs baseline 62%). The L6 probe reads a **distributed signal written across multiple early layers (L2-L4)**, not a single layer's output.

This is a cleaner, more publishable story than the original N=30 claim:
- N=30 said: "L4 writes, L5 modulates, L6 reads" (overstated, retracted)
- N=150 says: "L6 reads a distributed early-layer signal; L2 and L4 are both necessary"

### What's still unknown (in progress)

- L5 ablation effect (was 0.99 at N=30; need N=150 confirmation)
- L0-L5 simultaneous ablation effect
- Whether the distributed pattern replicates in Qwen

### Statistical confidence

At N=150, SE_null for AUROC ≈ 0.044. Minimum detectable Δ ≈ 0.13 (vs 0.44 at N=30). The L2 and L4 effects (Δ = -0.32, -0.29) are **8σ and 7σ** respectively — overwhelmingly significant.

---

## ⚠️ Important correction (2026-06-06 evening)

Initial N=30 findings (below) were overstated. With N=30 problems and only 4 wrong examples, the **minimum detectable AUROC difference is 0.44**. None of the probe-specific effects reach significance:

| Ablation | Δ AUROC | z | p | Verdict |
|---|---|---|---|---|
| L0 | -0.20 | -0.91 | 0.37 | ns |
| L1 | -0.06 | -0.26 | 0.80 | ns |
| L2 | -0.05 | -0.22 | 0.83 | ns |
| L3 | -0.01 | -0.04 | 0.97 | ns |
| L4 | -0.13 | -0.60 | 0.55 | ns |
| L5 | +0.29 | +1.29 | 0.20 | ns |
| L0-L5 all | -0.20 | -0.91 | 0.37 | ns |

**Only accuracy effects are real** (binomial test, not AUROC):
- L0 ablation: 26/30 → 0/30 correct (catastrophic)
- L1 ablation: 26/30 → 10/30 correct (severe)

The "L4 writes / L5 modulates / L6 reads" story is **not supported** at this sample size. It may be true, but we cannot tell from N=30.

A re-run at N=150 is in progress (`results/mechanistic_L6_N150/`, ~3 hours wall-clock).

---

## Original (overstated) findings, kept for reference

**DO NOT CITE** these without first checking the N=150 results.

### Original setup

- Script: `src/mech_ablation.py`
- Method: IdentityLayer ablation (replace layer's `__call__` with input passthrough)
- Scope: 30 problems × 8 conditions

### Original per-layer ablation results (N=30)

| Ablation | L6 AUROC | Δ vs baseline | Accuracy | Verdict |
|---|---|---|---|---|
| Baseline | 0.702 | — | 26/30 | reference |
| L0 | 0.500 | -0.20 | 0/30 | 💥 catastrophic (real) |
| L1 | 0.645 | -0.06 | 10/30 | accuracy drop (real) |
| L2 | 0.654 | -0.05 | 26/30 | noise |
| L3 | 0.692 | -0.01 | 26/30 | noise |
| L4 | 0.567 | -0.13 | 26/30 | ⚠️ NOT significant (was claimed as "probe break") |
| L5 | 0.990 | +0.29 | 26/30 | ⚠️ NOT significant (was claimed as "probe enhancement") |
| L0-L5 all | 0.500 | -0.20 | 0/30 | catastrophic (real) |

### Statistical power analysis

With N=30 and 4 wrong examples:
- SE_null for AUROC = 0.158
- Minimum detectable Δ (alpha=0.05, two-sided) = 0.44
- All probe Δ values are well below 0.44

For robust mechanistic claims we need:
- N=150 → 4× more wrong examples (~57) → min detectable Δ ≈ 0.13
- Or bootstrap CIs on the differences

## What's actually known (with confidence)

1. **L0 is structurally essential.** Ablating it crashes the model.
2. **L1 carries accuracy-critical signal.** Ablating it drops accuracy by ~65%.
3. **L2-L5 ablations preserve accuracy** (small N study, but consistent).
4. **Probe-specific effects at L4/L5 are inconclusive** at N=30.

## What's NOT yet known

- Whether L4 ablation specifically breaks the L6 probe (need N=150)
- Whether L5 ablation enhances the L6 probe (need N=150)
- Whether Hypothesis C (residual magnitude) explains the probe signal (need new test)
- Whether the pattern replicates in Qwen (need Qwen ablation)
- What L5 actually computes (need attention pattern analysis)

## Updated plan

| Step | Purpose | Status |
|---|---|---|
| **Re-run ablation at N=150** | Statistical power | ✅ partial (6/8 conditions); L5 + L0_L5_all in progress |
| Hypothesis C (magnitude probe) | Boring-signal test | ✅ REFUTED (L6 alone > L0-L5 alone) |
| Qwen replication | Cross-architecture | 🔄 queued behind Gemma N=150 |
| L5 attention investigation | Mechanistic depth | pending L5 N=150 result |

## Lessons

1. **Statistical power matters for mechanistic claims.** AUROC differences need Δ >~0.4 to be meaningful at N=30. Anything smaller is in the noise band.
2. **Don't claim mechanistic stories from underpowered experiments.** The L4-writes/L5-modulates/L6-reads story was based on Δ=-0.13 and Δ=+0.29 — both well within noise.
3. **Accuracy effects are easier to detect than AUROC effects.** Binomial tests on accuracy are more powerful than rank-order tests on continuous scores.
4. **The user was right to question the AUROC 0.99 result.** That's a near-perfect score from N=30 (4 wrong) — should always be sanity-checked.

## Files

- `src/mech_ablation.py` — supports `--n`, `--save-norms`, `--conditions` flags
- `results/mechanistic_L6/ablation_results.json` — N=30 results (underpowered, retracted)
- `results/mechanistic_L6_N150/partial_results_from_killed_run.json` — N=150 partial (6/8 conditions)
- `results/mechanistic_L6_N150_completion/` — N=150 completion run (L5 + L0_L5_all), in progress
