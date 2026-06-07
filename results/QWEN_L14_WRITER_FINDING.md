# Qwen L14 Dominant-Writer Finding (Move #5 Qwen replication)

**Date**: 2026-06-07
**Predecessor**: Move #5 (Gemma N=150, completed 2026-06-06 23:34 JST)
**Status**: COMPLETE — Qwen writer topology characterized

---

## 1. Word-for-word data (from `results/mechanistic_L15_qwen/ablation_results.json`)

```json
{
  "baseline": {
    "auroc_L15_n_spikes": 0.9189938601703308,
    "elapsed_s": 448.1381388329901,
    "n_correct": 99,
    "n_problems": 150
  },
  "ablate_L12": {
    "auroc_L15_n_spikes": 0.7504011410233553,
    "delta_auroc_vs_baseline": -0.1685927191469755,
    "n_correct": 79,
    "delta_correct_vs_baseline": -20,
    "elapsed_s": 543.6777818750124
  },
  "ablate_L13": {
    "auroc_L15_n_spikes": 0.7005,
    "delta_auroc_vs_baseline": -0.2184938601703308,
    "n_correct": 100,
    "delta_correct_vs_baseline": 1,
    "elapsed_s": 364.7182552078739
  },
  "ablate_L14": {
    "auroc_L15_n_spikes": 0.5594315245478036,
    "delta_auroc_vs_baseline": -0.35956233562252715,
    "n_correct": 21,
    "delta_correct_vs_baseline": -78,
    "elapsed_s": 658.1034044581465
  },
  "ablate_L0_L14_all": {
    "auroc_L15_n_spikes": 0.5,
    "delta_auroc_vs_baseline": -0.4189938601703308,
    "n_correct": 0,
    "delta_correct_vs_baseline": -99,
    "elapsed_s": 490.3908533330541
  }
}
```

Verbatim monitor event log:
```
L15 AUROC=0.701 (Δ=-0.218)  correct=100/150 (Δ=+1)  *** BREAKS  (364.7s)   [ablate_L13]
=== ABLATE L14 ===
L15 AUROC=0.559 (Δ=-0.360)  correct=21/150  (Δ=-78)  *** BREAKS  (658.1s)  [ablate_L14]
=== ABLATE L0..L14 ALL ===
L15 AUROC=0.500 (Δ=-0.419)  correct=0/150   (Δ=-99)             (490.4s)   [ablate_L0_L14_all]
=== VERDICT ===
Baseline L15 n_spikes AUROC: 0.919
ablate_L12: Δ=-0.169  *** BREAKS
ablate_L13: Δ=-0.218  *** BREAKS
ablate_L14: Δ=-0.360  *** BREAKS
ablate_L0_L14_all: Δ=-0.419  BREAKS
```

## 2. Tabulated result

| Ablation | L15 n_spikes AUROC | Δ AUROC | n_correct | Δ correct | Wall (s) | Verdict |
|---|---|---|---|---|---|---|
| baseline | 0.919 | — | 99/150 | — | 448 | — |
| L12 | 0.750 | −0.169 | 79 | −20 | 544 | BREAKS |
| L13 | 0.700 | −0.218 | 100 | +1 | 365 | BREAKS (no acc drop) |
| L14 | 0.559 | **−0.360** | 21 | **−78** | 658 | BREAKS (dominant) |
| L0..L14 all | 0.500 | −0.419 | 0 | −99 | 490 | probe at chance |

## 3. Analysis

### L14 is the dominant writer

L14 single ablation accounts for **0.360 / 0.419 = 86%** of the total upstream contribution to the L15 probe signal. The remaining 14 layers (L0..L13) together add only 0.059 AUROC on top of L14 alone.

### L12 and L13 are redundant backups

If L12, L13, L14 contributed independently, the sum of single-layer ablations would predict the all-ablation effect: Σ = 0.169 + 0.218 + 0.360 = **0.747**. Observed all-together = **0.419**. Redundancy = 0.328 AUROC — meaning these three layers overlap heavily in what they write to L15.

### L13 ablation breaks the probe but not accuracy

The L13 ablation is unusual: probe AUROC drops by 0.218 (large), but accuracy actually rises by +1 (100 vs 99). L13 contributes to the discriminative feature (n_spikes) but not to the model's answer correctness. Possible interpretations:
- L13 writes noise that the probe reads as signal
- L13 is part of the writer module but its contribution is functionally redundant
- L13 is a "modulator" similar to Gemma L5 (which enhanced signal when ablated)

The +1 accuracy change is within sampling noise (1/150 ≈ 0.7%) — do not over-interpret.

### L14 is also critical for accuracy (not just probe)

L14 ablation: n_correct drops from 99 → 21 (−78). L0..L14 all ablation: 99 → 0 (−99). L14 alone accounts for 79% of the accuracy drop. L14 is functionally central to both probe signal AND answer generation.

## 4. Cross-architecture comparison

| Property | Gemma (N=150, Move #5) | Qwen (this finding) |
|---|---|---|
| Probe layer | L6 | L15 |
| Probe AUROC | 0.981 | 0.919 |
| Writer topology | Distributed (L2 + L4 joint) | Concentrated (L14 alone) |
| Backup writers | Yes (multi-layer) | L12/L13 redundant |
| Modulator layer | L5 (ablation ENHANCES) | L13 (possible — needs follow-up) |
| Single dominant writer? | No | **Yes (L14)** |
| L0..L_probe_all ablation AUROC | (TBD if not run) | 0.500 (chance) |

**Asymmetry**: Gemma's signal arises from distributed early-layer writing; Qwen's signal arises from a single late-layer writer just before the probe. Different computational topologies achieve similar probe-discriminability.

## 5. Implications

1. **Move #1 transfer failure (probes don't transfer Gemma↔Qwen)** has a second independent cause beyond feature-family mismatch (Move #8 RMT cosine=0.47): **write locations differ architecturally**. Early-Gemma vs late-Qwen writers cannot transfer by construction.
2. **For activation steering / "modes"**: Qwen target = **L14** (single layer, dominant). Gemma target = L2+L4 jointly. Different intervention sites.
3. **For mechanistic narrative**: Qwen has a focused writer→probe pair (L14→L15) suitable for neuron-level follow-up. Gemma's distributed writer web resists single-site intervention.
4. **Why Qwen probe AUROC (0.996 raw, 0.919 L15-n_spikes-only) is higher than Gemma (0.981)**: concentrated late-layer signal is easier to read than distributed early-layer signal.

---

## 6. ROI-ranked next steps

Ordered by expected value × probability / cost.

### Rank 1 — L14-as-probe feature check (OBVIOUS; runs now)
**Question**: Does L14 itself carry the discriminative feature, or does it merely WRITE the feature to L15 without holding it?
**Method**: Train LR probe on L14's 14 features only (n_spikes, mean_norm, etc.), compare AUROC to L15 n_spikes baseline (0.919).
**Cost**: 5 min, uses existing captures.json (no model run).
**Outcome**:
- L14 AUROC ≈ 0.9 → L14 carries the signal at the feature level (passive holder + active writer)
- L14 AUROC ≈ 0.5 → L14 writes signal to L15 via a different mechanism (e.g., attention routing) without holding it
- L14 AUROC in (0.6, 0.85) → partial: L14 has the signal but L15 amplifies it
**ROI**: Highest. Cheap, direct extension, decides what L14 actually does.

### Rank 2 — L11 and L10 backward ablation (bounds writer region)
**Question**: Does the writer region extend earlier than L12?
**Method**: Ablate L11 alone, then L10 alone. Measure Δ AUROC.
**Cost**: 10 min × 2 = 20 min model time.
**Outcome**:
- Both < 0.05 → writer region confirmed L12-L14 (clean boundary)
- L11 ≥ 0.10 → writer region extends backward; re-test L8-L10
**ROI**: High. Closes the spatial loop on writer localization.

### Rank 3 — L13 modulator follow-up
**Question**: Is Qwen L13 a "modulator" (like Gemma L5)? L13 ablation dropped probe AUROC by 0.218 but accuracy +1.
**Method**: L13 ablation, larger N (300 problems), check if accuracy Δ is reliably 0 or slightly positive.
**Cost**: ~15 min model time.
**Outcome**: Identifies Qwen's "noise layer" analog to Gemma L5.
**ROI**: Medium. Refines mechanistic story but not critical.

### Rank 4 — L14 activation addition steering (causal intervention)
**Question**: Can we add a "correct direction" to L14 activations during inference and flip incorrect → correct predictions?
**Method**: Compute mean L14 activation difference between correct/incorrect problems; add scaled direction to L14 during forward pass on incorrect problems; measure flip rate.
**Cost**: 45-90 min, requires model + hooks + careful β-sweep.
**Outcome**: Tests causality. If flip rate > 30% at any β, intervention works → publishable.
**ROI**: Highest publishable value but highest cost.

### Rank 5 — L14 single-neuron importance analysis
**Question**: Which neurons in L14 carry the writer signal?
**Method**: Extract L14 activations on all 150 problems, compute per-neuron AUROC for correctness, identify top-K neurons.
**Cost**: 30-60 min, requires model run with hooks.
**Outcome**: Neuron-level targets for future fine-grained intervention.
**ROI**: High but more expensive than Rank 1-2.

### Skipped options (low ROI or out of scope)
- Cross-arch probe retrain (Move #1): already shown to fail by 2 independent methods.
- L15 self-ablation: probe reads from L15, ablation destroys probe trivially (not informative).
- L16+ ablation: downstream of probe, cannot affect probe AUROC.
- Gemma L0..L6_all ablation: would mirror Qwen's all-ablation but is structurally different (Gemma probe is at L6, no downstream writing).

---

## 7. Obvious next step selected

**Rank 1: L14-as-probe feature check** is the obvious next step. It:
- Uses existing data (no model run, no caching, no reset issues)
- Runs in 5 minutes
- Directly extends the L14 dominant-writer finding to the feature level
- Disambiguates "writer" vs "carrier" interpretation of L14

This step is taken in the same script that runs immediately after this document is written. No persistent state — pure re-analysis of `captures.json`.

## 8. L14-as-probe check result (Rank 1 next step — EXECUTED 2026-06-07)

Pure re-analysis of existing `captures.json`. No model run, no caching. Script: `src/qwen_l14_as_probe.py`. Output: `results/qwen_l14_as_probe.json`.

| Layer | CV AUROC | Bootstrap 95% CI | Top feature (single-feat AUROC) |
|---|---|---|---|
| L12 | 0.832 ± 0.058 | [0.826, 0.879] | L12_convergence_slope (0.843) |
| L13 | 0.842 ± 0.054 | [0.826, 0.885] | L13_convergence_slope (0.842) |
| **L14** | **0.886 ± 0.039** | [0.877, 0.921] | L14_n_spikes (0.857) |
| L15 | 0.910 ± 0.066 | [0.889, 0.930] | L15_n_spikes (0.889) |

**Difference L15 − L14 = +0.025** (negligible). L14 holds 97% of L15's discriminative power at the feature level.

### Interpretation refined

L14 is BOTH carrier AND writer:
- Carrier: L14's own 14 features predict correctness at AUROC 0.886 (nearly matches L15's 0.910).
- Writer: When L14 is ablated, L15 AUROC drops by 0.360 — confirming L14 actively writes its feature signal to L15.

The transfer efficiency from "L14 holds signal" to "L15 reflects signal" is ~93% (0.360 / (0.886 − 0.5) ≈ 0.93). L14's contribution to L15 is almost perfectly linear.

### Monotone gradient L12 → L15

Discriminability grows by ~0.025 per layer on average. The signal is essentially complete by L14 (only +0.025 remaining to L15). This is consistent with iterative refinement: each layer adds a small amount of correctness-predictive signal, with the largest single jump at L13→L14 (+0.044).

### All top features are negative-direction

Higher spike counts / norms → lower probability of correct. Errors are characterized by more entropy bursts and higher per-layer norms. Consistent with the "errors are noisier" hypothesis.

## 9. Files produced

| File | Purpose |
|---|---|
| `results/QWEN_L14_WRITER_FINDING.md` (this file) | Word-for-word finding + analysis + ROI ranking |
| `results/mechanistic_L15_qwen/ablation_results.json` | Raw ablation numbers |
| `results/qwen_l14_as_probe.json` | L14-as-probe check output |
| `src/qwen_l14_as_probe.py` | Per-layer probe analysis script |

---

## Execution log

| Timestamp (JST) | Event |
|---|---|
| 2026-06-07 00:00 | L12 ablation BREAKS Δ=−0.169 (control, from prior session) |
| 2026-06-07 00:10 | L13 ablation started |
| 2026-06-07 ~00:16 | L13 ablation BREAKS Δ=−0.218 |
| 2026-06-07 ~00:16 | L14 ablation started |
| 2026-06-07 ~00:27 | L14 ablation BREAKS Δ=−0.360 |
| 2026-06-07 ~00:27 | L0..L14 all ablation started |
| 2026-06-07 ~00:36 | L0..L14 all BREAKS Δ=−0.419 (probe at chance, correct=0) |
| 2026-06-07 ~00:40 | This finding document written; L14-as-probe check started |
