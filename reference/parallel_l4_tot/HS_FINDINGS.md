# Hidden State Signal Integration — Findings

## What Was Tested

Added L12 late-spike + L10 mid-spike hidden state signals to the parallel_l4_tot pipeline:
- `LayerCaptureWrapper` intercepts L2 norms from L10 and L12 during generation
- `PathScorer` computes `hs_score = sigmoid(L12) * sigmoid(L10)` with calibrated weights
- Comparison script scores paths once with HS (weight 0.10) and once without

## Results

| Test | N | Majority | No-HS | With-HS | Delta |
|------|---|----------|-------|---------|-------|
| Original (no verifiers) | 20 | 30% | 25% | 25% | 0pp |
| Fixed (with verifiers) | 5 | 0% | 0% | 0% | 0pp |

- **0 flips** in both directions across all problems
- **AUROC 0.44** (down from 0.76 in stepDecomposer v3)
- **Avg hs_score 0.153** — contributes only ~1.5% to composite

## Why It Failed

1. **Calibration doesn't transfer.** stepDecomposer v3 used temps 0.1/0.5/0.9; pipeline uses 0.1/0.7/1.2. Different sampling dynamics → different spike distributions.
2. **Wrong abstraction level.** HS signals predict per-sample correctness. The pipeline uses them for per-path selection. Path-level aggregation (mean/max of spikes) destroys the signal.
3. **Not the bottleneck.** The full pipeline achieves **80%** on 50 problems via step-level PAL + verification. The path scorer alone achieves **25%**. Improving the scorer from 25% to 26% has negligible impact on overall pipeline accuracy.

## Conclusion

**Path-level HS integration has negative ROI.** Stop investing in it.

## Where HS Might Actually Help

**Step-level early exit gate.** The gate currently uses only entropy features. Adding L12/L10 norms to trigger ESCALATE during generation (not after) would:
- Catch bad reasoning before wasting tokens
- Operate at the same abstraction level where HS was calibrated
- Affect every step, not just final path selection

This requires re-training the gate classifier with 7 features (5 entropy + 2 HS) instead of 5.
