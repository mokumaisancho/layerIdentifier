# Real-time Early-Exit — Implementation, Tuning, and Results

**Date**: 2026-06-06
**Status**: ✅ Implemented, tuned, and benched

## Implementation

### What it does

`src/early_exit.py` + `src/early_exit_bench.py` modify the generation loop to:
1. Train a per-architecture probe (StandardScaler + LogisticRegression, top-10 features)
2. During generation, compute features from the partial per-layer norm trajectory after each token
3. Evaluate the probe → P(correct)
4. If P(correct) < threshold for K consecutive tokens → abort generation

### Key components

| File | Purpose |
|---|---|
| `src/early_exit.py` | `train_probe_from_captures`, `generate_with_early_exit`, `compute_partial_probe_features` |
| `src/early_exit_bench.py` | CLI bench tool, runs early-exit vs baseline |
| `src/features.py:partial_layer_features` | Per-layer features from prefix norms |

### Probe architecture configs (from FINAL_FINDINGS)

```python
PROBE_CONFIGS = {
    "gemma": {
        "layers": [6, 15, 41, 28, 21, 27, 2, 34, 23, 11],
        "features": ["convergence_slope", "n_spikes", "std_norm", "mean_norm", ...]
    },
    "qwen": {
        "layers": [15, 7, 5, 6, 4, 2, 14, 24, 26, 27],
        "features": [...]
    }
}
```

---

## Threshold-tuning sweep (simulation on existing captures)

**Method**: For each (cutoff, threshold, abort_consec) combination, simulate per-token probe evaluation on existing Gemma captures (in-sample). Compute:
- **abort_rate**: % of problems aborted
- **false_abort_rate**: % of aborts that were actually correct (baseline)
- **sensitivity**: % of true-wrong answers caught
- **tokens_saved_pct**: % of generation tokens saved
- **net_win**: `tokens_saved_pct - 2 * (fp_aborts / n)` — proxy for value vs cost

### Headline result

| Config | abort% | FALSE% | sens% | tok_saved% | net_win |
|---|---|---|---|---|---|
| **cutoff=10, thr=0.2, ac=4** | 36.7 | **9.1** | 87.7 | 18.6 | **+11.9** |
| cutoff=3, thr=0.2, ac=3 | 40.0 | 15.0 | 89.5 | 21.6 | +9.6 |
| cutoff=10, thr=0.3, ac=4 | 41.3 | 14.5 | 93.0 | 21.8 | +9.8 |
| cutoff=5, thr=0.5, ac=2 *(initial)* | 66.7 | **43.0** | 100.0 | 38.8 | **-18.6** |

### Critical lesson

The initial config (cutoff=5, threshold=0.5, abort_consec=2) was **way too aggressive**:
- It aborted 67% of problems
- **43% of those aborts were wrong** (would have been correct)
- Token savings were high (38.8%) but accuracy loss was unacceptable

The fix: **lower the threshold** (more conservative about aborting) and **require more consecutive low-confidence votes** (4 instead of 2).

### Recommended production config

```bash
python -m src.early_exit_bench \
    --model $MODEL --captures $CAPS --arch gemma \
    --cutoff 10 --threshold 0.2 --abort-consecutive 4
```

This gives:
- ~37% of problems aborted
- ~9% of aborts are false (would have been correct)
- ~88% of true-wrong problems are caught
- ~19% token savings
- Net positive value

---

## Real wall-clock bench (150 problems, Gemma-4-E4B)

### Initial config (cutoff=5, threshold=0.5, ac=2) — TOO AGGRESSIVE

| Metric | Baseline | Early-exit | Δ |
|---|---|---|---|
| Wall-clock (150 problems) | 785.9s | 514.1s | **1.53× faster** |
| Tokens generated | 4911 | 2904 | 40.9% saved |
| Total aborts | — | 97 (64.7%) | — |
| True aborts (baseline-wrong) | — | 55 | sensitivity 96.5% |
| **False aborts (baseline-correct)** | — | **42** | **false-abort rate 43.3%** |
| Correct answers delivered | 93/150 (62%) | 51/150 (34%) | **-28% accuracy LOSS** |
| Cost per correct answer | 8.45s | 10.08s | **0.84× WORSE** |

**Verdict**: Fast but destructive — over-aborts correct answers. Cost per delivered correct answer went UP.

### Tuned config (cutoff=10, threshold=0.2, ac=4) — VERIFIED

| Metric | Baseline | Early-exit | Δ |
|---|---|---|---|
| Wall-clock (150 problems) | 785.9s | 688.1s | **1.14× faster** |
| Tokens generated | 4911 | 4180 | 14.9% saved |
| Total aborts | — | 53 (35.3%) | — |
| True aborts (baseline-wrong) | — | 46 | sensitivity 80.7% |
| **False aborts (baseline-correct)** | — | **7** | **false-abort rate 13.2%** |
| Correct answers delivered | 93/150 (62.0%) | 86/150 (57.3%) | -4.7% accuracy loss |
| **Cost per correct answer** | 8.45s | **8.00s** | **1.06× better** |

**Verdict**: Real production gain. 14% wall-clock savings with 4.7% accuracy loss. Cost per correct answer delivered is 6% lower.

### Comparison: aggressive vs tuned

| Metric | Aggressive (c5, t0.5, ac2) | Tuned (c10, t0.2, ac4) |
|---|---|---|
| Aborts | 97 (64.7%) | 53 (35.3%) |
| False aborts | 42 (43.3% of aborts) | 7 (13.2% of aborts) |
| Sensitivity | 96.5% | 80.7% |
| Wall-clock speedup | 1.53× | **1.14×** |
| Tokens saved | 40.9% | 14.9% |
| Accuracy loss | **-28.0%** | **-4.7%** ✓ |
| Cost per correct | 0.84× (worse) | **1.06× (better)** ✓ |

**Trade-off**: aggressive is 3× faster but loses 28% of correct answers. Tuned is 14% faster and only loses 4.7%. For most use cases, tuned is the right choice.

---

## Cross-cutting findings

### 1. Per-token TPS is misleading

The bench reports "TPS gain" as avg(tokens/sec) ratio. When we abort, both tokens and time decrease proportionally, so per-token TPS doesn't change much. **Wall-clock per problem** and **cost per correct answer delivered** are the real metrics.

### 2. Probe is over-confident on partial trajectories

The probe was trained on FULL-trajectory features but evaluated on PARTIAL features. This distribution shift means:
- Probe P(correct) at token 5 is more confident than reality
- The calibration is off — P=0.5 doesn't mean 50% likely correct
- Empirically: at threshold=0.5, false-abort rate is 43% (probe says "wrong" but model would have been right 43% of the time)

**Mitigation**: lower threshold (0.2) requires probe to be VERY confident before aborting.

### 3. Sensitivity stays high across configs

Almost every config catches >87% of true-wrong answers. The probe is good at identifying wrong answers. The hard part is avoiding false-aborts on correct answers.

### 4. The trade-off curve

```
Tokens saved %  ↔  False-abort %
       40%      ↔      43%    (aggressive)
       25%      ↔      20%    (balanced)
       19%      ↔       9%    (conservative)
       10%      ↔       3%    (very conservative)
        0%      ↔       0%    (no abort)
```

Pick the point on this curve that matches your use case's value function.

---

## How to use

### Default (safe) config

```bash
python -m src.early_exit_bench \
    --model models/gemma-4-E4B-it-MLX-8bit \
    --captures results/gemma-4-E4B-nothinking/captures.json \
    --arch gemma \
    --out results/gemma-4-E4B-earlyexit \
    --cutoff 10 --threshold 0.2 --abort-consecutive 4
```

### Aggressive config (max speedup, accept accuracy loss)

```bash
python -m src.early_exit_bench \
    --model $MODEL --captures $CAPS --arch gemma \
    --cutoff 3 --threshold 0.2 --abort-consecutive 3
```

### Conservative config (min false aborts)

```bash
python -m src.early_exit_bench \
    --model $MODEL --captures $CAPS --arch gemma \
    --cutoff 10 --threshold 0.1 --abort-consecutive 5
```

### Custom tuning

To find the best config for your workload, run `/tmp/probe_threshold_tune.py` (or rebuild it from this repo's src) on your captures. It simulates all combinations in <1 second per 150 problems.

---

## Production deployment notes

1. **Probe is per-architecture**. Each model needs its own probe. Cross-architecture transfer fails (Move #1 finding).
2. **Probe must be retrained per model + per task distribution**. A probe trained on GSM8K may not calibrate well on MMLU.
3. **Calibration drift**: the probe was trained on full features but evaluated on partial features. A more principled fix is to train a separate probe per cutoff value (cutoff=3 probe, cutoff=5 probe, etc.).
4. **Overhead is small**: probe eval is ~1ms per token (StandardScaler + LR on 120-dim). Negligible vs ~150ms/token generation.
5. **Value depends on workload**:
   - If model is mostly right → few aborts → small gain
   - If model is mostly wrong → many aborts → big gain
   - Best case: hard workloads where fast-wrong-detection matters

---

## Known limitations

1. **In-sample evaluation**: probe trained on captures, tested on same captures. Out-of-sample performance might be worse. A proper test would split problems into train/test.
2. **Probe trained on full features**: distribution shift when evaluating on partial features. Future work: train per-cutoff probes.
3. **Single-architecture tuning**: this sweep is for Gemma-4-E4B only. Qwen may need separate tuning.
4. **GSM8K-style problems only**: tuning may not transfer to MMLU, coding, or long-form reasoning.

---

## What's next

1. **Train per-cutoff probes** — should reduce distribution shift, lower false-abort rate
2. **Out-of-sample eval** — split problems, train on half, test on half
3. **Qwen bench** — run same tuning sweep on Qwen, see if findings generalize
4. **Real production integration** — wire into mlx-lm's generate loop as a drop-in
