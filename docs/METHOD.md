# Method — how layerIdentifier identifies entropy-emitting layers

This document is the canonical method reference. Anything not in this file is
either in `reference/parallel_l4_tot/` (copied, not modified) or in `src/`.

## 1. Hypothesis

A transformer emits per-token entropy `H_t = -Σ p_t · log p_t` over its output
vocabulary at every generation step. The entropy vector `{H_0, H_1, …, H_T}`
reflects the model's per-step uncertainty.

Hidden state activations at each layer `l ∈ [0, N)` produce a per-token L2-norm
scalar `‖h_l_t‖` that reflects the layer's internal activation magnitude.

**Claim**: certain layers' L2-norm time series correlate with the entropy
time series in a way that predicts downstream task outcomes (correct vs wrong).
The set of layers where this correlation is strongest is the model's
"entropy-emission layer set".

## 2. Capture primitives

### 2a. LayerCaptureWrapper (`src/layer_capture.py`)

Wraps `model.layers[i]`. For every forward pass during generation:

1. Calls the wrapped layer normally: `out = layer(*args, **kwargs)`.
2. Reads the layer's output tensor `out`.
3. Selects the last token of batch 0: `last = out[0, -1]` (or `out[-1]` if 2D).
4. Stores `‖last‖_2` as a float in `self.norms`.

Memory: 8 bytes/layer/token. For 42 layers × 256 tokens = ~85 KB per generation.

### 2b. Per-token entropy (`src/entropy_pipeline.py`)

After each `generate_step`:

```python
probs = softmax(logits)
ent = -Σ p · log(p + 1e-10)
entropies.append(float(ent))
```

The `entropy_vec` is a length-T float array.

## 3. Feature extraction (`src/features.py`)

All features use identical formulas to Qwen reference.

### Per-vector (entropy OR layer norms)

| Feature | Formula | Qwen reference |
|---|---|---|
| `mean_entropy` | `mean(vec)` | wired |
| `max_entropy` | `max(vec)` | wired |
| `first_token_entropy` | `vec[0]` | wired |
| `entropy_std` | `std(vec)` | wired |
| `n_spike_tokens` | `‖{x ∈ vec : x > mean + 1.5σ}‖` | wired |
| `max_positive_delta` | `max(diff(vec))` | wired, AUROC 0.718 |
| `max_negative_delta` | `min(diff(vec))` | discovered (L24) |
| `delta_variance` | `var(diff(vec))` | wired |
| `delta_mean`, `delta_std` | mean/std of diff | wired |
| `convergence_slope` | `polyfit(arange, diff, 1)[0]` | discovered (L22/L24) |
| `mid_delta_sum` | `sum(diff[15%:60%])` | discovered (L22/L24) |
| `n_delta_spikes` | count of large abs(diff) values | novel |

### Zone-normalized spike ratios

Per layer `l`:
- `early_spike_ratio_l` = (spikes in first 15%) / (all spikes)
- `mid_spike_ratio_l` = (spikes in 15-60%) / (all spikes)  ← Qwen L10 anti-signal
- `late_spike_ratio_l` = (spikes in last 40%) / (all spikes)  ← Qwen L12 normal signal

Returns 0.5 (neutral) when no spikes or empty vec.

## 4. Calibration

For each layer `l` and feature `f`, compute AUROC of `feature_value` against
`correct` (binary) across all problem-seed runs:

```python
scores = [c.layer_features[f"L{l}_{f}"] for c in captures]
labels = [1 if c.final_correct else 0 for c in captures]
auroc = max(MannWhitneyU(scores, labels), 1 - MannWhitneyU(scores, labels))
direction = "+" if higher→correct else "-"
```

Report `max(auroc, 1-auroc)` and direction. This is comparable to the Qwen
AUROC table in `reference/parallel_l4_tot/SIGNAL_CATALOG.md`.

## 5. Known confounds (from Qwen reference — apply equally here)

1. **Temperature** — Qwen L12 calibration at temps 0.1/0.5/0.9 did NOT transfer
   to 0.1/0.7/1.2. We use 0.1 throughout.
2. **Output format** — integrator_count measures PAL vs CoT, not reasoning quality.
   We use PAL (`result = X` prompt) to match InferenceLLM2 convention.
3. **Path-level vs step-level** — Qwen HS path-level integration had AUROC 0.44
   (negative ROI). The signal works at step-level early-exit gates. We capture
   path-level for cross-model comparability; step-level is Phase 2.

## 6. What the ranking tells us

For Gemma, the output `layer_ranking.md` will list, per layer, the best feature
and its AUROC. The interpretation:

- **AUROC > 0.6** with reasonable support (≥ 20 captures, balanced) → real signal.
- **AUROC > 0.65** → comparable to Qwen's best layers.
- **AUROC < 0.55** → no signal at this layer.
- **Top-5 layers by AUROC** → the model's "entropy-emission layer set".

Compare to Qwen's reported layers (relative depth):

| Qwen layer | Relative depth | Feature | AUROC |
|---|---|---|---|
| L10 | 0.31 | mid_spike_ratio (anti) | 0.62 |
| L12 | 0.375 | late_spike_ratio | 0.65 |
| L18 | 0.56 | mid_delta_sum | 0.59 |
| L22 | 0.69 | mid_delta_sum, convergence_slope | 0.62 |
| L24 | 0.75 | max_negative_delta | 0.65 |

If Gemma's top-5 layers cluster at similar relative depths (0.3-0.4 and 0.7-0.8),
that's evidence the entropy-layer phenomenon is **architectural**, not Qwen-specific.

## 7. Phase 2 (not implemented in this repo yet)

- AttentionProbeWrapper — per-head attention weights on full-attention layers.
- Step segmentation — split generation into reasoning steps, compute per-step features.
- Step-level early-exit gate — train classifier on (entropy + per-layer L2 norms) → continue/abort.
- Per-layer probe training — LR/MLP on raw hidden state (using `keep_raw=True`).
