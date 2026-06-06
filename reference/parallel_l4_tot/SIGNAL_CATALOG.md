# Signal Catalog — parallel_l4_tot Pipeline

Complete inventory of extractable signals, organized by source layer and whether currently wired into production.

---

## 1. Entropy Features (Pass 1 — Currently Wired)

Extracted per-token during generation in `generate_with_entropy()`.

| Feature | Type | Description |
|---|---|---|
| `mean_entropy` | float | Mean per-token entropy across full generation |
| `max_entropy` | float | Peak entropy value (single token) |
| `first_token_entropy` | float | Entropy of first generated token |
| `entropy_std` | float | Standard deviation of per-token entropy |
| `n_spike_tokens` | int | Count of tokens exceeding mean + 1.5σ |
| `spike_positions` | list[int] | Token indices where spikes occur |

**Derived delta features (wired, computed from `entropy_vec`):**

| Feature | Type | Description | AUROC |
|---|---|---|---|
| `max_positive_delta` | float | Largest entropy increase between consecutive tokens | 0.718 |
| `delta_variance` | float | Variance of entropy delta sequence | 0.597 |
| `delta_mean` | float | Mean of entropy deltas |
| `delta_std` | float | Std dev of entropy deltas |

---

## 2. Temporal Features (Currently Wired)

Derived from entropy spikes in `extract_temporal_features()`.

| Feature | Type | Description |
|---|---|---|
| `early_spike_ratio` | float | Fraction of spikes in first 15% of generation |
| `mid_spike_ratio` | float | Fraction of spikes in 15-60% zone |
| `late_spike_ratio` | float | Fraction of spikes in last 40% zone |
| `recovery_indicator` | float | 1.0 if late_spikes > early_spikes |
| `late_minus_early` | float | Normalized (late - early) spike count |
| `spike_amplitude_proxy` | float | max_entropy / n_spikes |
| `spike_density` | float | n_spikes / n_tokens |
| `spike_spread` | float | Std dev of spike positions |
| `spike_cv` | float | Coefficient of variation of spike positions |
| `q1_spike_density` | float | Quarter 1 (0-25%) spike density |
| `q2_spike_density` | float | Quarter 2 (25-50%) spike density |
| `q3_spike_density` | float | Quarter 3 (50-75%) spike density |
| `q4_spike_density` | float | Quarter 4 (75-100%) spike density |
| `entropy_trend_slope` | float | Linear slope of quarter densities |
| `max_mean_ratio` | float | max_entropy / mean_entropy |
| `entropy_concentration` | float | (max - mean) / mean |
| `first_token_relative` | float | first_token_entropy / max_entropy |

---

## 3. Hidden State Norm Features (Currently Wired)

Captured via `LayerCaptureWrapper` intercepting L2 norms during generation.

### 3a. Raw Norm Spike Ratios

| Feature | Layer | Zone | Calibration | Direction |
|---|---|---|---|---|
| `L12_late_spike_ratio` | L12 | Late (>60%) | mean=0.5127, std=0.2343 | +1.0 (normal) |
| `L10_mid_spike_ratio` | L10 | Mid (15-60%) | mean=0.4133, std=0.2109 | -1.0 (anti-signal) |

**Composite:** `hs_score = sigmoid(L12) * sigmoid(L10)` (calibrated, weight 0.10 in PathScorer)

### 3b. Delta Features (Discovered, NOT Wired)

Require storing full norm sequence and computing token-to-token deltas.

| Feature | Layer | AUROC | Strength |
|---|---|---|---|
| `max_negative_delta` | L24 | 0.652 | Moderate |
| `mid_delta_sum` | L22 | 0.628 | Moderate |
| `convergence_slope` | L24 | 0.623 | Moderate |
| `convergence_slope` | L22 | 0.622 | Moderate |
| `n_spikes` (delta) | L24 | 0.616 | Moderate |
| `mid_delta_sum` | L24 | 0.604 | Moderate |
| `mid_delta_sum` | L18 | 0.593 | Weak |

---

## 4. Verification Signals (Currently Wired)

Per-path or per-step verification results.

| Signal | Type | Description |
|---|---|---|
| `z3_valid` | bool | Chain-level Z3 verification passed |
| `z3_confidence` | float | Chain-level confidence score |
| `z3_arith_valid` | bool | Arithmetic-level Z3 passed |
| `z3_arith_confidence` | float | Arithmetic-level confidence |
| `logic_valid` | bool | Logic verifier passed |
| `code_valid` | bool | Code verifier passed |

---

## 5. Scoring Signals (Currently Wired)

Produced by `PathScorer` or gates.

| Signal | Type | Source | Weight |
|---|---|---|---|
| `temporal_score` | float | Trained probe on temporal features | 0.25 |
| `l4_score` | float | EntropyOnlyGate confidence | 0.15 |
| `hs_score` | float | L12×L10 sigmoid composite | 0.10 |
| `z3_signal` | float | Z3 arithmetic confidence | 0.20 |
| `logic_score` | float | 1.0 if logic_valid | 0.15 |
| `code_score` | float | 1.0 if code_valid | 0.15 |

---

## 6. Step-Level Features (Currently Wired)

Per-candidate in `StepGenerator` / `StepAttempt`.

| Feature | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature used |
| `attempt_index` | int | Candidate index (0, 1, 2) |
| `n_tokens` | int | Token count in generated text |
| `entropy` | float | Character-level entropy estimate |
| `intermediate_value` | float | Extracted numeric answer from step text |
| `valid` | bool | Step verification result |
| `confidence` | float | Verification confidence |

---

## 7. Retry Engine Features (Currently Wired)

Logged per PAL retry attempt.

| Feature | Type | Description |
|---|---|---|
| `failure_mode` | str | Diagnosed failure (arithmetic_error, missing_answer, etc.) |
| `strategy` | str | Retry strategy applied |
| `modifier` | str | Prompt modifier used |
| `retry_count` | int | Number of retries executed |

---

## Summary Count

| Category | Wired | Research-Only | Total |
|---|---|---|---|
| Entropy raw | 6 | — | 6 |
| Entropy delta | 4 | — | 4 |
| Temporal | 17 | — | 17 |
| HS norm raw | 2 | — | 2 |
| HS norm delta | — | 7 | 7 |
| Verification | 6 | — | 6 |
| Scoring | 6 | — | 6 |
| Step-level | 7 | — | 7 |
| Retry | 4 | — | 4 |
| **Total** | **52** | **7** | **59** |

---

## Next Adds (Ranked by Effort/ROI)

1. ~~**Entropy delta features** — `max_positive_delta`, `delta_variance` (AUROC 0.718). Wired.~~
2. **L24 delta features** — `max_negative_delta`, `convergence_slope` (AUROC 0.65). Requires extending `LayerCaptureWrapper` to L24 + delta computation.
3. **L22/L18 delta features** — `mid_delta_sum` variants (AUROC 0.59-0.63). Requires multi-layer capture + full norm storage.
