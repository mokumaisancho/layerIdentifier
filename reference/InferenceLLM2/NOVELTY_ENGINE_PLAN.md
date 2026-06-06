# Novelty Discovery Reasoning Engine — Implementation Plan

**Created**: 2026-06-02
**Status**: DRAFT — awaiting approval
**Estimated total**: 7.5 hours (4 waves + revised hypothesis test gate)

---

## Issue Statement

**Problem**: The LLM (qwen2.5:3b via Ollama) produces 0% novel mathematical conjectures — all output is recalled from training data (proven: 69/69 rounds in wave 2 benchmark). Abduction (data-driven generation) produces 100% novel (9/9). The engine must detect recall during generation and force constraint-based (abduction) mode.

**Why this matters**: Without novelty detection, the discovery loop generates plausible but unoriginal output. The engine's purpose is NOT to solve math problems (that's QA) — it's to ensure the reasoning pipeline produces genuinely novel conjectures that survive falsification.

**Current state**:
- `discoveryLoop/loop.py`: generate→falsify→refine loop, but 0% novel from LLM
- `discoveryLoop/abduction.py`: data-driven generation, 100% novel, but never triggered by recall detection
- `discoveryLoop/recall_detector.py`: heuristic regex matching (well_known_topic, predicate_triviality) — flags 100% as recall (not discriminative)
- `InferenceLLM2/`: capture infrastructure (MLX), probes (step AUROC=0.97, answer AUROC=0.75), all trained on math problems
- **Gap**: No hidden-state-based novelty detection. No bridge between MLX capture and discoveryLoop generation.

---

## Hypothesis

Hidden states during RECALLED generation differ from NOVEL generation in measurable ways.

**Evidence for**:
- We already proved hidden states carry a correct/wrong signal (Cohen's d=0.75, L31 std-pool)
- Correct reasoning = uniform hidden states (low std). Recalled output should be even MORE uniform (high-confidence, well-worn pathways)
- Novel reasoning = cross-domain connections, uncertainty → hidden states should show higher variance at "bridge points"
- Free-form LLM (all recall) vs abduction (all novel) gives us labeled data with ground truth

**Evidence against**:
- The model may be equally confident recalling and genuinely inferring — internal representations may not distinguish "I know this" from "I figured this out"
- 69 recalled + 9 novel = only 78 samples (small for probe training)
- No prior art on recall-vs-novel detection from hidden states (papers exist for memorization detection from OUTPUT, not from internal activations)

**Killing test** (Wave 0, 1.8h): Capture hidden states during 25 recalled + 25 novel generations. Multi-metric test (Cohen's d + permutation p-value + LOO-AUROC) with Monte Carlo verified thresholds. If NO_GO → pivot to always-abduction.

---

## Expected Outcome

| Metric | Before Engine | After Engine |
|--------|--------------|-------------|
| Novel conjecture rate | 0% (LLM alone) | ≥ 30% (probe gates recall → abduction) |
| Survival rate | 25% (discoveryLoop wave 2) | ≥ 35% (higher quality input to falsification) |
| False recall rate (flag novel as recall) | N/A | ≤ 20% |
| Generation time overhead | 0s | ≤ 5s per conjecture (probe inference only) |

**If hypothesis fails (Wave 0 kills)**: Pivot to "always-abduction" mode — skip probe entirely, force abduction on every round. This still produces 100% novel output but without the adaptive gating.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISCOVERY LOOP                                │
│                                                                  │
│  ┌──────────────┐     ┌───────────────────┐                     │
│  │  Data Tables  │────▶│  Abduction Engine  │──▶ Novel Conjecture │
│  └──────────────┘     │  (constraint-based)│    (100% novel)     │
│       ↑               └───────────────────┘                     │
│       │                        ↑                                  │
│       │                  ┌─────┴──────┐                          │
│       │                  │ Recall Gate │                          │
│       │                  │             │                          │
│       │                  │ if recall:  │                          │
│       │                  │   force     │                          │
│       │                  │   abduction │                          │
│       │                  │ if novel:   │                          │
│       │                  │   continue  │                          │
│       │                  └─────┬──────┘                          │
│       │                        ↑                                  │
│  ┌────┴─────────────────────────┴────────────────────┐          │
│  │           MLX Generation Client                    │          │
│  │                                                     │          │
│  │  1. Generate conjecture via MLX (replaces Ollama)  │          │
│  │  2. Capture hidden states at L0, L5, L31           │          │
│  │  3. Extract features (std-pool, mean-pool)         │          │
│  │  4. Run NoveltyProbe → recall_score 0..1           │          │
│  │  5. Return conjecture + recall_score               │          │
│  └────────────────────────────────────────────────────┘          │
│       ↑                                                          │
│  ┌────┴──────────┐                                              │
│  │ NoveltyProbe   │  LR or MLP on L31 std-pool features         │
│  │ (trained)      │  Input: 2560-dim L31 hidden state stats     │
│  │                │  Output: P(recall)  0..1                     │
│  └────────────────┘                                              │
│                                                                  │
│  Downstream: Falsification (CCE) → survive/falsify → output      │
└─────────────────────────────────────────────────────────────────┘
```

**Key architectural change**: Replace Ollama (`LLMClient`) with MLX-backed generation client that captures hidden states during conjecture generation. This is the bridge between InferenceLLM2's capture infrastructure and discoveryLoop's generation pipeline.

---

## Dependency Graph

```
Wave 0: Hypothesis Test (1.8h, n=25+25)
    │
    ├── SIGNAL EXISTS (GO) ──▶ Wave 1 + Wave 2 (parallel) ──▶ Wave 3 ──▶ Wave 4
    │                              (2h each)                    (2h)        (1h)
    │
    ├── AMBIGUOUS ──▶ Collect 10 more captures (22 min) ──▶ re-test
    │
    └── NO SIGNAL (NO_GO) ──▶ PIVOT: Always-Abduction Mode (1h)
                                 (skip probe, force abduction every round)
```

No circular dependencies. Wave 0 is the gate. Waves 1+2 can run partially in parallel. Wave 3 depends on both. Wave 4 depends on Wave 3.

---

## Waves / Tasks

### Wave 0: Hypothesis Test — "Does the signal exist?"
**ETA**: 1.8 hours (50 captures at ~130s each)
**Agent**: explorer (read-only, analysis)
**Dependencies**: None (can start immediately)

**Statistical Power Note**: The original plan (n=10+10, d ≥ 0.2) had 64% false negative rate at d=0.75. This revised plan uses n=25+25 with a multi-metric decision framework that achieves < 3% false negative rate. See `scripts/wave0_statistical_framework.py` for the implementation.

**What**:
1. Load Qwen 3.5-4B via MLX
2. Generate 25 free-form conjectures (recall condition) → capture L0, L5, L31 hidden states
3. Generate 25 abduction-based conjectures (novel condition) → capture same layers
4. Extract aggregate features per capture: mean, std, skew, kurtosis per layer (12 features total)
5. Run multi-metric decision framework:
   - **Tier 1**: Cohen's d + permutation p-value + bootstrap CI for each feature
     - GO if: (d ≥ 0.5 AND p < 0.10) OR d ≥ 0.8
     - NO_GO if: d < 0.2 AND p > 0.30
     - AMBIGUOUS otherwise
   - **Tier 2** (if not NO_GO): LOO-AUROC with regularized LR (C=0.01, balanced) + nearest centroid baseline
     - GO if: LOO-AUROC ≥ 0.70, or LOO-AUROC ≥ 0.60 AND NC-AUROC ≥ 0.60
     - NO_GO if: LOO-AUROC < 0.55 AND NC-AUROC < 0.55
6. Final decision → GO (build engine) / NO_GO (pivot to always-abduction) / AMBIGUOUS (collect 10 more)

**Monte Carlo verified decision rates at n=25+25**:
| True d | GO rate | AMBIGUOUS | NO_GO |
|--------|---------|-----------|-------|
| 0.0    | 4%      | 34%       | 62%   |
| 0.5    | 49%     | 36%       | 15%   |
| 0.75   | 80%     | 17%       | 3%    |
| 1.0    | 96%     | 3%        | 0.3%  |

**Acceptance Criteria**:
- AC0.1: 50 capture files produced (25 recall, 25 novel), each with layer_0/5/31 tensors
- AC0.2: Cohen's d + permutation p-value + bootstrap CI computed for ≥3 feature/layer combinations
- AC0.3: LOO-AUROC computed with nested C selection (C ∈ {0.001, 0.01, 0.1, 1.0})
- AC0.4: GO/NO-GO/AMBIGUOUS decision recorded with full statistical evidence
- AC0.5: Peak memory < 14GB (alert if exceeding, kill if > 15GB)
- AC0.6: Crash-safe log written after each capture (survives SIGTERM)

**Files to create**:
- `InferenceLLM2/scripts/wave0_hypothesis_test.py` — standalone script, self-contained
- `InferenceLLM2/captures/wave0/` — output directory

**Reuse**: `InferenceLLM2/src/inference_llm/capture/mlx_capture.py` (MLXCaptureAdapter), `InferenceLLM2/src/inference_llm/capture/capture.py` (save_capture), `InferenceLLM2/scripts/wave0_statistical_framework.py` (decision logic)

---

### Wave 1: MLX Generation Client — Bridge to discoveryLoop
**ETA**: 2 hours
**Agent**: engineer-1 (senior-python-engineer)
**Dependencies**: Wave 0 GO decision
**Blocked by**: None (can start placeholder before Wave 0 completes)

**What**:
1. Build `MLXConjectureClient` that replaces `LLMClient` (Ollama) in discoveryLoop
2. Reuses `MLXCaptureAdapter` from InferenceLLM2 for generation + capture
3. Implements same interface as `LLMClient`: `generate_conjecture()`, `refine_conjecture()`
4. Returns `ConjectureResponse` with additional `hidden_states` and `recall_score` fields
5. Handles prompt construction for conjecture generation (adapt from `llm.py:generate_conjecture` prompt)
6. Memory management: gc.collect() after each generation, model stays loaded

**Acceptance Criteria**:
- AC1.1: `MLXConjectureClient.generate_conjecture("number_theory")` returns valid `ConjectureResponse` with non-empty statement + predicate
- AC1.2: Hidden states captured for ≥3 layers (L0, L5, L31) per generation
- AC1.3: Same prompt format as Ollama client → comparable output quality
- AC1.4: Generation time ≤ 180s per conjecture (baseline: ~130s from KNOWN_LIMITATIONS L-4)
- AC1.5: 5 test cases pass (mock-free): generates valid predicate, handles LaTeX, rejects unsafe code, returns reasoning trace, captures hidden states
- AC1.6: No memory leak across 5 consecutive generations (RSS delta < 100MB)

**Files to create**:
- `discoveryLoop/discovery_loop/mlx_client.py` — MLX-backed conjecture client
- `discoveryLoop/tests/test_mlx_client.py` — 5 test cases

**Reuse**: `InferenceLLM2/src/inference_llm/capture/mlx_capture.py`, `discoveryLoop/discovery_loop/llm.py` (prompt format, ConjectureResponse, _extract_from_response, _validate_predicate, _sanitize_predicate)

**Contradiction check**: MLX model (Qwen 3.5-4B) ≠ Ollama model (qwen2.5:3b). Output quality may differ. This is ACCEPTABLE — we're upgrading the model. But must verify output format compatibility (REASONING/STATEMENT/PREDICATE headers).

---

### Wave 2: Novelty Probe Training
**ETA**: 1.5 hours
**Agent**: engineer-2 (senior-python-engineer)
**Dependencies**: Wave 0 GO decision + Wave 0 capture data (50 files)
**Can start**: Immediately after Wave 0 produces capture files

**What**:
1. Load 50 Wave 0 captures (25 recall, 25 novel)
2. Extract aggregate features per capture: mean, std, skew, kurtosis per layer (12 features)
3. Train binary classifier: LogisticRegression (balanced, C=0.01) — strong regularization for small n
4. Evaluate: leave-one-out cross-validation with nested C selection
5. Compare with nearest centroid baseline
6. Save probe to `discoveryLoop/probes/novelty_probe_v1.joblib`
7. Log: AUROC, accuracy, FPR, FNR, per-sample predictions

**Acceptance Criteria**:
- AC2.1: Probe trained on 50 samples with balanced class weights and C=0.01
- AC2.2: Leave-one-out AUROC >= 0.65 (below this = not useful for gating)
- AC2.3: FPR <= 0.30 (don't want to force abduction on genuinely novel output)
- AC2.4: Probe saved as joblib, loads in < 1s
- AC2.5: Feature importance logged (which layer/feature carries the signal)
- AC2.6: If AUROC < 0.55 (below random) -> KILL, log as NO SIGNAL, pivot

**Files to create**:
- `discoveryLoop/discovery_loop/novelty_probe.py` — probe training + inference
- `discoveryLoop/probes/` — output directory
- `discoveryLoop/tests/test_novelty_probe.py` — 3 test cases (train, predict, save/load)

**Reuse**: `InferenceLLM2/src/inference_llm/probe/training.py` (CaptureProbeModel pattern, LogisticRegression balanced), `InferenceLLM2/src/inference_llm/capture/capture.py` (load_capture)

**Contradiction check**: 50 samples is small for probe training but workable. The answer probe needed 150 samples for AUROC=0.75. But:
- We use aggregated features (12 dims, not 2560 raw) -> p < n, no overfitting
- LR with C=0.01 (strong regularization) handles small-n well
- Leave-one-out CV with nested C selection is the correct evaluation
- Nearest centroid baseline provides a no-overfitting comparison
- If Wave 0 shows d >= 0.5, 50 samples produces a usable probe (80% GO rate verified by Monte Carlo)

---

### Wave 3: Recall Gate Wiring
**ETA**: 2 hours
**Agent**: engineer-1 (after Wave 1) + engineer-2 (after Wave 2)
**Dependencies**: Wave 1 (MLX client) + Wave 2 (novelty probe)
**Blocked by**: Both Wave 1 and Wave 2 must complete

**What**:
1. Modify `DiscoveryLoop._next_conjecture()` to use `MLXConjectureClient` instead of `LLMClient`
2. Add recall gate: after generation, run `NoveltyProbe.predict(hidden_states)` → if recall_score > threshold, switch to abduction mode for this round
3. Wire abduction as the fallback: if recall detected → `run_abduction()` instead of free-form generation
4. Log every decision: round, recall_score, gate_decision (pass/force_abduction), final outcome
5. Crash-safe JSONL logging at every round boundary (fsync)

**Acceptance Criteria**:
- AC3.1: `DiscoveryLoop` runs with MLX client + recall gate without errors
- AC3.2: When recall_score > threshold → abduction is triggered (not free-form generation)
- AC3.3: When recall_score ≤ threshold → free-form generation proceeds normally
- AC3.4: Every round logged to JSONL with: round_idx, recall_score, gate_decision, conjecture, status
- AC3.5: Loop survives SIGTERM (state resumes from last logged round)
- AC3.6: No race conditions: probe inference is synchronous, single-threaded
- AC3.7: Memory bloat prevention: gc.collect() after each round, hidden states released before next round
- AC3.8: 3 test cases pass: gate triggers abduction on high recall score, gate allows normal on low score, state persists after mock crash

**Files to modify**:
- `discoveryLoop/discovery_loop/loop.py` — add recall gate in `_next_conjecture()`
- `discoveryLoop/tests/test_recall_gate.py` — 3 test cases

**Files to create**:
- `discoveryLoop/discovery_loop/recall_gate.py` — gate logic (probe + threshold + decision)

**Reuse**: `discoveryLoop/discovery_loop/abduction.py` (run_abduction), `discoveryLoop/discovery_loop/loop.py` (_safe_write_jsonl for crash safety)

**Contradiction check**: The loop currently supports two modes (LLM + rule-based fallback). Adding a third mode (abduction) must not break the existing refinement flow. Abduction results must be `ConjectureCandidate` objects compatible with falsification.

---

### Wave 4: Validation Benchmark
**ETA**: 1 hour
**Agent**: engineer-1
**Dependencies**: Wave 3

**What**:
1. Run discoveryLoop for 20 rounds with engine enabled (recall gate + novelty probe)
2. Run discoveryLoop for 20 rounds WITHOUT engine (baseline, same model via MLX)
3. Compare: novel_rate, survival_rate, recall_rate per condition
4. Log all conjectures with their recall_scores and gate decisions
5. Produce benchmark report JSON

**Acceptance Criteria**:
- AC4.1: 20 rounds complete for each condition without crash
- AC4.2: Engine condition novel_rate > 0% (baseline is 0%)
- AC4.3: Engine condition novel_rate ≥ 30% (target)
- AC4.4: False recall rate ≤ 20% (novel conjectures wrongly flagged as recall)
- AC4.5: Total runtime ≤ 90 min per condition (20 rounds × ~130s × 2 conditions ÷ 60)
- AC4.6: Benchmark report JSON contains: per-round recall_score, gate_decision, novelty verdict, survival status
- AC4.7: Watchdog timeout: per-round max 300s, script alarm at 5400s (90 min)

**Files to create**:
- `discoveryLoop/scripts/benchmark_novelty_engine.py`
- `discoveryLoop/benchmarks/novelty_engine_report.json`

**Reuse**: `discoveryLoop/discovery_loop/measure.py` (run_comparison), `discoveryLoop/benchmarks/` (report format)

---

## PIVOT: Always-Abduction Mode (if Wave 0 kills)

If Cohen's d < 0.2 (no signal in hidden states), the probe approach fails. Pivot:

**What**: Skip probe entirely. Force abduction on every round of the discovery loop.

**ETA**: 1 hour
**Agent**: engineer-1

**Acceptance Criteria**:
- AC-P.1: DiscoveryLoop runs 20 rounds in abduction-only mode
- AC-P.2: Novel_rate = 100% (abduction proven to produce 100% novel)
- AC-P.3: Survival_rate ≥ 20% (abduction wave 2 was 25%)

**This is still a valid engine output**: guaranteed novel conjectures via constraint-based generation, even without the adaptive gating.

---

## Agent Assignments

| Agent | Role | Waves | ETA |
|-------|------|-------|-----|
| **explorer** | Read-only analysis, hypothesis test | Wave 0 | 1.8h |
| **engineer-1** | MLX client, wiring, validation | Waves 1, 3, 4 | 5h |
| **engineer-2** | Probe training, feature engineering | Wave 2 | 1.5h |

**Parallelization**:
- Wave 0: explorer runs alone (1.8 hours)
- Waves 1 + 2: engineer-1 and engineer-2 run IN PARALLEL after Wave 0 GO (saves 1.5h)
- Wave 3: both engineers collaborate (engineer-1 wires, engineer-2 validates probe in loop)
- Wave 4: engineer-1 runs benchmark

**Total wall time with parallelization**: 1.8h (Wave 0) + 2h (Waves 1+2 parallel) + 2h (Wave 3) + 1h (Wave 4) = **6.8 hours**
**Total wall time if sequential**: 8.3 hours

---

## Contradiction Assessment

| Potential Contradiction | Resolution |
|------------------------|------------|
| MLX model (Qwen 3.5-4B) ≠ Ollama model (qwen2.5:3b) | ACCEPTABLE: upgrading the model. Verify output format compatibility in Wave 1 AC1.3 |
| 20 samples may be too few for probe | ADDRESSED: Wave 2 AC2.6 has kill threshold. Can re-run Wave 0 with more captures if d is large but probe fails |
| Capture during conjecture generation ≠ capture during math solving | ADDRESSED: The capture infrastructure is generic (layer wrappers don't care about prompt content). Wave 0 validates this |
| Recall gate may slow the loop | ADDRESSED: Probe inference is sklearn LR, < 1ms. Capture adds ~5s per generation (MLX overhead, not probe overhead) |
| Abduction may produce lower survival than free-form | ADDRESSED: Wave 2 benchmark showed 25% survival with abduction. Free-form had 0% novel but potentially higher survival on recalled conjectures. The engine trades survival for novelty — correct tradeoff |
| Memory bloat from hidden state storage | ADDRESSED: AC1.6, AC3.7, AC0.4 all enforce memory limits. gc.collect() after every round. Hidden states released before next round |

---

## Existing Repos / Tools (No Reinvention)

| Tool/Repo | What it provides | How we use it |
|-----------|-----------------|---------------|
| **InferenceLLM2 capture** (`mlx_capture.py`, `capture.py`) | MLX hidden state capture during generation | Direct reuse — Wave 0 and Wave 1 |
| **InferenceLLM2 probe training** (`training.py`) | ProbeModel, CaptureProbeModel, feature extraction | Reuse pattern (LR balanced) for Wave 2 |
| **discoveryLoop loop.py** | Generate→falsify→refine loop with crash-safe JSONL | Modify in Wave 3 (add recall gate) |
| **discoveryLoop abduction.py** | Data-driven conjecture generation (100% novel) | Reuse as forced-fallback in Wave 3 |
| **discoveryLoop llm.py** | ConjectureResponse, prompt format, predicate validation | Reuse output types and prompt structure in Wave 1 |
| **discoveryLoop falsifier.py** | CCE-based counterexample search | No changes needed |
| **TransformerLens** (neelnanda-io) | Activation probing patterns for PyTorch models | Reference only — concepts apply, code is PyTorch-only |
| **ReProbe** (Ni et al. 2025) | Probe on frozen LLM states matches PRMs 810x larger | Validates our approach (external probe on frozen model) |
| **sklearn LogisticRegression** | Balanced LR for small-sample classification | Direct use for novelty probe (Wave 2) |
| **TxGraffiti** | Automated conjecture generation from graphs | Supplementary only (assessed as buggy, v0.4.1) |

---

## Process Safeguards

| Concern | Mitigation |
|---------|-----------|
| Crash-safe logs | `_safe_write_jsonl` with fsync after every round (already exists in loop.py) |
| Memory bloat | gc.collect() after each round, RSS monitoring, kill at 15GB |
| Race conditions | Single-threaded: probe inference is synchronous, no parallel generation |
| Context overflow | Each wave is a separate agent with scoped context. No agent reads > 5 files |
| Session death | Wave completion state written to `InferenceLLM2/wave_status.json` after each wave |
| Over-engineering | Kill criteria at Wave 0 and Wave 2. Pivot path defined. Maximum 5.5h total |
| Watchdog timeouts | Per-round timeout 300s, per-wave timeout 3600s |

---

## MVP Definition

**MVP = Wave 0 (1.8 hours)**

If the signal doesn't exist, nothing else matters. The MVP is the hypothesis test.

**MVP success -> full engine (6.8h total, 5h after Wave 0)**
**MVP failure -> always-abduction mode (1h after Wave 0)**

Either way, the discovery loop produces novel conjectures within 7.5 hours maximum.

---

## Execution Order (Explicit)

```
T+0:00  ──▶ Start Wave 0 (explorer agent, 1.8h)
              ↓
T+1:48  ──▶ Check Wave 0 result
              │
              ├── GO ──▶ Start Wave 1 (engineer-1) + Wave 2 (engineer-2) IN PARALLEL
              │                ↓
              │          T+3:48 ──▶ Start Wave 3 (engineer-1 + engineer-2)
              │                         ↓
              │                   T+5:48 ──▶ Start Wave 4 (engineer-1)
              │                                ↓
              │                          T+6:48 ──▶ DONE. Report.
              │
              ├── AMBIGUOUS ──▶ Collect 10 more captures (22 min) ──▶ re-test
              │
              └── NO-GO ──▶ Start Pivot (engineer-1, 1h)
                                ↓
                          T+2:48 ──▶ DONE. Report.
```
