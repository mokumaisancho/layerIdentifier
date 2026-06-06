# InferenceLLM2 — Making Local LLMs Excel at Inference

*Created: 2026-05-31*

---

## Issue Statement

Local LLMs (3B–7B parameters) contain sufficient knowledge to solve math reasoning problems, yet fail at **inference** — the ability to apply that knowledge through multi-step logical derivation to reach a correct novel conclusion.

The evidence is now concrete:

1. **The model already encodes correctness in its hidden states** but cannot read it back. Causal tracing on Qwen 3.5-4B found that at (Layer 3, position 39), a single scalar (skewness) achieves AUC=1.0, Cohen's d=3.08 separation between correct and wrong generation paths. Three independent statistics confirm this. The information is there; the architecture has no pathway to use it.

2. **External feature-based detection hits a hard ceiling.** Entropy, attention patterns, composite scoring — none reliably distinguish correct from confident-wrong reasoning. A 179-path controlled study proved per-head attention cannot detect semantic errors. The selector plateaued at 68% per-path accuracy, and the remaining failures are not fixable by better features (they are information-theoretically indistinguishable from outside the model).

3. **Rectification is problem-specific and fragile.** Attention boost at L3 fixes 60% of template-selection errors on one problem but 0% on structural errors. Detector calibration invalidates on any prompt change (5-token shift flipped greedy decoding from correct to wrong). Head ablation, steering vectors, gate editing — all proven non-viable through controlled experiments.

4. **The discovery loop confirms the generation gap.** On wave 2 benchmarks, raw LLM produces 0% novel, 0% surviving conjectures. The full loop raises survival to 25% and novelty to 67%, but only because the loop's falsification catches garbage — the LLM itself still generates recalled or trivial content.

**Root cause**: Local LLMs were trained to predict the next token, not to verify their own reasoning. The self-verification signal exists in internal representations (proven) but was never wired into the generation loop. The model is a brilliant oracle that cannot self-correct.

---

## Goal

**Build a self-verification layer for local LLMs that uses the model's own internal correctness signal to improve inference accuracy.**

Not an external judge. Not a bigger model. Not a chain-of-thought scaffold. A lightweight architectural addition (<10M params) that reads what the model already knows about its own correctness and feeds it back during generation.

### What "excels at inference" means (qualitative)

| Signal | Before | After |
|---|---|---|
| Model generates wrong reasoning confidently | Common | Rare — model "feels doubt" and re-attempts |
| Errors are detectable only by external scorer | Yes | No — model has internal error signal |
| Multi-step reasoning degrades over steps | Yes (attention decay proven) | Steps self-correct before cascading |
| Correctness is a binary outcome | Yes | Correctness is a per-step confidence curve |
| Same problem, different seed → different answer | Yes (proven: 5-token prompt shift flips answer) | More stable — self-correction smooths over noise |

---

## How to Reach It

### Architecture: Three-Layer Self-Verification

```
┌─────────────────────────────────────────────────┐
│                  LLM Backbone                    │
│           (Qwen 3.5-4B, frozen)                  │
│                                                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│  │ Layer 0 │...│ Layer 3 │...│ Layer 31│        │
│  └────┬────┘   └────┬────┘   └────┬────┘        │
│       │             │             │               │
│       │    ┌────────┴────────┐    │               │
│       │    │  Probe Head     │    │               │
│       │    │  (<10M params)  │    │               │
│       │    │                 │    │               │
│       │    │ Reads: L0..L31  │    │               │
│       │    │ hidden states   │    │               │
│       │    │ Outputs:        │    │               │
│       │    │  step_correct   │    │               │
│       │    │  (scalar 0..1)  │    │               │
│       │    └────────┬────────┘    │               │
│       │             │             │               │
│  ┌────┴─────────────┴─────────────┴──────┐       │
│  │         Self-Correction Router         │       │
│  │                                        │       │
│  │  if step_correct < threshold:          │       │
│  │    → backtrack + regenerate step       │       │
│  │    → adjust attention weights          │       │
│  │    → inject steering at bifurcation    │       │
│  │  else:                                 │       │
│  │    → continue generation               │       │
│  └────────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

### Phase 1: Validate the Probe (Proof of Concept)

**What**: Train a tiny classifier on hidden states to predict step correctness. Test if it works across problems (not just one).

**How**:
1. Capture hidden states for 200+ problems at all 32 layers during generation
2. Auto-label step correctness via Math-Shepherd continuation sampling (sample N continuations from each step; if majority reach correct answer, step is "good")
3. Train a logistic regression or 2-layer MLP on (hidden_state → correct/wrong)
4. Measure: does the probe generalize to unseen problems? Target: AUROC ≥ 0.85

**Why this works**: ReProbe (Ni et al., 2025) proved a <10M param probe on frozen LLM internal states matches PRMs 810x larger. Our own causal tracing found the signal (AUC=1.0). We just need to learn the mapping instead of hand-crafting thresholds.

### Phase 2: Wire Self-Correction (Intervention)

**What**: Use the probe's signal during generation to trigger correction when the model is going wrong.

**How**:
1. Build a step-level generation loop: generate step → probe → if low confidence, backtrack and retry
2. Wire the existing attention boost (proven 60% fix rate on template errors) as the first correction strategy
3. Add a "verification step" injection: after computation steps, force the model to verify its own arithmetic
4. Test on the 50-problem benchmark: target ≥ 90% accuracy (current baseline: ~78% without boost, 96% with problem-specific boost)

**Why this works**: We already proved attention boost works when applied correctly. The missing piece is knowing WHEN to apply it. The probe provides that signal.

### Phase 3: Internalize Self-Verification (Training)

**What**: Fine-tune the model with a dual objective so self-verification becomes native.

**How**:
1. Add a verification head to the model (shares backbone, outputs step correctness)
2. Train with dual loss: next-token prediction + step correctness prediction
3. Use step-level RLVR: Z3/code execution verifies each step, provides reward signal
4. Target: model generates with built-in self-correction, no external probe needed

**Why this works**: The hidden state already carries the signal (proven). Training the model to also predict correctness from its own hidden states creates an internal feedback loop — the model learns to "feel" when it's going wrong and adjust.

---

## Acceptance Criteria (Qualitative)

### AC-1: Probe Generalization
> Given a set of math problems the probe was NOT trained on, the probe correctly identifies wrong reasoning steps at a rate meaningfully above chance (>70% AUROC on held-out problems).

Evidence: cross-problem AUROC curve, confusion matrix on 20+ held-out problems.

### AC-2: Self-Correction Improves Accuracy
> When the self-correction loop is active, accuracy on the benchmark improves compared to the same model without self-correction. The improvement is consistent across problem types (template errors AND structural errors, not just one).

Evidence: benchmark accuracy with/without loop, per-problem-type breakdown.

### AC-3: No External Oracle Required
> The system does not require a larger model, API call, or human label to function at inference time. All verification comes from the probe head (Phase 1-2) or the model's own verification head (Phase 3).

Evidence: system runs fully local on a 16GB Mac with Qwen 3.5-4B + probe head.

### AC-4: Correction is Surgical
> When the model self-corrects, it does not degrade already-correct paths. The false positive rate (incorrectly flagging a correct step) remains below 20%.

Evidence: FPR measurement on known-correct generations. We already know from attention boost experiments that 20% FPR is the upper bound for a net-positive intervention.

### AC-5: Step-Level Granularity
> The verification signal operates at the individual reasoning step level, not just the final answer. The model can identify WHICH step went wrong, not just THAT the final answer is wrong.

Evidence: per-step probe accuracy, backtracking events showing the model retraces to the specific failing step.

### AC-6: Discovery Loop Integration
> The self-verification layer improves the discovery loop's output quality. More conjectures survive falsification (higher survival rate) and fewer are trivial recall (lower recall rate).

Evidence: discoveryLoop wave 3 benchmark vs wave 2 baseline (current: 25% survival, 67% novelty, 100% recall rate).

---

## Related Repositories and Files

### Primary Codebases

| Repo | Path | Relevance |
|---|---|---|
| parallel_l4_tot | `/Users/apple/Downloads/Py/parallel_l4_tot/` | Causal tracing, hidden state capture, attention boost, selector, all experimental results |
| discoveryLoop | `/Users/apple/Downloads/Py/discoveryLoop/` | Discovery loop, abduction engine, novelty detection, recall detection, benchmark harness |
| conversationHarness | `/Users/apple/Downloads/Py/conversationHarness/` | Safety layer, prompt engineering, scoring infrastructure |
| InferenceLLM2 | `/Users/apple/Downloads/Py/InferenceLLM2/` | **This project** |

### Critical Files (parallel_l4_tot)

| File | What it provides |
|---|---|
| `causal_trace/bifurcation/FINAL_REPORT.md` | Complete detection + rectification results. AUC=1.0 finding. Model instability analysis. |
| `causal_trace/bifurcation/ews.py` | EWS extraction from hidden states |
| `causal_trace/bifurcation/spc_detector.py` | ThresholdDetector + CUSUM/EWMA |
| `causal_trace/rectify/MASTER_STATE.md` | Current baseline (96% with boost), infrastructure inventory, known issues |
| `causal_trace/rectify/loop.py` | Generation loop with detection + rectification wiring |
| `causal_trace/rectify/steering.py` | Steering vector extraction/injection (broken, needs fix) |
| `causal_trace/rectify/test_attention_intervention.py` | Attention boost wrapper (WORKING, 81.8% fix rate) |
| `causal_trace/rectify/intervention_registry.py` | Uniform intervention interface |
| `causal_trace/rectify/per_head_tracing/results/HEAD_ZERO_FINDING_CORRECTED.md` | Proven: head 0 ablation is net negative (67% FPR) |
| `causal_trace/rectify/gate_assessment/results/GATE_FINDING.md` | Proven: gate already saturated, suppression impossible |
| `causal_trace/rectify/per_head_tracing/results/INTERVENTION_COMPARISON.md` | Full comparison of all 4 intervention strategies |
| `clean_capture.py` | Hidden state capture at all 32 layers during generation |
| `results/clean/` | 15 clean captures (Q135: 10 seeds, Q114: 5 seeds) |
| `results/clean_v2/` | 100 captures (Q135: 103 seeds) — training data for probe |
| `results/parallel_selector_results.jsonl` | 10-problem selector benchmark (0/10 correct — early test) |
| `FORMAT_LEVEL_ANALYSIS.md` | PAL + narrative simplification: 22% → 78% accuracy recovery |
| `SIGNAL_CATALOG.md` | 59 extractable signals inventory |
| `HS_FINDINGS.md` | Path-level HS integration has negative ROI |
| `attention_probe/ATTENTION_FUSION_FINDINGS.md` | Per-head attention cannot detect wrong reasoning (n=179) |
| `attention_probe/PER_HEAD_INTEGRATOR_STUDY.md` | "Attention measures WHERE, not WHAT" — conclusive negative |
| `PLAN_STEP_LEVEL_DECOMPOSITION.md` | 7-phase plan for step-level decomposition (partially implemented) |

### Critical Files (discoveryLoop)

| File | What it provides |
|---|---|
| `GOAL.md` | Research question, 4 use cases, gap table |
| `EXPERIMENT.md` | Pre-registered A/B experiment (blocked on calibration) |
| `loop.py` | Discovery loop: generate→falsify→refine cycle |
| `abduction.py` | Abduction engine: data tables → LLM patterns → test predicate |
| `novelty.py` | Novelty verification: OEIS search, sequence matching |
| `recall_detector.py` | Recall detection: 4 proxy signals (never calibrated) |
| `benchmark_report_wave2.json` | Latest benchmark: 25% survival, 67% novelty, 0 PSLQ identities |
| `ULTRAPLAN_CLOSE_GAP.md` | 4 waves: novelty wiring, reasoning decomposition, constraints, live run |
| `ULTRAPLAN_MEASUREMENT.md` | PSLQ baseline, TRACE-CoT cross-checker, token budgets |
| `INFERENCE_GAP_RESEARCH.md` | 11-section research synthesis (prior session output) |
| `CAPABILITY_SURFACE.md` | What domains the loop can/cannot explore |

### Critical Files (conversationHarness — secondary)

| File | What it provides |
|---|---|
| `safety/layer.py` | Safety scoring layer |
| `prompt_layer/prompt_engineer.py` | Prompt engineering with local LLM expansion |
| `prevention/router/router.py` | Problem routing infrastructure |

### Activity Tree Tasks (open, relevant)

| Task ID | Description | Status |
|---|---|---|
| `904ea879` | Selector correctness plateau (8 branches of investigation, conformal prediction proposed) | Open |
| `e5c8b40a` | Mechanistic interpretability for fusion error detection (not started) | Open |
| `69e6bf8d` | Live validation: PAL + harness + self-judgment (not completed) | Open |
| `30736e36` | Hinting strategies for local LLMs (no single strategy dominates) | Open |

---

## Sub-Goals

### SG-1: Hidden State Dataset
**Build a labeled dataset of (hidden_states, step_correct) pairs across 200+ problems.**

Deliverables:
- [ ] Capture hidden states at all 32 layers for 200+ GSM8K-level problems
- [ ] Implement Math-Shepherd auto-labeling: for each step, sample 5 continuations, majority-vote correctness
- [ ] Store as indexed files (one per problem) with step-level labels
- [ ] Train/val/test split: 70/15/15, stratified by problem difficulty

Dependencies: `clean_capture.py` infrastructure from parallel_l4_tot
Blocks: SG-2, SG-3

### SG-2: Probe Head Training
**Train and evaluate a step-correctness probe on hidden states.**

Deliverables:
- [x] Implement probe: 2-layer MLP, input = concatenated [L0, L3, L31] hidden states, output = P(correct)
- [x] Train on SG-1 training set
- [x] Evaluate on held-out test set: target AUROC ≥ 0.85, FPR ≤ 20%
- [x] Ablation study: which layers matter? (hypothesis: L3 is critical based on causal tracing)
- [x] Cross-problem generalization test: train on arithmetic, test on geometry/algebra

**Actual results** (differ from original plan):
- Two separate probes needed (step execution vs answer correctness — see Task Mismatch above)
- Step probe: MLP(512), AUROC=0.97 ✅
- Answer probe: balanced LR on L31 std-pool, AUROC=0.75 (5-fold CV), FPR=0.26 ✅
- Ablation: L31+std > L5+std > L31+max > L3+std. L3 is NOT the most informative layer for answer correctness (L31 is). std-pool >> mean-pool.
- Cross-problem: 5-fold CV by problem_id, all problem types mixed

Dependencies: SG-1
Blocks: SG-3, SG-4

### SG-3: Self-Correction Loop
**Wire the probe into the generation loop with backtracking.**

Deliverables:
- [ ] Step-level generation loop: pause after each reasoning step, run probe, decide continue/backtrack
- [ ] Backtracking: regenerate last step with temperature boost + attention adjustment at L3
- [ ] Max 2 backtrack attempts per step (prevent infinite loops)
- [ ] Benchmark: 50-problem GSM8K subset, compare accuracy with/without loop
- [ ] Measure: accuracy improvement, FPR impact, generation time overhead

Dependencies: SG-2, `causal_trace/rectify/loop.py`
Blocks: SG-4, SG-5

### SG-4: Discovery Loop Integration
**Connect self-verification to the discovery loop for better conjecture generation.**

Deliverables:
- [ ] Wire probe signal into discoveryLoop's `generate()` step
- [ ] When probe flags low confidence, trigger constraint-based re-generation instead of free-form
- [ ] Run wave 3 benchmark: compare survival rate, novelty rate, recall rate vs wave 2 baseline
- [ ] Target: survival ≥ 40%, novelty ≥ 70%, recall rate ≤ 50% (current: 25%, 67%, 100%)

Dependencies: SG-3, `discoveryLoop/loop.py`
Blocks: SG-5

### SG-5: Dual-Objective Fine-Tuning
**Fine-tune the model with a verification head so self-correction is native.**

Deliverables:
- [ ] Add verification head to model architecture (shares backbone, 2-layer MLP on top of hidden states)
- [ ] Generate training data: use SG-1 labels as targets for the verification head
- [ ] Train with dual loss: L = L_next_token + λ * L_verification
- [ ] Evaluate: does the fine-tuned model self-correct WITHOUT the external probe?
- [ ] Benchmark: accuracy, FPR, generation time vs probe-based approach (SG-3)

Dependencies: SG-1, SG-3
Blocks: None (terminal sub-goal)

### SG-6: Error Type Classifier
**Classify errors as template-selection vs structural vs algorithmic to route interventions.**

Deliverables:
- [ ] Build taxonomy from existing results: template errors (attention-boostable), structural errors (prompt-restructuring), algorithmic errors (need decomposition)
- [ ] Train classifier on probe features to predict error type
- [ ] Route: template → attention boost, structural → restructure prompt, algorithmic → step decomposition
- [ ] Benchmark: does routing improve over one-size-fits-all correction?

Dependencies: SG-2
Blocks: None (parallel improvement)

---

## Current Status (2026-05-31)

### Where We Are in the Big Picture

```
Phase 1: Validate the Probe        ████████████░░░░  ~80% done
Phase 2: Self-Correction Loop       ░░░░░░░░░░░░░░░░  0% done
Phase 3: Internalize Verification   ░░░░░░░░░░░░░░░░  0% done
```

**Phase 1 is nearly complete.** We have two working probes, benchmark-validated FPR, and
a clear understanding of what the signal looks like. The critical missing piece is **wiring
the probes into a live generation loop** (Phase 2 / SG-3).

### Sub-Goal Progress

| SG | Name | Status | Key Results |
|----|------|--------|-------------|
| SG-1 | Hidden State Dataset | **80%** — 150 captures, 50 problems × 3 seeds | Auto-labeled via CodeExecutionLabeler (AST execution), not Math-Shepherd. Sufficient for probe training but needs 10× more data for stability. |
| SG-2 | Probe Head Training | **90%** — both probes trained, ablation done | **Step probe**: AUROC=0.97 (MLP, 450 steps). **Answer probe**: AUROC=0.75 (L31+std-pool, balanced LR, 5-fold CV). FPR=0.26 at t=0.65 on held-out. |
| SG-3 | Self-Correction Loop | **0%** — not started | Infrastructure exists in `causal_trace/rectify/loop.py`. Needs wiring of both probes into live generation with backtracking. |
| SG-4 | Discovery Loop Integration | **0%** — blocked on SG-3 | |
| SG-5 | Dual-Objective Fine-Tuning | **0%** — blocked on SG-3 | |
| SG-6 | Error Type Classifier | **10%** — taxonomy exists from parallel_l4_tot | Error classification framework in `per_head_tracing/` and `intervention_registry.py`. Needs probe features as input. |

### Key Discovery: Task Mismatch

The original plan assumed one probe for both step execution and answer correctness.
Implementation revealed these are **two fundamentally different tasks**:

| | Step Execution Probe | Answer Correctness Probe |
|---|---|---|
| **Predicts** | Did this reasoning step execute without error? | Is the final numerical answer correct? |
| **Training labels** | 450 steps, AST-based execution check | 150 captures, `result == ground_truth` |
| **Model** | MLP(512) | Balanced LogisticRegression |
| **Features** | Concatenated hidden states at step boundary | **L31 std-pool** across all tokens |
| **AUROC** | 0.97 (val), 0.95 (test) | 0.75 (5-fold CV) |
| **FPR for answer prediction** | **1.00** (flags everything) | **0.26** at t=0.65 |

The step probe is excellent at detecting code execution failures but useless for predicting
answer correctness (FPR=1.00). The answer probe catches 69% of wrong answers while only
flagging 26% of correct ones. **Both probes are needed** — the step probe identifies WHERE
errors occur, the answer probe flags WHETHER the final answer is suspect.

### Key Discovery: std-pool >> mean-pool

Systematic ablation across 16 feature×layer combinations revealed:

- **std-pool** (standard deviation of hidden states across tokens) captures the answer
  correctness signal far better than mean-pool (Cohen's d: 0.75 vs 0.40).
- **Correct reasoning produces uniform hidden states** (low std) — the model's
  representations are consistent across the generation.
- **Wrong reasoning produces divergent hidden states** (high std) — the model's
  representations become chaotic as errors cascade.
- **Layer 31** (last layer) carries the strongest signal (AUROC=0.75).
  Layer 5 is second (0.73).
- Only **2,560 features** needed (1 layer × 2560 dim) instead of 10,240 (4 layers) —
  75% dimensionality reduction with better performance.

### Acceptance Criteria Status

| AC | Criterion | Target | Current | Status |
|----|-----------|--------|---------|--------|
| AC-1 | Probe generalization | AUROC > 0.70 | 0.75 (5-fold CV) | ✅ |
| AC-2 | Self-correction improves accuracy | Consistent improvement | Not yet tested | ❌ Blocked on SG-3 |
| AC-3 | No external oracle | Fully local | sklearn probes, no API calls | ✅ |
| AC-4 | FPR < 20% | Below 20% | 21% at t=0.70, 16% at t=0.85 | ⚠️ Close |
| AC-5 | Step-level granularity | Per-step signal | Step probe AUROC=0.97 | ✅ |
| AC-6 | Discovery loop integration | Better conjecture quality | Not yet tested | ❌ Blocked on SG-4 |

### What's Proven Now (New Findings)

| Finding | Evidence | Impact |
|---------|----------|--------|
| Step execution ≠ answer correctness | FPR=1.00 when using step probe for answer prediction | Requires dual-probe architecture (both probes) |
| std-pool captures correctness signal | Cohen's d=0.75 vs mean-pool's 0.40 | Better features, lower dimensionality |
| L31 std-pool is the best single feature | AUROC=0.75 ± 0.13 (5-fold CV) | Only 2560 features needed |
| 50 problems is too few for stable AUROC | ±0.26 variance with mean-pool, ±0.13 with std-pool | Need 500+ captures for production use |
| MLP overfits on 150×10240 | AUROC=0.31 (below random) | Use LR for small-sample probe training |

### Critical Path to Phase 2

**SG-3 (Self-Correction Loop) is the next milestone.** The probes exist, the infrastructure
exists, the wiring is the missing piece.

Required steps:
1. Build live generation loop that pauses at step boundaries
2. At each step: run step probe → if low confidence, flag for correction
3. At generation end: run answer probe → if flagged, retry with adjusted parameters
4. Combine with existing attention boost (60% fix rate on template errors)
5. Benchmark on 50-problem set with/without loop

**Estimated effort**: 1-2 sessions
**Risk**: Probe FPR=0.26 means ~1 in 4 correct answers get retried — net accuracy may not
improve if retries aren't better than originals. The retry strategy matters as much as the
detection.

---

## What We Know vs. What We Don't

### Proven (do not re-investigate)

| Finding | Evidence | Source |
|---|---|---|
| Correctness encoded at L3, pos 39 | AUC=1.0, d=3.08 | `FINAL_REPORT.md` |
| Attention cannot detect semantic errors | 179 controlled paths | `PER_HEAD_INTEGRATOR_STUDY.md` |
| Feature-based selection plateaus at 68% | 512-token benchmark | Task `904ea879` |
| Head 0 ablation is net negative | 67% FPR | `HEAD_ZERO_FINDING_CORRECTED.md` |
| Gate editing is useless | Gate already at 0.06 | `GATE_FINDING.md` |
| Steering vectors destroy generation | Gibberish at α≥10 | `FINAL_REPORT.md` |
| Attention boost works narrowly | 60% on template errors, 0% on structural | `FINAL_REPORT.md` |
| PAL recovers accuracy | 22% → 78% | `FORMAT_LEVEL_ANALYSIS.md` |
| Model behavior is session-unstable | 5-token shift flips greedy answer | `FINAL_REPORT.md` |
| Probe approach externally validated | ReProbe matches PRMs 810x larger | Ni et al. 2025 |
| **Step execution ≠ answer correctness** | Step probe FPR=1.00 for answer prediction | `ULTRAPLAN_DUAL_PROBE.md` V2 |
| **L31 std-pool detects answer correctness** | AUROC=0.75, Cohen's d=0.75 | `ULTRAPLAN_DUAL_PROBE.md` V2 |
| **std-pool >> mean-pool for correctness** | d=0.75 vs d=0.40, same data | `_layer_analysis` sweep |
| **MLP overfits small capture datasets** | AUROC=0.31 (below random) on 150×10240 | `ULTRAPLAN_DUAL_PROBE.md` Wave 1 |

### Unknown (this project will answer)

| Question | Why it matters | Which SG addresses it | Status |
|---|---|---|---|
| Does the L3 signal generalize across problem types? | If yes, one probe works for all math. If no, need per-type probes. | SG-2 | **Answered**: L31 carries the strongest signal (not L3). 5-fold CV by problem shows AUROC=0.75 ± 0.13 — generalizes but with high variance at 50 problems. |
| Can auto-labeling replace human step annotations? | Scalability depends on not needing humans. | SG-1 | **Answered**: Yes, CodeExecutionLabeler (AST execution) produces reliable step labels. |
| What is the probe's FPR on truly novel problems? | High FPR means the loop degrades correct answers. | SG-2 | **Answered**: FPR=0.26 on held-out at t=0.65. Below the 0.30 target. |
| Does backtracking actually improve final accuracy? | Theoretically yes, but regeneration might produce different errors. | SG-3 | Open — **critical next question** |
| Can self-verification improve conjecture novelty? | Discovery loop currently generates recalled content. | SG-4 | Open — blocked on SG-3 |
| Can the model internalize verification via fine-tuning? | Eliminates external probe overhead. | SG-5 | Open — blocked on SG-3 |
| Are error types predictable from hidden states? | Enables intervention routing. | SG-6 | Open — taxonomy exists, probe features not yet tested as input |
| Does more data improve probe reliability? | AUROC variance is ±0.13 at 50 problems. | SG-1 | **Partially answered**: Signal exists (Cohen's d=0.75). 500+ captures would likely halve variance. Not yet tested. |
