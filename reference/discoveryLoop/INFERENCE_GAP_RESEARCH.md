# The Inference Gap: Why Local LLMs Fail at Reasoning and What Would Fix It

**Date:** 2026-05-31
**Sources:** parallel_l4_tot (mechanistic research), discoveryLoop (inference measurement), external papers (PRM, interpretability, RLVR)
**Question:** A local 4B LLM has more mathematical knowledge than any human in history. Why can't it reason? What's missing at the ML level?

---

## 1. The Diagnosis (What the Two Repos Proved)

### What parallel_l4_tot established through direct experimentation:

**Finding 1: Correctness IS encoded in hidden states — but the model doesn't use it.**

Causal tracing (Qwen3.5-4B on GSM8K) found that at (Layer 3, position 39), a single scalar (skewness of the hidden state) perfectly separates correct from wrong paths (AUC=1.0, Cohen's d=3.08). The information exists inside the model. The model just doesn't read it back to self-correct.

*Source: `parallel_l4_tot/causal_trace/bifurcation/FINAL_REPORT.md`*

**Finding 2: Attention measures WHERE the model looks, not WHAT it computes.**

Per-head attention study (179 paths, controlled experiments) proved that correct and wrong paths have near-identical attention patterns when format is controlled. A wrong formula (`*` instead of `/`) and a correct formula attend to the same tokens. Attention cannot detect semantic errors because the error is in token identity, not attention distribution.

*Source: `parallel_l4_tot/attention_probe/PER_HEAD_INTEGRATOR_STUDY.md`*

**Finding 3: The model doesn't know it's wrong — it works slightly harder on wrong paths.**

In controlled PAL experiments (same problem, same temperature, different seeds), wrong paths showed *higher* cross-step attention than correct paths (31/32 heads, +0.006 mean). The model expends more effort on errors without detecting them.

*Source: `parallel_l4_tot/attention_probe/PER_HEAD_INTEGRATOR_STUDY.md` §4.5*

**Finding 4: Errors happen at template selection, not computation.**

The divergence point (L3, pos 39) is where the model commits to a solution approach — not where it computes. Once the wrong template is selected, all downstream computation is wasted. The model is like a student who writes down the wrong formula first, then computes flawlessly from it.

*Source: `parallel_l4_tot/causal_trace/bifurcation/FINAL_REPORT.md` §Part 1*

**Finding 5: 36% of failures are surface narrative, 45% are structural, only 18% are algebraic.**

On GSM8K, narrative stripping recovers 36% of failures. Restructuring into explicit relationships recovers another 45%. The hard floor (18%) is where the model genuinely misunderstands the math — "twice as many X as Y" read as "Y = 2×X" (wrong direction).

*Source: `parallel_l4_tot/FORMAT_LEVEL_ANALYSIS.md`*

**Finding 6: The calibration is fragile — session-specific, invalidated by 5-token prompt changes.**

The perfect hidden-state detector breaks when the prompt changes by even 5 tokens. The model's internal representation shifts enough to invalidate the threshold. Detection works; deployment doesn't.

*Source: `parallel_l4_tot/causal_trace/bifurcation/FINAL_REPORT.md` §Part 3*

### What discoveryLoop established through design and analysis:

**Finding 7: Current metrics measure survival, not reasoning quality.**

The core loop measures "did the predicate survive n=1..100,000?" — that's falsification, not inference. A recalled theorem (Euler's polynomial) and a genuinely constructed insight both "survive." The system can't tell them apart.

*Source: `discoveryLoop/GOAL.md`*

**Finding 8: The recall-vs-construction distinction is measurable but not measured.**

The repo has four proxy signals (well_known_topic, predicate_triviality, novel_combination, statement_specificity) and OEIS lookup. But they've never been calibrated against ground truth. The experiment protocol exists but has no data.

*Source: `discoveryLoop/EXPERIMENT.md`, `discoveryLoop/ULTRAPLAN_CLOSE_GAP.md`*

**Finding 9: PAL (Program-Assisted Language) recovers +56% over baseline.**

Forcing the model to generate Python code instead of free-text reasoning, then executing it, recovered accuracy from 22% to 78% on GSM8K. This is the single most effective intervention — it works because execution bypasses the model's inability to verify its own reasoning.

*Source: `parallel_l4_tot/FORMAT_LEVEL_ANALYSIS.md` §ROI Summary*

---

## 2. The Unified Diagnosis

These findings converge on a single picture:

```
Current LLM Architecture:

  Knowledge (recall)  ──────────────►  Works well
       ↓
  Template selection  ──────────────►  Knife-edge, fragile
       ↓
  Step-by-step computation  ───────►  Each step looks right
       ↓
  Self-verification  ──────────────►  MISSING ENTIRELY
       ↓
  Error detection  ────────────────►  Information exists but unused
       ↓
  Self-correction  ────────────────►  NOT POSSIBLE without detection
```

The model is a knowledge retrieval system with a reasoning-shaped hole. It can retrieve the right approach and compute individual steps. What it cannot do is:

1. **Detect when a step is wrong** (Finding 2: attention can't do it; Finding 1: hidden states can but aren't used)
2. **Detect when it selected the wrong template** (Finding 4: the error is early and invisible to downstream computation)
3. **Distinguish recall from construction** (Finding 7: it measures survival, not reasoning quality)
4. **Calibrate its own confidence** (Finding 6: calibration is session-specific and fragile)

---

## 3. What Has Been Tried (And Its Limits)

| Approach | What it does | Why it's limited | Source |
|---|---|---|---|
| **PAL (code execution)** | Forces model to write executable code | Only works for computable problems; doesn't help with template selection | parallel_l4_tot |
| **Parallel paths + selection** | Generate N solutions, pick best | Selector is only as good as its features; entropy-based selection is weak | parallel_l4_tot |
| **Attention boosting** | Amplify correct-template signal at L3 during prefill | 60% success on one problem type; non-monotonic; problem-specific | parallel_l4_tot causal trace |
| **Hidden state detection** | Threshold on skewness at divergence point | Perfect offline but session-specific; invalidated by prompt changes | parallel_l4_tot causal trace |
| **Narrative stripping** | Remove surface noise before reasoning | Recovers 36% of failures; doesn't touch the 18% algebraic floor | parallel_l4_tot format analysis |
| **Step-level decomposition** | Break into sub-steps, verify each | Engineering infrastructure works; verification depends on Z3/symbolic tools | parallel_l4_tot step plan |
| **CCE falsification** | Deterministically test conjectures up to N | Tests survival, not reasoning quality | discoveryLoop |
| **Recall proxy signals** | 4 heuristic signals for recall vs construction | Never calibrated; regex-based, no value verification | discoveryLoop |
| **Abduction (data-first)** | Present computed data, ask for patterns | Forces construction over recall; but model still produces trivial results on small LLMs | discoveryLoop |

---

## 4. What External Research Shows

### The Process Reward Model (PRM) Direction (Most Active)

The dominant research thread is **Process Reward Models** — training a separate model to score each reasoning step.

**Key papers:**

| Paper | Year | Key Finding |
|---|---|---|
| **OpenAI "Let's Verify Step by Step"** | 2023 | PRM outperforms outcome-based reward by 2x on MATH. Step-level supervision is essential. |
| **Math-Shepherd (Wang et al., ACL 2024)** | 2024 | Auto-label step correctness via continuation sampling — no human labels needed. Sample N continuations from step i; if majority reach correct answer, step i is "good." |
| **uPRM (Gadetsky et al., 2026)** | 2026 | Unsupervised PRM using next-token probabilities. No human supervision needed at any level. +15% over LLM-as-Judge on ProcessBench. |
| **ReProbe (Ni et al., 2025)** | 2025 | **Directly validates parallel_l4_tot findings.** Lightweight (<10M param) probe on frozen LLM internal states matches PRMs that are 810x larger. Internal states encode step credibility. |
| **GroundedPRM (Zhang et al., 2025)** | 2025 | Tree-guided + tool-verified process supervision. MCTS for credit assignment + external tool verification per step. 10% of data, 26% better than auto-labeled PRMs. |
| **SPAE (Wu et al., 2026)** | 2026 | Step Potential signal: training-free probing extracts intermediate confidence + correctness. Amplifies gains, penalizes drops, stops when saturated. |
| **SCRIBE (Jiang & Ferraro, 2026)** | 2026 | Mid-level skill-conditioned rewards. Qwen3-4B: 43.3% → 63.3% on AIME25. Mastery of mid-level skills precedes high-level planning. |

### The Mechanistic Interpretability Direction

| Paper | Year | Key Finding |
|---|---|---|
| **Ostmeier "Head Entropy"** | 2026 | Per-head attention entropy carries confidence signal. Individual heads at specific layers encode certainty. |
| **Nguyen "Attention-Aware Intervention"** | 2026 | Modifying attention patterns during CoT improves reasoning accuracy. Attention patterns have causal influence on reasoning quality. |
| **Shelmanov "Uncertainty Heads"** | 2025 | Auxiliary uncertainty quantification heads trained on LLM activations detect hallucinations. Activations more informative than attention. |
| **Chen "CoT Features" (AAAI 2026)** | 2026 | Sparse autoencoders reveal CoT induces modular internal structures. Feature-level causal study of reasoning. |

### The Key External Insight

**ReProbe (2025) directly confirms what parallel_l4_tot found empirically:** LLM internal states encode step-level correctness, and a tiny probe (<10M params) can extract this signal without any training data labels. This is the bridge between "the information exists" (Finding 1) and "the model doesn't use it" (the gap).

---

## 5. The Hypothesis: What's Actually Missing

### The Architecture Problem

Current transformers do everything through **next-token prediction over a single forward pass**. There is no architectural distinction between:

```
Recall:    "sigma(p) = p+1 for primes" → generated because it matches training data
Inference: "sigma(6) = 1+2+3+6 = 12"  → generated by computing from properties
```

Both are just `P(token_t | token_1...token_{t-1})`. The model has no mechanism to:

1. **Query its own confidence about a specific step** — it produces tokens, not confidence scores
2. **Detect logical inconsistency between steps** — there's no "consistency checker" in the architecture
3. **Select between competing approaches** — template selection happens implicitly in early layers with no explicit evaluation
4. **Know when it doesn't know** — uncertainty is distributed across hundreds of token decisions, each individually indistinguishable from correct

### The Training Objective Problem

Current models are trained to maximize `P(correct_next_token | context)`. This optimizes for:
- ✅ Fluency (produce coherent text)
- ✅ Recall (reproduce training data patterns)
- ❌ Internal consistency between steps
- ❌ Detection of logical contradictions
- ❌ Optimal problem decomposition
- ❌ Verification of intermediate results

### The "Not That Difficult" Insight

Looking at the evidence, I believe the missing piece is **not** a fundamental architecture change. It's something more specific:

**The model needs a learned self-verification signal that operates at the step level, trained to predict "this step follows from the previous step" as a separate objective from "this step looks like good text."**

This is different from PRM (which is an external model). This is about giving the model an **internal error-detection pathway** — essentially teaching it to predict its own mistakes during generation, not just after.

Why this might be "not that difficult":

1. **The information already exists** inside the model (Finding 1, confirmed by ReProbe). We don't need to create it; we need to route it.

2. **The probe is tiny** — ReProbe shows <10M params is sufficient. This isn't adding a second brain; it's adding a small readout head.

3. **The training signal is available** — Math-Shepherd showed you can auto-label step correctness without human annotation. The model can generate its own training data.

4. **The impact is multiplicative** — if each step has even a 95% chance of being correct, a 5-step chain is 77% reliable. If the model can detect and retry the 5% of wrong steps, reliability jumps to ~99%.

---

## 6. What Would a Self-Verifying Local LLM Look Like?

### Option A: Internal Probe Head (Lowest Cost)

Add a small prediction head (<10M params) that reads from intermediate layers and predicts step correctness during generation.

```
                    ┌──────────────────┐
                    │  Probe Head      │
                    │  (<10M params)   │
                    │  Input: L3-L12   │
                    │  Output: P(correct)│
                    └────────┬─────────┘
                             │
   Token_1 ... Token_N ──► Transformer ──► Token_{N+1}
                             │
                      [at step boundary]
                             │
                    ┌────────▼─────────┐
                    │ IF P(correct) < τ │
                    │ THEN: backtrack   │
                    │   + regenerate    │
                    └──────────────────┘
```

**Training:**
1. Generate N solutions per problem (Math-Shepherd style)
2. Auto-label each step as correct/incorrect using continuation sampling
3. Train probe head on hidden states at step boundaries → predict step correctness
4. Freeze transformer, train only the probe head

**Why this is feasible for a local LLM:**
- Probe head is <10M params — fits in memory alongside a 4B model
- Training requires only the frozen model + auto-generated labels
- No architecture change to the base model
- ReProbe showed this works with <10M params, matching PRMs 810x larger

### Option B: Step-Level RLVR (Medium Cost)

Train the model itself (not a probe) with step-level rewards using Reinforcement Learning with Verifiable Rewards.

```
Problem → Model generates Step 1
              │
         External verifier (Z3/sympy/execution)
              │
         Reward: +1 if correct, -1 if wrong, 0 if unverifiable
              │
         Update model weights via GRPO/PPO at STEP level
              │
         Model generates Step 2 (conditioned on verified Step 1)
              ...
```

**Training:**
1. Use existing tools (Z3 for arithmetic, code execution for PAL, symbolic math for algebra)
2. Generate solution chains with per-step verification
3. Backpropagate step-level rewards, not just outcome rewards
4. Key insight from SPAE (2026): penalize "potential drops" where model goes from on-track to off-track

**Why this changes things:**
- The model learns to predict step correctness as part of generation, not as a separate model
- Self-verification becomes internalized rather than external
- GRPO with step-level rewards is already proven (DeepSeekMath used outcome-level RLVR for 51.7% on MATH)

### Option C: Dual-Objective Training (Highest Cost, Most Fundamental)

Train the model with TWO objectives simultaneously:

```
Objective 1: P(correct_next_token | context)           — standard LM training
Objective 2: P(this_step_is_correct | context, step)   — verification training
```

The verification objective forces the model to develop internal representations that distinguish correct from incorrect reasoning. This is different from a probe head (Option A) because the base model's weights are shaped by the verification signal.

**Training data:**
- Correct solutions: from existing math benchmarks (GSM8K, MATH)
- Incorrect solutions: from model's own failed attempts (negative examples)
- Step labels: auto-generated via Math-Shepherd continuation sampling

**Why this is the "not that difficult" thing:**
- It doesn't require new architecture — just a new training objective
- The data is self-generated — the model produces both correct and incorrect solutions during training
- It directly addresses the core gap: the model learns to distinguish its own correct and incorrect reasoning
- The parallel_l4_tot findings prove the signal exists in hidden states — dual-objective training would make the model use it

---

## 7. Concrete Research Roadmap

### Phase 1: Validate the Probe Hypothesis (2-3 weeks)

Use the existing parallel_l4_tot infrastructure to test whether a probe head can be trained on the hidden states that causal tracing identified as informative.

1. **Data collection**: Run 200 problems × 3 temperatures = 600 paths with L2/L10/L12/L24 hidden state capture
2. **Auto-labeling**: Use Math-Shepherd continuation sampling to label each step correct/incorrect
3. **Probe training**: Train a <10M param probe on hidden states → step correctness
4. **Validation**: Test on held-out problems. Target: AUROC ≥ 0.85

**Existing infrastructure:** parallel_l4_tot has `LayerCaptureWrapper`, `entropy_capture.py`, and 600+ paths of attention data. The collection pipeline exists.

### Phase 2: Build the Self-Correction Loop (2-4 weeks)

If Phase 1 works, integrate the probe into the generation loop:

1. At each step boundary, run the probe
2. If P(correct) < threshold, backtrack and regenerate
3. Measure: does backtracking improve accuracy?

**Existing infrastructure:** parallel_l4_tot has the step decomposition pipeline, Z3 verifier, and PAL fallback. The correction loop just needs the probe decision gate.

### Phase 3: Train with Dual Objective (1-2 months)

If Phase 2 shows improvement, move from inference-time probing to training-time dual objectives:

1. Fine-tune the base model with step-level RLVR
2. Use Z3/code execution as the verification reward
3. Compare: probe-guided inference vs. dual-objective training vs. baseline

**Existing infrastructure:** discoveryLoop has the CCE falsification, abduction engine, and measurement infrastructure. The RLVR training loop would be new but builds on existing evaluation tools.

---

## 8. Why This Hasn't Emerged Yet

Three reasons the industry hasn't converged on this:

1. **The PRM detour**: The field has been focused on external Process Reward Models — a separate model that scores steps. This works but is expensive (you need two models) and doesn't make the base model better. The insight that internal states carry the signal (ReProbe, our causal tracing) is very recent (late 2025).

2. **Scale chauvinism**: The dominant belief is "bigger models will solve this." And they do — GPT-4 and Claude reason better than 4B models. But the parallel_l4_tot research shows the *mechanism* of failure is the same at all scales: template selection, no self-verification. Scale masks the problem; it doesn't fix it.

3. **The measurement gap**: Until discoveryLoop's measurement infrastructure, there was no way to distinguish "the model recalled the answer" from "the model inferred the answer." Without this measurement, you can't tell whether an intervention improves reasoning or just improves recall. The UC-1 through UC-4 framework from GOAL.md provides the measurement.

---

## 9. Summary of Key Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 1 | Correctness information exists in hidden states at identifiable positions | parallel_l4_tot causal tracing: AUC=1.0; ReProbe: <10M probe matches 810x larger PRM | **High** |
| 2 | The model doesn't use this information to self-correct | parallel_l4_tot: wrong paths have same/higher attention than correct; no backtracking behavior observed | **High** |
| 3 | Attention patterns cannot detect semantic errors | parallel_l4_tot: 179 controlled paths, wrong formulas indistinguishable from correct | **Very High** |
| 4 | A tiny probe head (<10M params) can extract step-correctness signal | ReProbe (2025): validated across math, planning, QA domains | **High** |
| 5 | Auto-labeling step correctness doesn't require human annotation | Math-Shepherd (2024), uPRM (2026) | **High** |
| 6 | PAL + step decomposition + verification recovers most errors | parallel_l4_tot: 22% → 78% on GSM8K | **Very High** |
| 7 | The remaining 18% algebraic floor requires internal self-verification | parallel_l4_tot format analysis: no external preprocessing fixes operator confusion | **Moderate** |
| 8 | Dual-objective training (generation + verification) could internalize self-correction | Theoretical; no direct experimental evidence yet | **Speculative** |

---

## 10. Papers Referenced

| Paper | Year | Venue | Relevance |
|---|---|---|---|
| Lightman et al. "Let's Verify Step by Step" | 2023 | OpenAI | PRM outperforms ORM by 2x |
| Wang et al. "Math-Shepherd" | 2024 | ACL | Auto-labeling step correctness via continuation sampling |
| Ostmeier et al. "Head Entropy" | 2026 | arXiv | Per-head entropy carries confidence signal |
| Nguyen et al. "Attention-Aware Intervention" | 2026 | EACL | Attention modification improves reasoning |
| Shelmanov et al. "Uncertainty Heads" | 2025 | EMNLP | Activation-based uncertainty detection |
| Chen et al. "CoT Features" | 2026 | AAAI | SAE reveals modular CoT structures |
| Meng et al. "ROME" | 2022 | NeurIPS | Causal tracing methodology |
| Gadetsky et al. "uPRM" | 2026 | arXiv | Unsupervised PRM from next-token probabilities |
| Ni et al. "ReProbe" | 2025 | arXiv | Internal state probe matches large PRMs |
| Zhang et al. "GroundedPRM" | 2025 | arXiv | MCTS + tool-verified process supervision |
| Wu et al. "SPAE" | 2026 | arXiv | Step potential for intermediate confidence |
| Jiang & Ferraro "SCRIBE" | 2026 | arXiv | Mid-level skill-conditioned rewards |
| Liu et al. "Graph Reasoning Paradigm" | 2026 | arXiv | Structured symbolic reasoning with topology-aware RL |
| Williamson et al. "Syntactic Blind Spots" | 2025 | MathNLP@ACL | DLT scoring + WLK matching for narrative simplification |
| Srivatsa et al. "What Makes MWPs Challenging" | 2024 | — | 23 features predicting MWP difficulty |

---

## 11. Existing Repo Assets Mapped to This Research

| Asset | Location | How it supports the research |
|---|---|---|
| LayerCaptureWrapper | `parallel_l4_tot/` | Captures L2/L10/L12/L24 norms during generation — infrastructure for probe training |
| Causal tracing pipeline | `parallel_l4_tot/causal_trace/` | Identifies divergence points per problem — where to attach probes |
| Entropy features (59 signals) | `parallel_l4_tot/SIGNAL_CATALOG.md` | Baseline features that a probe would augment |
| Step decomposition plan | `parallel_l4_tot/PLAN_STEP_LEVEL_DECOMPOSITION.md` | 7-phase implementation plan for step-level verification |
| PAL pipeline + Z3 verifiers | `parallel_l4_tot/verifiers/` | External verification infrastructure for auto-labeling |
| CCE falsification | `discoveryLoop/cce.py` | Deterministic testing of conjectures — ground truth for recall vs construction |
| Abduction engine | `discoveryLoop/abduction.py` | Data-first generation that forces construction over recall |
| Recall detector | `discoveryLoop/recall_detector.py` | 4 proxy signals — baseline for measuring recall vs construction |
| Novelty checker | `discoveryLoop/novelty.py` | OEIS + known-results DB — ground truth for conjecture novelty |
| Experiment protocol | `discoveryLoop/EXPERIMENT.md` | Pre-registered A/B experiment design |
| Measurement infrastructure | `discoveryLoop/ULTRAPLAN_MEASUREMENT.md` | 5-wave plan for measurement gap closure |
| Reasoning verifier | `discoveryLoop/reasoning_verifier.py` | Extracts numerical claims and verifies against computed data |
| 49 passing tests | `discoveryLoop/tests/` | Mock-based test coverage of all modules |
| Qwen3.5-4B-MLX-4bit on disk | `/Volumes/BUF_2T_02/` | Local model for all experiments |
