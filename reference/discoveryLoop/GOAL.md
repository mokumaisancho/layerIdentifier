# Goal Decomposition: Autonomous Inference Engine

## Research Question

**Can an LLM autonomously arrive at a true, novel conclusion through
iterative logical inference and self-criticism — without the answer being
recallable from training data?**

This is NOT a math tool. Math is the testbed because it's cheaply falsifiable.
The real question is about **reasoning vs. recall**.

---

## The Core Distinction

```
RECALL:    Model outputs "n²+n+41 is prime for n<40" because it saw this
           in Wikipedia during training. Zero inference. Not discovery.

INFERENCE: Model generates "for composite n, sigma(n) > sigma(n-1) + sigma(n+1)"
           by REASONING from properties of divisors. Even if wrong, the
           reasoning chain is the discovery mechanism.
```

**What we want to measure: reasoning quality. What we currently measure: predicate survival.**

These are completely different things.

---

## Use Cases

### UC-1: Novel Conjecture Discovery

**Actor**: Researcher
**Goal**: Find a mathematical statement that is (a) true within test range,
         (b) not in standard references, (c) arrived at through reasoning.

**Flow**:
1. Researcher specifies a domain (e.g., "properties of sigma(n)")
2. System runs N rounds of generate→test→refine
3. Surviving conjectures are checked against known results
4. Non-trivial survivors are surfaced with full reasoning chains

**Acceptance Criteria**:
- [ ] System produces ≥ 1 conjecture per 3 runs that survives n=1..100,000
- [ ] That conjecture does NOT appear in OEIS, Wikipedia "List of conjectures",
      or Hardy & Wright's "Introduction to the Theory of Numbers"
- [ ] The reasoning chain is traceable: each step shows WHAT evidence led to it
- [ ] The conjecture is not a trivial restatement (e.g., "is_prime(n) for prime n")

**Current gap**: No novelty detection. No known-results database. No reasoning trace.

---

### UC-2: Reasoning Chain Soundness Assessment

**Actor**: Researcher
**Goal**: Measure whether each refinement step is logically connected to its
         evidence, or is random mutation.

**Flow**:
1. System runs a loop producing a chain of conjecture → falsification → refinement
2. Each refinement is classified as:
   - STRUCTURAL: changes the logical form (e.g., "all n" → "odd n only")
   - EXCLUSION: adds a guard (e.g., "n ≠ 40") — addresses symptom, not cause
   - RANDOM: unrelated to the counterexample
3. Soundness score = structural refinements / total refinements

**Acceptance Criteria**:
- [ ] Each refinement is classified with type + justification
- [ ] Soundness score ≥ 0.5 means at least half the refinements are structural
- [ ] Full reasoning chain is human-readable (not just predicate→predicate)
- [ ] Chain shows PROGRESSIVE CONSTRAINT: hypothesis space narrows over rounds

**Current gap**: Refinements are unclassified. System can't distinguish "n≠40"
from "odd n only". No chain visualization beyond round outcomes.

---

### UC-3: Inference Quality Benchmarking

**Actor**: Model evaluator
**Goal**: Compare models on their ability to reason (not recall).

**Flow**:
1. Run identical domain + seed on Model A and Model B
2. Measure:
   - Novelty rate: % of conjectures not in known-results database
   - Soundness: structural refinement ratio (UC-2)
   - Efficiency: rounds-to-first-survivor
   - Diversity: unique conjecture classes generated (not just rephrasings)
3. Output comparison report

**Acceptance Criteria**:
- [ ] Same input produces deterministic output when model is deterministic
- [ ] Metrics are model-independent (work for any Ollama model)
- [ ] At least 3 models can be compared (qwen2.5:3b, qwen2.5:14b, qwen3.5:4b)
- [ ] Results are reproducible across runs (±10% for stochastic models)

**Current gap**: No model comparison. No reproducibility controls. Metrics are
survival-based, not inference-based.

---

### UC-4: Knowledge Boundary Exploration

**Actor**: Researcher
**Goal**: Distinguish what the model RECALLS from what it CONSTRUCTS.

**Flow**:
1. Run system on domain X
2. Classify each conjecture as:
   - RECALLED: matches known result verbatim or near-verbatim
   - CONSTRUCTED: combines concepts in a way not in training corpus
   - HYBRID: recalled components, novel combination
3. Map the boundary: where does recall end and construction begin?

**Acceptance Criteria**:
- [ ] Known-results database covers ≥ 50 standard number theory conjectures
- [ ] Classification accuracy ≥ 80% vs. human judgment on 20-example test set
- [ ] System outputs a "knowledge boundary map" showing recall vs. construction
- [ ] Construction rate increases with domain specificity (less training data)

**Current gap**: No known-results database. No recall detection. No boundary mapping.

---

## What's Missing in the Current Implementation

| Use Case | What exists | What's needed |
|----------|-------------|---------------|
| UC-1 Novelty | CCE falsification, survival tracking | Known-results DB, novelty scoring, reasoning traces |
| UC-2 Soundness | Round-by-round outcomes | Refinement classification, structural vs. exclusion detection, chain visualization |
| UC-3 Benchmarking | Single-model loop | Model-agnostic metrics, reproducibility, comparison report |
| UC-4 Boundary | Nothing | Recall detection, construction classification, boundary mapping |

## Priority Order

1. **UC-2 (Soundness)** — Refinement classification is the most fundamental gap.
   Without it, we can't tell if the system is reasoning or guessing.
   ETA: 1-2 hours.

2. **UC-4 (Boundary)** — Recall vs. construction is the core research question.
   Even a simple known-results database would be transformative.
   ETA: 2-3 hours.

3. **UC-1 (Novelty)** — Depends on UC-2 + UC-4. Once we can measure soundness
   and detect recall, novelty becomes: sound + non-recalled.
   ETA: builds on UC-2 + UC-4.

4. **UC-3 (Benchmarking)** — Requires all above metrics to be stable.
   ETA: after UC-1 through UC-3.

## Current Implementation vs. Goal

```
WHAT WE BUILT:                          WHAT THE GOAL REQUIRES:
┌─────────────────────────┐            ┌─────────────────────────┐
│ Generate predicate      │            │ Reasoning chain trace   │
│ Test via CCE            │            │ Why was this proposed?  │
│ Refine from CE          │            │ What evidence connects  │
│ Score survival          │            │   each step?            │
│ Decompose survivors     │            │                         │
│                         │            │ Soundness classification│
│ MEASURES: survival      │            │   structural vs. guard  │
│                         │            │                         │
│                         │            │ Novelty detection       │
│                         │            │   recalled vs. reasoned │
│                         │            │                         │
│                         │            │ MEASURES: inference     │
└─────────────────────────┘            └─────────────────────────┘
```

The left column is engineering infrastructure. The right column is what
actually answers the research question. We need both — but we only have
the left.
