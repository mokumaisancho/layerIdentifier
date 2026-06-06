# Three Approaches to Separating Knowledge from Reasoning in LLMs

**Date**: 2026-06-02
**Context**: InferenceOnlyfromLocal — can we extract pure reasoning from a local LLM by surgically removing or suppressing recalled knowledge?

---

## The Central Question

All 19 repos orbit one question:

**Is "recall" vs "reasoning" a real distinction in the model's activation space, or is it a false dichotomy we're imposing from the outside?**

This is the fork. If they're separable — a linear direction, distinct features, a causal locus — then all downstream work (suppress recall, amplify reasoning, force novelty) has a mechanistic foundation. If they're entangled — the same circuits do both, inseparably — then the entire initiative is built on a category error, and that negative result is itself the publishable contribution.

The goal everyone is working toward ("Can an LLM reason without recalling?") is downstream of this question. You can't suppress what doesn't exist as a separate mechanism.

The empirical evidence so far:
- `discoveryLoop`: Raw LLM produces 0% novel conjectures. Abduction engine raises novelty but triviality dominates.
- `InferenceLLM2`: The model "knows" when wrong (causal tracing, AUC=1.0 at L3) but has no architectural pathway to use that signal.
- `cot-correctness-probe`: CoT correctness peaks at L16 (50% depth). The model *decides* correctness mid-layers.
- `parallel_l4_tot`: Best pipeline (80-90% GSM8K) but still mixes recall + reasoning inseparably.
- `conversationHarness`: Context hurts 4/10 times. More information ≠ better reasoning.

The hypothesis: if we could physically separate "knowledge retrieval" from "reasoning computation" inside the model's activation space, we could suppress the former and amplify the latter.

Three techniques from mechanistic interpretability claim to do exactly this. Below: what each is, what it actually does, and what it means for *this specific project ecosystem*.

---

## 1. Representation Engineering (RepE) — Contrastive Activation Steering

### The Paper

**Zou et al. (2023)** "Representation Engineering: A Top-Down Approach to AI Transparency"
- 21 authors: CMU, Stanford, MIT, Berkeley, Northeastern
- arXiv: 2310.01405 | 1,030 citations
- PDF: https://arxiv.org/pdf/2310.01405

### What It Is

RepE adapts cognitive neuroscience methods (fMRI-style population analysis) to neural networks. Instead of asking "what does neuron #4711 do?" (bottom-up), it asks "what direction in activation space distinguishes condition A from condition B?" (top-down).

**Method**:
1. Collect contrastive pairs: same prompt, two conditions (e.g., "answer honestly" vs "answer deceptively")
2. Extract hidden states at target layers for both conditions
3. Compute reading vector: **r = mean(condition_A_states) − mean(condition_B_states)**
4. During generation, add **+αr** to steer toward condition A, or **−αr** to suppress

**What they proved**: High-level cognitive behaviors (honesty, harmlessness, power-seeking, emotion) are *linearly encoded* in hidden-state space. A single vector captures each behavior and can be added/subtracted at inference time.

### Significance for This Ecosystem

| Ecosystem Repo | Connection | What RepE Enables |
|---|---|---|
| `InferenceLLM2` | Direct feed. RepE reading vectors ARE the "correctness signal" they're looking for | Compute a "reasoning vs recall" steering vector at L15-L16, apply during generation to suppress recall |
| `cot-correctness-probe` | Same layer (L16), same model (Qwen3.5-4B) | The correctness signal at L16 may already BE a reasoning-vs-recall distinction. RepE would make it explicit and steerable |
| `parallel_l4_tot` | Already captures per-path hidden states | Add steering vector to boost "reasoning" paths and suppress "recall" paths before scoring |
| `entropy_router` | RepE could replace entropy as the routing signal | Instead of entropy-gating, use a "novelty steering vector" to classify routine vs novel |
| `inferenceFocusLLM` | Layer-drop may accidentally remove the steering layer | RepE tells you WHICH layers encode the recall/reasoning distinction — don't drop those |
| `discoveryLoop` | The 0% novelty problem is a recall-dominance problem | Apply anti-recall steering during conjecture generation |

**Why it matters**: RepE is the cheapest technique to try. No training required. The Wave 0 data (25 recall + 25 novel captures) already provides the contrastive pairs. A steering vector can be computed and applied in an afternoon.

**Critical limitation**: Steering is a *soft nudge*, not a surgical excision. It shifts the probability distribution but doesn't guarantee reasoning-only output. The model will still sometimes recall when you tell it not to.

### Concrete Next Step

```
1. Load Wave 0 activations (25 recall, 25 novel) from InferenceLLM2
2. Extract L16 hidden states for both conditions
3. Compute r = mean(novel_states) - mean(recall_states)
4. Apply +αr during generation on 50 new math problems
5. Measure: does it increase novelty without destroying correctness?
```

Estimated cost: 2-4 hours on M1 16GB. No additional training.

---

## 2. Sparse Autoencoders (SAEs) — Feature Decomposition

### The Papers

**Cunningham et al. (2023)** "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
- 5 authors: UNC Chapel Hill, Independent, Google DeepMind
- arXiv: 2309.08600 | ICLR 2024 | 1,201 citations
- PDF: https://arxiv.org/pdf/2309.08600

**Bricken et al. (2023)** "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic)
- Part of Anthropic's interpretability research program
- Introduced the "scaling law for SAEs" framework

**Key follow-ups**:
- **Gemma Scope** (Lieberum et al., 2024): Open SAEs trained on every layer of Gemma 2
- **HSAE** (Luo et al., 2026): Hierarchical SAEs capturing feature parent-child relationships
- **Monet** (Park et al., 2024): Mixture-of-Monosemantic-Experts with 262K experts per layer

### What It Is

Neurons are *polysemantic* — one neuron responds to "DNA sequences" AND "Python code" AND "French grammar." SAEs decompose these into *monosemantic* features where each feature responds to exactly one concept.

**Method**:
1. Collect activation vectors from a target layer across many inputs
2. Train autoencoder: large hidden layer (e.g., 16K–256K features from 2560-dim activations) + L1 sparsity penalty
3. Each learned feature becomes interpretable: "mathematical proof," "DNA sequence," "sycophantic language," etc.
4. Cluster features into groups (knowledge retrieval, reasoning, formatting, etc.)

**What they proved**:
- Features are causally responsible for behavior (ablation of specific features changes output predictably)
- Features cluster into semantically meaningful groups
- Anthropics "golden gate" feature demonstrates single-feature steering
- Features can identify what the model "knows" vs what it "computes"

### Significance for This Ecosystem

| Ecosystem Repo | Connection | What SAEs Enable |
|---|---|---|
| `InferenceLLM2` | Replace the binary probe with SAE feature analysis | Instead of "is this correct?" (AUC=1.0 binary), get "which features caused the error?" (mechanistic) |
| `redundantHeads` | Orthogonal. Head pruning removes entire attention heads; SAEs decompose MLPs | SAEs could identify if "redundant heads" are actually doing recall that could be selectively suppressed |
| `memory_entropy_validator` | Direct complement | SAE features replace entropy as the "is this genuine information?" signal |
| `layerprobe` | Same layer-wise analysis philosophy | SAEs add feature-level granularity to layer importance scoring |
| `discoveryLoop` | Identify WHY the LLM produces 0% novel | SAE features would show: during conjecture generation, "recall" features fire and "reasoning" features don't |
| `conversationHarness` | SAEs could detect "context is activating recall features" | Replace the semantic drift detector with feature-level monitoring |

**Why it matters**: SAEs give you a *decomposition* of the model's internal computation. Instead of "Layer 16 is important," you get "Feature #3847 encodes prime-number recall and fires during Step 3." This is the only technique that provides feature-level granularity.

**Critical limitation**: The separation is statistical, not categorical. A "prime number knowledge" feature and a "divisibility reasoning" feature often co-activate. You can identify which features *lean* toward recall, but can't cleanly zero them out without degrading reasoning. Training an SAE on Qwen3.5-4B at L15 takes ~2-4 hours on M1, but may require 50K+ activation samples for quality features.

**Cost vs. benefit analysis**:
- Training time: 2-4 hours on M1 16GB
- Data needed: 50K+ activation samples (forward passes through L15)
- Interpretability work: manual or automated feature labeling (days)
- Payoff: mechanistic understanding of recall vs reasoning (unique to SAEs, not available from RepE or ROME)

### Concrete Next Step

```
1. Collect 50K L15 activations from Qwen3.5-4B across diverse math problems
2. Train SAE with 16K features, L1 penalty (standard recipe from Cunningham et al.)
3. Label features manually: present top-activating inputs for each feature
4. Identify "knowledge recall" vs "reasoning" feature clusters
5. Ablation experiment: suppress top-K recall features during novel problem solving
```

Estimated cost: 2-4 hours training + 1-2 days feature labeling. Medium effort, high insight.

---

## 3. Causal Tracing / Knowledge Editing (ROME/MEMIT)

### The Papers

**Meng et al. (2022)** "Locating and Editing Factual Associations in GPT" (ROME)
- 4 authors: MIT, Northeastern
- arXiv: 2202.05262 | 174 citations (OpenAlex)
- PDF: https://arxiv.org/pdf/2202.05262

**Meng et al. (2023)** "Mass-Editing Memory in a Transformer" (MEMIT)
- 5 authors: MIT, Northeastern
- arXiv: 2210.07229 | ICLR 2023 | 1,004 citations
- PDF: https://arxiv.org/pdf/2210.07229

### What It Is

ROME and MEMIT apply *causal intervention* to identify where specific facts are stored in a transformer. Don't just observe activations — corrupt them, restore them, and measure the effect on output.

**Method (Causal Tracing)**:
1. Run model on prompt ("The Eiffel Tower is in ___"), record all hidden states
2. **Corrupt**: add noise to a specific (layer, token position) activation
3. **Restore**: replace corrupted activation with the clean one from a different run
4. Measure: which restorations recover the correct output? Those are "causally decisive"
5. **Finding**: mid-layer MLP modules at subject tokens store factual associations as key-value pairs

**Method (ROME/MEMIT editing)**:
1. Identify the MLP weight matrices that store a specific fact
2. Compute a rank-one update to those weights that changes the stored association
3. Apply the update — the model now outputs the new fact
4. MEMIT extends this to thousands of simultaneous edits

**What they proved**:
- Facts are stored as key-value pairs in mid-layer MLP weights (approximately layers 12-20 in a 32-layer model)
- Individual facts can be surgically edited without affecting other knowledge
- MEMIT scales to thousands of simultaneous edits with minimal degradation
- The causal locus of "Eiffel Tower → Paris" is the MLP at "Eiffel Tower" token position in layers 12-20

**Critical caveat (Hase et al., 2023)**: "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing." Localization and editability are *not* the same thing. Where you find a fact is not necessarily where you can best edit it. This paper (22 citations) found that ROME's edits sometimes degrade unrelated capabilities.

### Significance for This Ecosystem

| Ecosystem Repo | Connection | What ROME/MEMIT Enables |
|---|---|---|
| `InferenceLLM2` | Direct parallel — both do causal tracing | ROME provides the surgical edit capability that InferenceLLM2's "probe head" lacks |
| `redundantHeads` | Complementary — ROME traces MLPs, redundantHeads traces attention | Together they'd cover both pathways: attention heads (redundantHeads) + MLP knowledge stores (ROME) |
| `research_gguf_surgery` | The GGUF surgery repo is the deployment target | ROME/MEMIT edits on GGUF-quantized models would be the final deliverable |
| `layerprobe` | Same goal (identify important layers), different method | Causal tracing provides ground truth for layerprobe's importance scoring |
| `inferenceFocusLLM` | If factual knowledge is in layers 12-20, those are the layers you MUST NOT drop | ROME identifies exactly which layers store knowledge, informing layer-drop decisions |
| `discoveryLoop` | Theoretical: could we *delete* the model's knowledge of specific conjectures, forcing it to reason? | Extreme intervention: remove "known conjecture" facts and see if the model can reconstruct them via reasoning |

**Why it matters**: ROME/MEMIT is the only technique that provides *surgical editing* — you can actually change what the model knows. If you could identify all "prime number facts" in layers 12-20 and suppress them, the model would be forced to reason rather than recall.

**Critical limitation**: Causal tracing is per-fact and per-token. It tells you where "Eiffel Tower → Paris" is stored, NOT where "all mathematical knowledge" lives. You'd need to trace hundreds of facts and look for common patterns. This is expensive. Also, the Hase et al. (2023) finding that localization ≠ editability means you might find where knowledge lives but be unable to cleanly remove it.

**Cost vs. benefit analysis**:
- Tracing cost: ~1 minute per fact, ~100 facts needed = ~2 hours
- Edit validation: ~1 hour per edit batch
- Risk: edits may degrade unrelated capabilities (Hase et al.)
- Payoff: the only technique that can *physically remove knowledge* from the model

### Concrete Next Step

```
1. Select 20 prime number facts and 20 reasoning patterns
2. Causal trace each on Qwen3.5-4B
3. Map: do prime facts cluster in specific MLP layers/positions?
4. If yes: MEMIT batch-suppress those facts
5. Test: does the model reason (correctly) about suppressed facts?
```

Estimated cost: 4-8 hours. Highest risk, highest potential payoff.

---

## Comparative Analysis: Which to Pursue First?

```
                     Cost    Insight   Intervention   Risk
                     ─────   ───────   ───────────    ─────
RepE Steering        LOW     MEDIUM    Soft nudge     LOW
SAE Decomposition    MEDIUM  HIGH      Indirect       LOW
ROME/MEMIT Tracing   HIGH    HIGH      Surgical       HIGH
```

### Recommended Sequence

**Phase 1 (Day 1-2): RepE Steering**
- Cheapest, fastest, uses existing Wave 0 data
- Tests the core hypothesis: "is there a linear direction separating recall from reasoning?"
- If the steering vector has near-zero effect, the entire premise is wrong — stop early
- If it has measurable effect, it quantifies the separability of the two modes

**Phase 2 (Day 3-5): SAE Training**
- Only pursue if Phase 1 shows separability
- Provides mechanistic understanding: *which specific features* correspond to recall vs reasoning
- Feature-level analysis guides all downstream interventions

**Phase 3 (Day 5-10): ROME Tracing + MEMIT Editing**
- Only pursue if Phase 2 identifies clear "recall feature clusters"
- Use causal tracing to validate that the SAE-identified features are causally decisive
- Attempt surgical suppression via MEMIT

### What Failure Looks Like

- **RepE shows no separability**: Recall and reasoning are not linearly separable in this model. The question "knowledge vs reasoning" is a false dichotomy at the activation level.
- **SAE features co-activate**: Knowledge and reasoning features are entangled — the same features support both. Suppressing recall necessarily suppresses reasoning.
- **ROME edits degrade reasoning**: Removing specific facts causes collateral damage to reasoning pathways. The model cannot reason about what it doesn't know.

Any of these failures would be a genuine research contribution. They would prove that the knowledge/reasoning distinction is not implementable at the activation level for models of this scale.

---

## Paper Citations

| Paper | Authors | Year | Citations | Venue |
|---|---|---|---|---|
| Representation Engineering | Zou et al. | 2023 | 1,030 | arXiv:2310.01405 |
| ROME: Locating and Editing Factual Associations | Meng et al. | 2022 | 174 | arXiv:2202.05262 |
| MEMIT: Mass-Editing Memory in a Transformer | Meng et al. | 2023 | 1,004 | ICLR 2023, arXiv:2210.07229 |
| SAEs Find Highly Interpretable Features | Cunningham et al. | 2023 | 1,201 | ICLR 2024, arXiv:2309.08600 |
| Towards Monosemanticity | Bricken et al. (Anthropic) | 2023 | — | Anthropic Research |
| Does Localization Inform Editing? | Hase et al. | 2023 | 22 | arXiv:2301.04213 |
| Gemma Scope | Lieberum et al. | 2024 | 26 | BlackboxNLP 2024 |
| HSAE: Hierarchical SAEs | Luo et al. | 2026 | — | arXiv:2602.11881 |
| Monet: Mixture of Monosemantic Experts | Park et al. | 2024 | — | arXiv:2412.04139 |

---

## Ecosystem Dependency Map

```
                         RepE Steering
                              │
                              ▼
                    ┌─────────────────┐
                    │  Is there a     │──NO──▶ Hypothesis falsified.
                    │  linear recall/ │       Publish negative result.
                    │  reasoning      │
                    │  direction?     │
                    └────────┬────────┘
                             │ YES
                             ▼
                    ┌─────────────────┐
                    │  SAE Training   │
                    │  Which features │
                    │  are recall?    │
                    │  Which are      │
                    │  reasoning?     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  ROME Tracing   │
                    │  Are the SAE-   │
                    │  identified     │
                    │  recall features│
                    │  causally       │
                    │  decisive?      │
                    └────────┬────────┘
                             │
                    ┌────YES─┴──NO───┐
                    ▼                ▼
              MEMIT Editing    Features are
              Suppress recall  correlational,
              during novel     not causal.
              inference.       Try RepE alone.
```

### Receiving Repos

If Phase 1 succeeds, the steering vector feeds directly into:
- `InferenceLLM2` — as the Probe Head's input signal
- `parallel_l4_tot` — as a path-level scoring feature
- `discoveryLoop` — as an anti-recall intervention during conjecture generation
- `entropy_router` — as a replacement routing signal

If Phase 2 succeeds, the SAE features feed into:
- `redundantHeads` — cross-reference attention head roles with SAE feature clusters
- `memory_entropy_validator` — replace entropy with feature activation as the "genuine information" test
- `conversationHarness` — feature-level context utility scoring

If Phase 3 succeeds, MEMIT edits deploy to:
- `research_gguf_surgery` — apply knowledge suppression to quantized GGUF models
- `inferenceFocusLLM` — inform which layers are safe to drop

---

## Appendix A: The Realm Hypothesis

### Why "Recall vs Reasoning" May Be the Wrong Cut

The binary framing above — recall on one side, reasoning on the other — may be too coarse. A different view:

**Reasoning is not one thing. It is partitioned into realms.**

A *realm* is a bounded set of inferential operations grounded in a specific physicality:

| Realm | Physical anchor | Core operation | Finite set of moves |
|---|---|---|---|
| **Spatial** | Bodies in space | Rotation, containment, adjacency | Small, enumerable |
| **Temporal** | Before/after, cause/effect | Sequencing, precedence, duration | Small, enumerable |
| **Counting** | Discrete objects | Increment, compare, partition | Small, enumerable |
| **Arithmetic** | Quantity conservation | Add, subtract, compose | Finite axiom set |
| **Algebraic** | Structural symmetry | Substitute, balance, invert | Finite rule set |
| **Geometric** | Shape invariance | Congruence, similarity, projection | Axiomatizable |
| **Probabilistic** | Uncertain outcomes | Weight, sample, update | Finite calculus |
| **Ethical** | Social cooperation | Fairness, harm, reciprocity | Not axiomatizable — but still bounded by physical sociality |

The key claim: **rationality is strongly bonded with physicality.** We can reason about numbers because we can count physical objects. We can reason about symmetry because physical bodies are symmetric. We can reason about causality because physical events have temporal order. Even abstract mathematics is ultimately grounded — often several layers of abstraction deep — in some physical intuition.

### What This Predicts About Activation Space

If the realm hypothesis is correct, the model's internal representations should show:

1. **Realm clusters, not a recall/reasoning axis.** Spatial reasoning features cluster separately from algebraic reasoning features, not along a single "reasoning" direction.

2. **Physically-grounded realms are more distinct.** Realms close to direct physical experience (spatial, temporal, counting) should form tighter, more separable clusters in activation space. Abstract realms (ethics, aesthetics) should be more diffuse — less physicality to anchor on.

3. **Connections exist between realms, not a single undifferentiated capability.** There are bridges: algebraic reasoning borrows from counting, geometric from spatial. These bridges should appear as *pathways* between realm clusters, not as a single "reasoning region."

4. **Recall is a substrate within each realm, not a separate realm.** "Recalling prime numbers" lives inside the arithmetic realm. "Recalling the capital of France" lives inside a geographic/linguistic realm. There is no unified "recall center" — recall is realm-specific.

### Implications for the Three Techniques

This reframes what each technique should look for:

**RepE Steering**: Instead of one "recall vs reasoning" vector, there should be *multiple* contrastive directions — one per realm. The "spatial reasoning vs spatial recall" direction should be different from the "algebraic reasoning vs algebraic recall" direction. If they're all the same direction, the realm hypothesis is wrong. If they differ, the realm hypothesis has activation-space evidence.

**SAE Decomposition**: Instead of clustering features into two groups ("recall" and "reasoning"), features should cluster into realm-specific groups. Within each realm, you'd find recall features and reasoning features. The hierarchy would be: realm → (recall features, reasoning features), not (recall) vs (reasoning).

**ROME/MEMIT**: Causal tracing should show that factual knowledge is stored *by realm* — prime number facts near arithmetic features, capital-city facts near linguistic features — not in a single "knowledge layer." MEMIT edits to arithmetic facts should not degrade spatial reasoning.

### Connection to Existing Work

| Repo | Realm-based reinterpretation |
|---|---|
| `discoveryLoop` | 0% novelty because the model has no "conjecture realm" — it only has a "mathematical recall realm." The abduction engine works because it forces the model into operations that don't belong to any existing realm. |
| `InferenceLLM2` | AUC=1.0 "correctness" at L3 may actually be **realm membership detection** — the model knows *which realm* it's operating in, and signals when the output doesn't belong to that realm. |
| `cot-correctness-probe` | L16 correctness peak may be the layer where **realm selection crystallizes** — the model commits to a realm and the output is locked in. |
| `parallel_l4_tot` | Multiple temperatures work because they **sample different realms**, not because they explore different "reasoning paths." Temperature ≈ realm hopping. |
| `research_topological_inference` | "Reasoning as navigating paths between topoi on a topological grid" — the topoi ARE the realms. The grid topology encodes the physicality-bonded connections between them. This is the realm hypothesis expressed mathematically. |
| `entropy_router` | Entropy signals may differ by realm. High entropy in a familiar realm = uncertainty within known operations. High entropy in an unfamiliar realm = the model has no realm to ground on. These are different failure modes. |
| `inferenceFocusLLM` | Layer-drop destroys some realms but not others. If geometric reasoning lives in layers 8-14 and algebraic reasoning in layers 12-18, dropping layers 15+ kills algebra but preserves geometry. |
| `redundantHeads` | "10% of heads are redundant for math" — redundant for WHICH realm of math? Arithmetic heads vs geometric heads may show different redundancy patterns. |

### A Testable Prediction

The realm hypothesis makes a specific, falsifiable prediction that none of the three techniques alone can test:

**Prediction**: Contrastive pairs drawn from the same realm (e.g., "recall this prime" vs "prove this number is prime") will produce a steering vector that is *different* from pairs drawn from a different realm (e.g., "recall this capital" vs "deduce this capital from neighboring countries"). If all steering vectors are the same regardless of realm, the realm hypothesis is falsified.

**Test**:
```
1. Collect contrastive pairs from 4 realms:
   - Arithmetic (recall primes vs prove primality)
   - Spatial (recall distances vs deduce from geometry)
   - Temporal (recall dates vs deduce from cause/effect)
   - Linguistic (recall definitions vs infer meaning from context)

2. Compute RepE steering vectors for each realm independently

3. Measure pairwise cosine similarity between the 4 steering vectors

4. If similarity > 0.9: recall/reasoning is realm-independent → binary hypothesis correct
   If similarity < 0.6: recall/reasoning is realm-specific → realm hypothesis supported
   If 0.6 < similarity < 0.9: partial realm structure → refine
```

This test costs the same as Phase 1 (RepE) but produces a richer result.

### Why This Matters for the Initiative

If the realm hypothesis holds, the entire initiative shifts from:

> "Suppress recall, amplify reasoning"

to:

> "Navigate to the correct realm, then suppress realm-specific recall"

This is a harder problem but a more honest one. It explains why blunt approaches (temperature, prompting, context engineering) produce inconsistent results — they don't address realm structure. And it connects the topological inferencing work (which was right all along) to the mechanistic interpretability tools that can validate it.

### Correction: Perspective, Not Partition

The realm tables above draw hard lines for simplicity. But the actual structure is not partitioned — **perspective is the only distinction.**

The underlying activation space is one continuous manifold. What we call "realms" are projections of that manifold — the same space viewed from different angles. A cylinder projects as a circle from one perspective and a rectangle from another. Same object. No boundary between "circle realm" and "rectangle realm."

Concretely, this means:

- **There is no boundary to find.** The three techniques won't discover sharp edges between realms because sharp edges don't exist. What they'll find are smooth gradients — regions where one perspective dominates, transitional zones where perspectives blend.
- **Realm clusters are perspectival artifacts.** If SAE features cluster into "arithmetic" and "spatial" groups, that's a projection, not a partition. The same features, viewed from a different angle, might cluster differently.
- **"Navigating between realms" is rotating perspective**, not crossing borders. RepE steering vectors wouldn't move you from one region to another — they'd rotate the model's perspective on the same underlying representation.
- **Physicality is shared, not realm-specific.** The same physical grounding (symmetry, conservation, continuity) underpins all perspectives. Counting and geometry feel different because they highlight different aspects of the same physical reality, not because they draw from different physics.
- **Connections between realms are trivially explained.** Of course realms connect — they're the same space. The "bridges" in the earlier table aren't bridges between separate islands; they're the fact that you can see the same terrain from multiple vantage points.

**Implication for the three techniques**: The test shouldn't ask "are there distinct realm clusters?" (presupposing partition). It should ask "does rotating perspective produce measurably different activation patterns?" The prediction becomes:

> Steering vectors computed from different perspectives will be **non-orthogonal but non-identical** — correlated (same space) but distinguishable (different angles). The cosine similarity between "arithmetic perspective" and "spatial perspective" vectors should be moderate (0.3–0.7), not near-zero (separate spaces) or near-one (same direction).

This is a weaker but more honest prediction. It also aligns with the topological framework: the topoi in the grid aren't separate worlds, they're different coordinate charts on the same manifold.

---

## Appendix B: Beyond Math — The Natural Extension to Pure Logic

### Why Math Was the Testbed, Not the Goal

Math was chosen because falsification is cheap: check a proof, test a conjecture, verify a computation. But if reasoning is a continuous manifold with perspectival distinctions, then the initiative was never about mathematical reasoning specifically. Math is simply the perspective where measurement is cheapest.

The natural extension is **pure logic** — propositional, first-order, modal. Not because logic is "another domain to test," but because logic and math are *neighboring perspectives* on the same underlying reasoning manifold. They share the deepest physicality bond:

- **Logical consequence** ≈ **physical causality** (if A then B, in both worlds)
- **Logical consistency** ≈ **physical possibility** (a world without contradiction)
- **Proof** ≈ **demonstration** (the same constructive act, different notation)
- **Counterexample** ≈ **experiment** (falsification by construction)

Logic is the perspective closest to math on the manifold. If the perspective model is correct, the "logic perspective" and the "math perspective" should:
- Have high cosine similarity (they share physicality grounding)
- Be distinguishable (different formal apparatus: syntax/rules vs quantity/structure)
- Show smooth transition (no gap — problems mixing logic and math should activate both perspectives simultaneously)

### What Extending to Logic Buys

| Advantage | Why it matters |
|---|---|
| **Independent falsification** | Proof check is mechanical (Z3, Lean, Coq). Same cheap-measurement property as math. |
| **Different perspective on the same reasoning** | If RepE steering vectors for logic problems differ from math problems *in the predicted way* (moderate similarity, not identical), that's evidence for the perspectival model. If they're identical, logic and math are the same perspective — still informative. |
| **Existing infrastructure** | `deepInference` already has Z3 + FOL engine. `verifiableReasoningLab` has step decomposition with contract verification. `stepDecomposer` has Hoare-style pre/post conditions. All of these work on logic, not just arithmetic. |
| **Harder problems become accessible** | Math conjectures are hard to evaluate beyond prime testing. Logical theorems can be verified mechanically for arbitrarily complex proofs. This opens up a richer test suite. |
| **Bridge to other perspectives** | Logic is the *hub* perspective. It connects math → logic → philosophy → law → computer science. If we can map the math→logic transition, we can predict the logic→philosophy transition by the same framework. |

### Proposed Extensions to the Test Plan

The Phase 1 (RepE) test from Appendix A should be extended from 4 perspectives to 6:

```
Original 4:
  - Arithmetic (recall primes vs prove primality)
  - Spatial (recall distances vs deduce from geometry)
  - Temporal (recall dates vs deduce from cause/effect)
  - Linguistic (recall definitions vs infer meaning from context)

Added 2:
  - Propositional logic (recall truth tables vs prove validity via natural deduction)
  - First-order logic (recall quantifier rules vs prove entailment via resolution)
```

The **key prediction**: the logic perspective vectors should have *higher* cosine similarity with arithmetic vectors than with spatial or temporal vectors. This is because logic and arithmetic share the strongest physicality bond (discrete, symbolic, constructive). If the similarity ordering is:

```
sim(arithmetic, logic) > sim(arithmetic, spatial) > sim(arithmetic, linguistic)
```

that's strong evidence for the perspectival model. The manifold has geometry — some perspectives are closer neighbors than others.

### Connection to Existing Repos

| Repo | Logic extension |
|---|---|
| `deepInference` | Already has Z3 + FOL engine. Was limited by OOM at scale. Logic perspective steering could *reduce* the search space by keeping the model in the logic perspective rather than drifting to recall. |
| `verifiableReasoningLab` | Step decomposition with pre/post conditions is essentially Hoare logic. The verification tools are already logic tools. Adding perspective steering would make the decomposition more reliable. |
| `stepDecomposer` / `stepDecomposer_v3` | PAL fallback does 87.5% of the work. If the model stayed in the "logic perspective" during decomposition instead of drifting to recall, the structured decomposition might actually outperform PAL. |
| `discoveryLoop` | Conjecture generation + counterexample search is already a logical process (hypothesis → falsification). Explicit logic-perspective steering could improve both phases. |
| `inferenceHarness` | The 14 benchmarks include logical reasoning benches. Adding perspective-steered runs would provide a direct comparison. |

### The Long Arc

Math → Logic → what? If the perspectival model holds, the natural progression is:

```
Math (testbed)
  └→ Logic (nearest neighbor, shares physicality)
      └→ Causal reasoning (logic + temporal physicality)
          └→ Scientific reasoning (logic + empirical observation)
              └→ Legal reasoning (logic + social physicality)
                  └→ Ethical reasoning (logic + cooperation physicality)
```

Each step adds a layer of physicality grounding. Each should be progressively harder to steer (less physical anchor, more diffuse perspective). The cosine similarities between perspective vectors should decrease along this chain — providing a measurable "perspectival distance" on the reasoning manifold.

The prediction: **the model's ability to sustain reasoning without recall should degrade along this chain in proportion to the distance from direct physical grounding.** Math and logic are easy (strong physicality). Ethics is hard (weak physicality). This is testable, falsifiable, and connects the mechanistic interpretability work to a theory of *why* some reasoning is harder than others.

---

*Next: See [WHY_CODING_FIRST.md](./WHY_CODING_FIRST.md) for the practical direction — coding as the high-ROI target for studying pure inferencing.*
