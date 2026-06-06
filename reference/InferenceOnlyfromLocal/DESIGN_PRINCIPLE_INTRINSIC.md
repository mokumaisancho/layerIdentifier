# Design Principle: Intrinsic Meta-Cognitive Position

**Date**: 2026-06-02

---

## The Principle

The model's positional awareness within a hierarchical problem must come from **dynamic, intrinsic understanding of the issue** — not from externally imposed hierarchical structure.

- ❌ Impose hierarchy: "here's the problem tree, attend at level N"
- ✅ Emerge hierarchy: the model understands the issue deeply enough to naturally know where it is, what's local, what's global, what's missing

## Why This Matters

External scaffolding (prompt engineering, structured decomposition, tree-of-thought) works for simple problems but fails when the problem structure is genuinely novel — which is exactly the case where pure inferencing matters most. You can't scaffold what you don't yet understand.

Intrinsic meta-cognition doesn't have this limitation. The model that *understands* the problem generates its own scaffolding from comprehension, not from external templates.

## Evidence the Model Already Does This

| Finding | Source | What it shows |
|---|---|---|
| AUC=1.0 correctness at L3 | InferenceLLM2 | Model intrinsically knows when wrong — no external signal |
| Correctness peaks at L16 | cot-correctness-probe | Model commits before output — internal decision, not prompted |
| Entropy predicts correctness | parallel_l4_tot | Model's own uncertainty is diagnostic — intrinsic quality signal |

These are emergent meta-cognitive signals. Nobody told the model to compute them. They arise from the model's own processing.

## What This Means for Design

1. **Don't impose structure.** The harness should not tell the model "this is a 3-level problem, you are at level 2." The model should discover that through understanding.

2. **Read off intrinsic signals.** The harness should OBSERVE the model's emergent positional awareness (via hidden states, entropy, activation patterns) rather than IMPOSE a structure.

3. **Create conditions, not instructions.** The role of the harness is to create conditions where intrinsic understanding can develop (proper context, time to process, relevant retrieval) — not to dictate the processing structure.

4. **The hierarchy is discovered, not given.** When the model truly understands an issue, the local/global/external structure becomes apparent to it naturally. That's what "understanding" IS — grasping the structure of something.

## How Does the Model Know Local vs Global?

Two approaches:

**1. Dependency tree** (external) — analyze the problem structure externally, feed the model positional information. Works but brittle for novel problems where the structure isn't known in advance.

**2. Intrinsic signal** — the model already processes at different dependency depths. The layer stack IS the dependency resolution pipeline. Layer 1 sees immediate token neighbors. Layer 16 sees the full context's dependency structure. The model doesn't need to build a tree — it already processes through one.

The hypothesis: the model already knows whether it's working on a local or global issue. The signal is encoded in activation dynamics. We just haven't read it as a positional signal.

### Readable Intrinsic Signals

| Signal | Local problem | Global problem | How to read |
|---|---|---|---|
| **Attention entropy** at target layer | Low (focused on few tokens) | High (attending broadly across context) | Compute entropy of attention weights per head |
| **Activation shift** L15→L16 | Small (already resolved) | Large (model is "deciding" here) | Cosine distance between consecutive layer activations |
| **L3 correctness signal** | Strong early (local is easy to verify) | Weak early (can't verify global yet) | InferenceLLM2 probe output at early layers |
| **Cross-layer consistency** | Stabilizes by mid-layers | Keeps shifting through deep layers | Running variance of activations across layer range |

### The Test

```
1. Prepare 50 clearly-local problems (fix this bug, compute this value)
   and 50 clearly-global problems (design this system, refactor this module)

2. Capture per-layer attention entropy + activation shifts for all 100

3. Check separation:
   - Can a linear probe distinguish local from global at any layer?
   - Which signal separates best?
   - At which layer does the separation peak?

4. If separation exists: the model intrinsically knows its position.
   The harness reads this signal and uses it — no external dependency tree needed.
```

This test costs the same as the RepE experiment from SIGNIFICANCE_THREE_APPROACHES.md
but answers a different question: not "recall vs reasoning" but "local vs global position."

### Why Both Approaches May Be Needed

Intrinsic signals tell you where the model THINKS it is. Dependency trees tell you where it ACTUALLY is. When they disagree, the model is confused about scope — that's a failure mode worth detecting.

The harness could:
1. Read intrinsic signal → model's self-assessed position
2. Compute dependency tree → actual position (when possible)
3. Flag disagreements → the model is treating a global problem as local (or vice versa)

## Connection to Other Documents

- **SIGNIFICANCE_THREE_APPROACHES.md**: RepE, SAE, ROME are tools to READ the model's intrinsic positional signals — they don't impose structure, they observe it
- **WHY_CODING_FIRST.md**: Coding is the testbed because the model's intrinsic understanding of code structure is observable (does the generated code work?) — the feedback loop validates intrinsic comprehension without imposing structure
- **Realm hypothesis**: Realms are not external categories imposed on the model. They're perspectives the model naturally adopts based on its intrinsic understanding of the problem type
- **Sheaf theory**: The local-to-global gluing is not imposed — it's what the model discovers when it understands how local pieces relate to the global structure
