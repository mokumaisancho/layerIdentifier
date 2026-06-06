# Attention Fusion Probe: Findings Report

**Date**: 2026-05-26
**Model**: Qwen3.5-4B-MLX-4bit (32 layers, 8 full-attention)
**Problems**: Q110, Q112, Q136 (3 benchmark failures from 47/50 run)
**Runs**: v1 (original), v2 (post bug-fix, same seeds), 18 paths total
**Data**: `results/` and `results_v2/` directories

---

## Hypothesis

> When the model produces wrong answers, it does so because at "fusion points"
> (where it integrates information across reasoning steps), the attention mechanism
> reads from wrong prior tokens in the KV cache. Detecting excessive cross-step
> attention would identify and enable rejection of wrong paths.

## Verdict: Inconclusive. Insufficient data to confirm or deny.

---

## Do we know enough to decide?

**No.** We have 18 paths from 3 problems. That is statistically meaningless.
You cannot establish whether a signal is an indicator or a tendency with n=6
wrong paths and n=12 correct paths.

What we CAN say:
- The direction (correct=higher cross-step) held in 5 of 6 comparisons
- One exception exists (v1 Q136 P1, wrong but cross-step=0.972)
- The ranges overlap: correct [0.260, 0.958], wrong [0.214, 0.972]

What we CANNOT say:
- Whether the tendency holds across all 50 benchmark problems
- Whether the tendency is stable across runs (it flipped between v1/v2)
- Whether per-head granularity reveals a stronger signal (we averaged everything)
- Whether the correlation is statistically significant (need Mann-Whitney U with n>>18)

**To determine significance we would need:**
- All 50 problems × 3 temperatures = 150 paths with attention capture
- Per-head granularity (not averaged across 4 KV heads × 8 layers)
- At least 2 runs for stability check
- ~12 hours of compute

---

## What prior research exists?

Research on using internal model states to detect errors in generated text
is active. The closest papers:

### Directly relevant: attention-based error/confidence detection

| Paper | Year | Venue | What they did | Relation to our work |
|---|---|---|---|---|
| **Shelmanov et al.** "A Head to Predict and a Head to Question" | 2025 | EMNLP | Trained auxiliary uncertainty quantification heads on LLM activations. Detects hallucinations from hidden states. | They use **activations**, not attention patterns. But same goal: detect wrong outputs from internals. |
| **Nguyen et al.** "Attention-Aware Intervention for CoT Logical Reasoning" | 2026 | EACL Findings | Intervenes on attention patterns during CoT reasoning to improve logical accuracy. | **Directly related.** They modify attention to improve reasoning. We measure attention to detect reasoning failure. Their work implies attention patterns DO carry causal information about reasoning quality. |
| **Huang et al.** "Confidence-Based Response Abstinence via Activation-Based Uncertainty" | 2025 | UncertaiNLP | Uses raw FFN activations (not logits) as confidence signal for RAG. Single layer (L16) sufficient. | They found activations more informative than token probabilities. Our entropy captures token probs; we didn't look at activations. |
| **Chen et al.** "How Does CoT Think? Mechanistic Interpretability of CoT" | 2026 | AAAI | Sparse autoencoders on CoT reasoning features. Found CoT induces modular internal structures. | Feature-level causal study of CoT. They swap CoT features between runs and measure impact. More granular than our approach. |
| **Ostmeier et al.** "Head Entropy" | 2026 | arXiv:2602.13699 | Per-head attention entropy as confidence signal. Found specific heads at specific layers carry confidence. | **Closest to our approach.** Key difference: they use per-head granularity, we averaged. Their per-head approach works; our averaged approach doesn't. |

### The gap in existing research

**No paper has studied cross-step attention as a selector signal.** The research
falls into two camps:

1. **Activation-based**: Use hidden states/FFN activations as confidence features
   (Shelmanov, Huang). Works well but requires training data.

2. **Attention-head-based**: Use per-head attention entropy/patterns as confidence
   features (Ostmeier, Nguyen). Works with per-head granularity.

Our approach — measuring how much attention crosses reasoning step boundaries —
is novel. But our execution (averaging across all heads and layers) destroyed the
per-head signal that Ostmeier showed is critical.

### What the research tells us about our approach

1. **The direction is right** — Nguyen 2026 shows attention patterns carry causal
   information about reasoning quality. Our finding (correct=more cross-step) is
   consistent with their intervention results.

2. **Our granularity was wrong** — Ostmeier 2026 showed individual heads carry
   the signal. We averaged 32 heads × 8 layers into one number, destroying the
   signal. This is likely why we see only a weak tendency.

3. **Activations might be better than attention** — Shelmanov 2025 and Huang 2025
   both found activations more informative than attention-based signals. Our
   pipeline already captures L10/L12 hidden state norms (entropy_capture.py),
   which is closer to the activation approach.

4. **Nobody has done large-scale cross-step attention analysis** — the closest is
   Nguyen's intervention work, but they modify attention rather than measuring it.
   Our question (can cross-step attention serve as a no-training-needed selector?)
   remains unanswered by existing literature.

---

## Is cross-step attention an indicator or a tendency?

**With n=18, we cannot tell.** The data is consistent with both interpretations:

- **If it's an indicator**: the 1 exception (v1 Q136 P1, wrong at 0.972) is noise.
  With more data, the separation would sharpen.

- **If it's a tendency**: the 5/6 agreement is just correlation with a confound
  (entropy). With more data, the overlap would stay large.

The difference matters:
- An **indicator** can be used alone as a selector feature
- A **tendency** can only be used as one input to a multi-feature classifier

Given that Ostmeier found per-head signals work, and Nguyen found causal effects,
the signal likely exists at per-head granularity. Our averaged metric may be
hiding a real indicator inside the noise.

---

## What is the root cause of wrong answers?

Looking at the actual generated text for each wrong path:

| Path | Wrong answer | What the text shows | Root cause |
|---|---|---|---|
| v1 Q110 P2 | 3.0 (GT=600) | Model debates what "####" means instead of computing 3×25×8 | Format fixation |
| v1 Q136 P1 | 2.0 (GT=40) | Model computes Jill's hours (2+1=3) and returns that | Task confusion |
| v1 Q136 P2 | 2.0 (GT=40) | Model appears to compute correctly but answer extraction fails | Extraction or truncation |
| v2 Q112 P1 | 3.0 (GT=4) | Model computes "40 - 36 = 4 minutes" correctly but parser gets 3 | Parser bug (BUG-007) |
| v2 Q112 P2 | 3.0 (GT=4) | No #### marker, generation truncates mid-calculation | Truncation |
| v2 Q136 P2 | 1.0 (GT=40) | "This seems contradictory" — model second-guesses itself | Self-doubt loop |

The root causes are **text-level failures** (format fixation, truncation,
task confusion, self-doubt), not attention-level failures. The attention pattern
reflects what the model was doing when the failure occurred but does not cause it.

However: Nguyen 2026 showed that modifying attention patterns CAN improve
reasoning accuracy. This means attention patterns have **causal influence** on
reasoning quality, not just correlation. Our finding (wrong=less cross-step)
is consistent with this — if cross-step attention is reduced, integration fails.
But the relationship is not deterministic: v1 Q136 P1 had very high cross-step
attention and still got the wrong answer because it was confused about the task.

---

## What alternatives were compared?

All candidate selector signals measured from the same 18 paths:

| Signal | What it measures | Mechanism | Separation | Already in pipeline? |
|---|---|---|---|---|
| avg cross-step attention | How much attention crosses step boundaries | Attention probe (new) | 0.433 | No |
| attention entropy | Spread of attention distribution | Attention probe (new) | 0.300 | No |
| max output entropy | Most uncertain token | Logprob analysis (existing) | 0.270 | Yes |
| entropy mean | Average token uncertainty | Logprob analysis (existing) | 0.220 | Yes |
| step diversity (text) | Count of reasoning step markers | Regex on text (new) | 0.181 | No |
| cross-step stability | Variance of cross-step across positions | Attention probe (new) | 0.068 | No |
| delta_std | Std of token-to-token entropy changes | Logprob analysis (existing) | 0.017 | Yes |

The "separation" metric (|mean_correct - mean_wrong| / pooled_std) is crude
with n=18. All separations are small. None reach statistical significance.

The comparison IS valid (apples-to-apples in the sense that all are candidate
features for the same binary classification task: correct vs wrong). But the
user is right that the mechanisms differ:
- Entropy signals measure **output confidence** (the model's uncertainty about
  what to say next)
- Attention signals measure **input selection** (where the model looks for
  information)

These are mechanistically different, which means they could be complementary.
The negative correlation (r=-0.442) between them means they carry partially
independent information. A multi-feature classifier combining both could
outperform either alone — but we lack the data to test this.

---

## What should happen next?

Three paths, ordered by expected ROI:

1. **Drop attention probe, improve existing entropy signals** — lowest effort,
   no new infrastructure. The pipeline already uses entropy; improving the
   classifier (better features, more training data) is the safe play.

2. **Per-head attention entropy** (Ostmeier approach) — medium effort, proven
   to work in literature. Requires modifying the probe to capture per-head
   data, then running all 150 paths. ~12 hours compute.

3. **Activation-based confidence** (Shelmanov/Huang approach) — highest
   potential but requires training data and auxiliary heads. More engineering
   but the literature shows it works best.

---

## Data files

- `results/q{110,112,136}_fusion_analysis.json` — v1 (18 paths)
- `results_v2/q{110,112,136}_fusion_analysis.json` — v2 (18 paths)
- Each path: 768 tokens × 8 layers × 4 KV heads = 24,608 attention snapshots
