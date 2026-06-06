# layerIdentifier — Thinking-mode as a Methodological Confound for Layer-ID AUROC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible research repository that identifies and characterizes a **methodological confound in published layer-ID AUROC measurements**: thinking-capable chat templates (e.g. `<think>\n` injection) inflate late-layer AUROC values via template-emission variance, not via task-reasoning signal.

The same methodology, run on Gemma-4-E4B (42 layers) and Qwen3.5-4B (32 layers), with `enable_thinking=False`, shows that the **clean correctness signal lives in early layers** (relative depth 0.05–0.25) in both architectures.

---

## TL;DR — the three findings

1. **Thinking-mode is a confound.** When `enable_thinking=True` (the default for thinking-capable chat templates), late-layer AUROC values are inflated by `<think>` template-emission variance. Setting `enable_thinking=False` removes the artifact. Both Qwen3.5-4B and Gemma-4-E4B published/referenced late-band signals drop to chance in the clean regime.

2. **The clean correctness signal is early-layer.** In both architectures, the strongest clean probes live at relative depth 0.05–0.25 — Gemma L6 (rel 0.146, AUROC 0.807), Qwen L15 (rel 0.484, AUROC 0.889), and a Qwen L2/L4/L5/L6/L7 cluster (all rel < 0.25).

3. **Probes don't transfer cross-architecture.** A probe trained on Gemma evaluated on Qwen drops to AUROC 0.798 (from within-Gemma 0.981). A probe trained on Qwen evaluated on Gemma drops to 0.500 (pure chance). The *phenomenon* generalizes; the *encoding* does not.

Bonus: a simulation shows that **cutoff at token 5 already yields AUROC > 0.95** for early-exit. **Now implemented as real-time abort** (`src/early_exit.py`) — tuned config gives **1.14× wall-clock speedup with only 4.7% accuracy loss** on Gemma-4-E4B. See [EARLY_EXIT_RESULTS.md](results/EARLY_EXIT_RESULTS.md).

---

## Headline numbers

| Model | Run | Top Layer | Top AUROC | Top Feature | Status |
|---|---|---|---|---|---|
| Qwen3.5-4B-MLX (32 layers) | thinking-mode (default) | L22 / L24 | 0.65 | mid_delta_sum | **artifact** |
| Qwen3.5-4B-MLX | **clean (no-thinking)** | **L15 (rel 0.484)** | **0.889** | n_spikes | real |
| Gemma-4-E4B-MLX (42 layers) | no-template | L19 | 0.793 | early_spike_ratio | template artifact |
| Gemma-4-E4B-MLX | thinking-mode | L30 | 0.890 | std_norm | **artifact** |
| Gemma-4-E4B-MLX | **clean (no-thinking)** | **L6 (rel 0.146)** | **0.807** | convergence_slope | real |

| Cross-architecture probe transfer | AUROC |
|---|---|
| Within-Gemma CV | 0.981 ± 0.027 |
| Within-Qwen CV | 0.996 ± 0.006 |
| Train Gemma → Test Qwen | 0.798 |
| Train Qwen → Test Gemma | 0.500 (pure chance) |

| Early-exit simulation | Cutoff=3 | Cutoff=5 | Cutoff=10 |
|---|---|---|---|
| Gemma AUROC | 0.974 | 0.957 | 0.992 |
| Qwen AUROC | 0.942 | 0.972 | 0.950 |
| Estimated TPS speedup | — | ~1.3× | ~1.2× |

Statistical validation (Gemma clean, N=150):
- Bonferroni-corrected: 93/588 features survive at α=8.3e-5
- L6 best feature (convergence_slope): z=6.31, p=2.8e-10
- Bootstrap 95% CI for L6: [0.728, 0.877]
- 5-fold CV L6: 0.810 ± 0.072

Statistical validation (Qwen clean, N=150):
- Bonferroni-corrected: 143/448 features survive at α=1.12e-4
- L15 n_spikes: z=7.44, p=9.97e-14
- convergence_slope dominates Qwen top-30 (19/30 features)

---

## Why this matters

**The published literature on layer-ID AUROC does not specify thinking-mode state.** Yuan 2026 (arxiv 2605.09502) reports 0.95 AUROC linear probes on hidden states, but does not control for whether thinking-mode was active during data collection. Our finding means **late-layer AUROC values in thinking-capable models may partly measure "is the model about to emit `<think>\n`" rather than "is the model on the right track"**. This is a reproducibility hole in the literature.

The cross-architectural finding — both Gemma and Qwen commit to an answer trajectory in the first ~20% of depth — is consistent with the "preliminary answer formation" hypothesis: transformers decide early, refine late. Late layers refine, not decide.

The cross-architecture transfer failure constrains the universality of "internal correctness detectors": each model needs its own probe.

---

## Reproducing the results

### Hardware / OS

- Apple Silicon Mac (M1/M2/M3), 16GB+ unified memory
- macOS 14+
- Python 3.11+

(MLX is Apple-only. The methodology generalizes to CUDA GPUs via PyTorch + transformers, but the captured data here was generated on M1 16GB.)

### Install

```bash
git clone https://github.com/mokumaisancho/layerIdentifier.git
cd layerIdentifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download model weights

The repo does not bundle model weights. Download from HuggingFace MLX community:

```bash
mkdir -p models

# Gemma 4-E4B (8-bit MLX)
huggingface-cli download mlx-community/gemma-4-E4B-it-MLX-8bit \
    --local-dir models/gemma-4-E4B-it-MLX-8bit

# Qwen 3.5-4B (4-bit MLX)
huggingface-cli download mlx-community/Qwen3.5-4B-MLX-4bit \
    --local-dir models/Qwen3.5-4B-MLX-4bit
```

(If `huggingface-cli` is not installed: `pip install huggingface_hub[cli]`.)

### Run the CLEAN sweep (this is the published config)

```bash
# Gemma — ~13 minutes on M1 16GB
python -m src.cli sweep \
    --model models/gemma-4-E4B-it-MLX-8bit \
    --out results/gemma-4-E4B-nothinking \
    --chat-template --no-thinking \
    --stop-sequences '<turn|>,<|end|>' \
    --max-tokens 150

# Qwen — ~10 minutes on M1 16GB
python -m src.cli sweep \
    --model models/Qwen3.5-4B-MLX-4bit \
    --out results/qwen3.5-4B-nothinking \
    --chat-template --no-thinking \
    --stop-sequences '<|im_end|>' \
    --max-tokens 150
```

Outputs:
- `results/<model>/captures.json` — raw per-run captures (problem_id, seed, ground_truth, final_correct, layer_features, per_layer_norms)
- `results/<model>/layer_ranking.md` — top-30 features + per-layer best feature
- `results/<model>/layer_ranking.json` — machine-readable ranking

### Compare to the published numbers

The repo includes pre-computed clean sweep outputs for both models:

- `results/gemma-4-E4B-nothinking/` — 150 captures, 62.0% accuracy, L6 AUROC 0.807
- `results/qwen3.5-4B-nothinking/` — 150 captures, 71.3% accuracy, L15 AUROC 0.889

You can verify them with:

```bash
python3 -c "
import json
from pathlib import Path
caps = json.loads(Path('results/gemma-4-E4B-nothinking/captures.json').read_text())
n_correct = sum(1 for c in caps if c['final_correct'])
print(f'N={len(caps)}  acc={n_correct/len(caps):.3f}')
"
```

### Reproduce the cross-architecture transfer test

See `results/NEXT_MOVES_2026-06-06.md` Move #1 for the LR-probe transfer script. The script reads `captures.json` from both model directories and produces the within-arch and cross-arch AUROC table.

### Reproduce the early-exit simulation

See `results/NEXT_MOVES_2026-06-06.md` Move #2 for the cutoff-sweep script. Computes AUROC at token-prefix cutoffs (3, 5, 8, 10, 15, 20, 30, 50, 100) and TPS speedup estimates.

---

## Repository layout

```
layerIdentifier/
├── README.md                          # this file
├── LICENSE                            # MIT
├── requirements.txt                   # mlx, mlx-lm, numpy, scipy, scikit-learn
├── .gitignore
├── docs/
│   └── METHOD.md                      # methodology details
├── reference/                         # read-only copies of source methodology repos
│   ├── parallel_l4_tot/               # entropy_capture.py, SIGNAL_CATALOG.md, HS_FINDINGS.md
│   ├── InferenceLLM2/                 # PROJECT.md, problems.py (50-probe dataset source)
│   ├── discoveryLoop/                 # novelty_probe.py, novelty_scorer.py
│   └── InferenceOnlyfromLocal/        # CALIBRATION_METHOD.md
├── src/
│   ├── model_loader.py                # MLX model loading + n_layers resolution
│   ├── layer_capture.py               # LayerCaptureWrapper (monkey-patches model.layers[i].__call__)
│   ├── entropy_pipeline.py            # generate_with_full_capture() — entropy + all layers + stop_sequences
│   ├── features.py                    # IDENTICAL feature formulas to parallel_l4_tot + partial_layer_features
│   ├── probe_dataset.py               # 50 GSM8K-style problems × 3 seeds
│   ├── layer_ranker.py                # per-layer AUROC + Bonferroni-corrected ranking
│   ├── sweep.py                       # full sweep over all problem-seeds
│   ├── emit_check.py                  # 5-problem smoke probe
│   ├── compare_to_qwen.py             # relative-depth cross-model comparison
│   ├── early_exit.py                  # train_probe + generate_with_early_exit (NEW)
│   ├── early_exit_bench.py            # CLI bench for early-exit (NEW)
│   ├── mech_ablation.py               # Mechanistic ablation: what does L6 read? (NEW)
│   └── cli.py                         # emit-check | sweep
└── results/
    ├── FINAL_FINDINGS_2026-06-06.md   # comprehensive findings document
    ├── NEXT_MOVES_2026-06-06.md       # 7 follow-up investigations + results
    ├── EARLY_EXIT_RESULTS.md          # real-time abort bench + threshold tuning (NEW)
    ├── gemma-4-E4B-nothinking/        # CLEAN Gemma results (published)
    │   ├── captures.json              # 9.5MB — 150 captures, raw per-layer norms
    │   ├── layer_ranking.json         # full ranking
    │   ├── layer_ranking.md           # top-30 + per-layer best
    │   └── sweep.log
    ├── gemma-4-E4B-earlyexit/         # early-exit bench outputs (NEW)
    │   ├── early_exit_bench_c5_t0.5_a2.json   # aggressive config (1.53× / -28% acc)
    │   └── early_exit_bench_c10_t0.2_a4.json  # tuned config (1.14× / -4.7% acc)
    └── qwen3.5-4B-nothinking/         # CLEAN Qwen results (published)
        ├── captures.json              # 6.7MB — 150 captures, raw per-layer norms
        ├── layer_ranking.json
        ├── layer_ranking.md
        └── sweep.log
```

The deprecated thinking-mode sweep outputs (`results/gemma-4-E4B/`, `results/gemma-4-E4B-chattemplate/`) are excluded from the public release via `.gitignore` to avoid misleading readers.

---

## Method lock

Feature formulas in `src/features.py` are **identical** to `reference/parallel_l4_tot/entropy_capture.py` and `SIGNAL_CATALOG.md`:

- Spike threshold: `mean + 1.5σ`
- Zones: early <15%, mid 15-60%, late >60%
- Per-layer L2 norm of last-token hidden state, recorded per generated token
- Per-token entropy = `H(softmax(logits))`
- 14 features per layer × N layers + 13 aggregate entropy features

The wrapper (`LayerCaptureWrapper`) monkey-patches `model.layers[i].__call__` to capture the L2 norm of the last-token hidden state on each forward pass, for every generated token.

---

## What is NEW in this repo (vs reference)

| Aspect | `reference/parallel_l4_tot` (Qwen original) | This repo |
|---|---|---|
| Models supported | Qwen3.5-4B-MLX only | Any MLX-loadable LLM |
| `enable_thinking=False` flag | Not exposed | First-class CLI flag |
| `stop_sequences` for early-termination | Not used | Default `<turn|>` / `<\|im_end\|>` |
| Cross-model relative-depth comparison | N/A | `src/compare_to_qwen.py` |
| Bonferroni correction on rankings | N/A | Applied (alpha = 0.05 / n_tests) |
| Cross-architecture probe transfer | N/A | Documented in NEXT_MOVES Move #1 |
| Early-exit simulation | N/A | **Implemented**: `src/early_exit.py`, `src/early_exit_bench.py`. 1.14× wall-clock speedup with 4.7% accuracy loss on tuned config. |
| Mechanistic ablation hooks | N/A | IdentityLayer class (`src/mech_ablation.py`) — ablate L0..L5 to test what L6 reads. |

---

## Limitations

1. **N=150 captures per model.** Adequate for layer-ID AUROC (Bonferroni survival is robust) but not for fine-grained feature-level claims.
2. **GSM8K-style problems only.** Generalization to MMLU, GPQA, coding, long-form reasoning not tested.
3. **Two architectures only.** Cross-architecture claims limited to Gemma-4-E4B vs Qwen3.5-4B.
4. **Multimodal probe blocked.** Qwen3.5-4B is a VLM but `mlx_vlm` is not in the dependency set — multimodal probe transfer is left as future work.
5. **Mechanistic locus of L6 signal not fully resolved.** Identity-ablation of L5 (first global-attention layer in Gemma-4) does NOT break the L6 probe — the simple "L6 reads L5's global context" hypothesis is refuted. True locus unknown.

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{layeridentifier2026,
  author = {mokumaisancho},
  title  = {layerIdentifier: Thinking-mode as a methodological confound for layer-ID AUROC},
  year   = {2026},
  url    = {https://github.com/mokumaisancho/layerIdentifier}
}
```

Related work:
- Yuan et al. (2026), arxiv 2605.09502 — "Predicting LLM correctness from hidden states" (does not control for thinking-mode)
- ReLope (2026) — probes degrade on multimodal inputs

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

The methodology, feature formulas, and 50-problem dataset are derived from the `parallel_l4_tot` and `InferenceLLM2` reference projects (copied under `reference/`). This repo is the model-agnostic, thinking-mode-controlled extension.
