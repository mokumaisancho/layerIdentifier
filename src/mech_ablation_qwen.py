"""Qwen replication of the writer/modulator/reader pattern.

Gemma finding (N=30, underpowered but suggestive):
  - L6 is the reader (AUROC 0.70 conv_slope)
  - L4 ablation KILLS L6 signal → L4 is the writer
  - L5 ablation ENHANCES L6 signal (0.70 → 0.99) → L5 is the modulator/noise

Qwen question: does the same pattern hold for Qwen L15 (rel 0.484)?
  - L15 is the reader
  - What is L13 / L14 doing? writer? modulator? neither?

This script tests:
  - baseline (no ablation)
  - ablate_L13, ablate_L14 (immediate predecessors)
  - ablate_L12 (control)
  - ablate_L0_L14_all (everything below L15)

NOTE: Copied from mech_ablation.py with target_layer=15 and Qwen model path.
Original mech_ablation.py is left untouched for Gemma N=150 run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/apple/Downloads/Py/layerIdentifier')

from src.model_loader import load_model, quick_summary
from src.probe_dataset import build_probe_dataset, is_correct
from src.layer_capture import resolve_layers
from src.entropy_pipeline import generate_with_full_capture
from src.features import layer_norm_features


class IdentityLayer:
    """Legacy: calls original.__call__ then discards result (wastes compute).
    Kept for backwards comparison."""
    def __init__(self, original):
        self.original = original
    def __getattr__(self, name):
        return getattr(self.original, name)
    def __call__(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        x = args[0]
        if isinstance(out, (list, tuple)):
            return (x,) + tuple(out[1:])
        return x


class IdentityLayerFast:
    """Optimized: skips original.__call__ entirely.

    mlx-lm contract (qwen3_5, gemma3, etc.): decoder layers return just
    hidden_states. The KV-cache is mutated in-place on the cache object
    passed in — no tuple wrapping needed.

    Skipping compute means:
      - The ablated layer contributes nothing to the residual stream (correct)
      - The cache for that layer is never populated (consistent — we don't
        use it because we also skip compute on subsequent steps)

    Returns args[0] (input hidden_states) unchanged.
    """
    def __init__(self, original):
        # Avoid triggering __setattr__ override during init
        object.__setattr__(self, '_original', original)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __setattr__(self, name, value):
        # Forward attribute writes (e.g., MLX layer.weight = ...) to original
        setattr(self._original, name, value)

    def __call__(self, *args, **kwargs):
        return args[0]


def auroc(scores, labels):
    s = np.array(scores); l = np.array(labels)
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    rs = np.sum([(neg < x).sum() + 0.5 * (neg == x).sum() for x in pos])
    a = rs / (len(pos) * len(neg))
    return max(a, 1 - a)


def wrap_prompt(lm, prompt):
    msgs = [{"role": "user", "content": prompt}]
    return lm.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def run_captures(lm, examples, target_layer=15, feature_name="n_spikes"):
    """Run captures on `examples`, return per-problem target-layer features + correctness.

    Optimizations vs original:
      - max_tokens=60 (was 80) — most GSM8K answers fit in <50 tokens
      - layer_indices=[target_layer] (was None=capture-all) — skip 31 other layer captures
    """
    results = []
    for i, ex in enumerate(examples, 1):
        np.random.seed(ex.seed)
        prompt = wrap_prompt(lm, ex.prompt)
        trace = generate_with_full_capture(
            lm.model, lm.tokenizer, prompt,
            max_tokens=60, temperature=0.1,
            layer_indices=[target_layer],
            stop_sequences=["<|im_end|>"],
        )
        correct = is_correct(trace.text, ex.ground_truth)
        norms = trace.per_layer_norms.get(target_layer, [])
        feats = layer_norm_features(norms, target_layer)
        results.append({
            "problem_id": ex.problem_id,
            "seed": ex.seed,
            "correct": correct,
            f"L{target_layer}_norms": norms,
            f"L{target_layer}_features": feats,
        })
        if i % 5 == 0:
            print(f"  [{i}/{len(examples)}]", flush=True)
    return results


def measure_signal(results, target_layer=15, feature_name="n_spikes"):
    """Return AUROC of target-layer feature for predicting correctness."""
    scores = [r[f"L{target_layer}_features"].get(f"L{target_layer}_{feature_name}", 0.0) for r in results]
    labels = [1 if r["correct"] else 0 for r in results]
    return auroc(scores, labels)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--target-layer", type=int, default=15)
    ap.add_argument("--feature", default="n_spikes")
    ap.add_argument("--model", default="/Volumes/BUF_2T_02/QwenMLB/models/Qwen3.5-4B-MLX-4bit")
    ap.add_argument(
        "--conditions",
        default="baseline,L12,L13,L14,L0_L14_all",
        help="comma-separated subset of conditions",
    )
    ap.add_argument("--out", default="/Users/apple/Downloads/Py/layerIdentifier/results/mechanistic_L15_qwen")
    args = ap.parse_args()

    print("[mech-qwen] loading model")
    lm = load_model(args.model)
    print(f"[mech-qwen] {quick_summary(lm)}")

    layers = resolve_layers(lm.model)
    print(f"[mech-qwen] {len(layers)} layers, target=L{args.target_layer}, feature={args.feature}")

    examples = build_probe_dataset(seeds=(7, 42, 123))[:args.n]
    print(f"[mech-qwen] {len(examples)} problems")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = [c.strip() for c in args.conditions.split(",")]
    all_results = {}
    baseline_auroc = None

    # === BASELINE ===
    if "baseline" in conds:
        print("\n=== BASELINE ===")
        t0 = time.perf_counter()
        baseline = run_captures(lm, examples, args.target_layer, args.feature)
        elapsed = time.perf_counter() - t0
        baseline_auroc = measure_signal(baseline, args.target_layer, args.feature)
        print(f"  L{args.target_layer} {args.feature} AUROC: {baseline_auroc:.3f}  ({elapsed:.1f}s)")
        all_results["baseline"] = {
            f"auroc_L{args.target_layer}_{args.feature}": baseline_auroc,
            "elapsed_s": elapsed,
            "n_correct": sum(1 for r in baseline if r["correct"]),
            "n_problems": len(baseline),
        }

    # === SINGLE-LAYER ABLATIONS ===
    for cond in conds:
        if cond == "baseline" or cond.startswith("L0_"):
            continue
        # Parse "L12" -> 12
        try:
            abl_idx = int(cond.lstrip("L"))
        except ValueError:
            print(f"  Skipping unknown condition: {cond}")
            continue
        if abl_idx >= args.target_layer:
            print(f"  Skipping L{abl_idx} (>= target L{args.target_layer})")
            continue
        original = layers[abl_idx]
        layers[abl_idx] = IdentityLayer(original)
        try:
            print(f"\n=== ABLATE L{abl_idx} ===")
            t0 = time.perf_counter()
            res = run_captures(lm, examples, args.target_layer, args.feature)
            elapsed = time.perf_counter() - t0
            au = measure_signal(res, args.target_layer, args.feature)
            n_correct = sum(1 for r in res if r["correct"])
            delta_correct = n_correct - all_results["baseline"]["n_correct"]
            delta_auroc = au - baseline_auroc
            marker = "*** BREAKS" if delta_auroc < -0.10 else (
                "** weakens" if delta_auroc < -0.03 else (
                    "++ ENHANCES" if delta_auroc > 0.10 else "OK"))
            print(f"  L{args.target_layer} AUROC={au:.3f} (Δ={delta_auroc:+.3f})  "
                  f"correct={n_correct}/{len(res)} (Δ={delta_correct:+d})  {marker}  ({elapsed:.1f}s)")
            all_results[f"ablate_L{abl_idx}"] = {
                f"auroc_L{args.target_layer}_{args.feature}": au,
                "delta_auroc_vs_baseline": delta_auroc,
                "n_correct": n_correct,
                "delta_correct_vs_baseline": delta_correct,
                "elapsed_s": elapsed,
            }
        finally:
            layers[abl_idx] = original

    # === ALL-LAYERS BELOW TARGET ===
    if "L0_L14_all" in conds and args.target_layer == 15:
        print("\n=== ABLATE L0..L14 ALL ===")
        originals = [layers[i] for i in range(15)]
        for i in range(15):
            layers[i] = IdentityLayer(originals[i])
        try:
            t0 = time.perf_counter()
            res = run_captures(lm, examples, args.target_layer, args.feature)
            elapsed = time.perf_counter() - t0
            au = measure_signal(res, args.target_layer, args.feature)
            n_correct = sum(1 for r in res if r["correct"])
            delta_correct = n_correct - all_results["baseline"]["n_correct"]
            delta_auroc = au - baseline_auroc
            print(f"  L{args.target_layer} AUROC={au:.3f} (Δ={delta_auroc:+.3f})  "
                  f"correct={n_correct}/{len(res)} (Δ={delta_correct:+d})  ({elapsed:.1f}s)")
            all_results["ablate_L0_L14_all"] = {
                f"auroc_L{args.target_layer}_{args.feature}": au,
                "delta_auroc_vs_baseline": delta_auroc,
                "n_correct": n_correct,
                "delta_correct_vs_baseline": delta_correct,
                "elapsed_s": elapsed,
            }
        finally:
            for i, orig in enumerate(originals):
                layers[i] = orig

    # === VERDICT ===
    print("\n=== VERDICT ===")
    print(f"Baseline L{args.target_layer} {args.feature} AUROC: {baseline_auroc:.3f}")
    print(f"\nSingle-layer ablation impact:")
    for k in sorted([k for k in all_results if k.startswith("ablate_L") and not k.endswith("_all")]):
        d = all_results[k]["delta_auroc_vs_baseline"]
        marker = "*** BREAKS" if d < -0.10 else ("** weakens" if d < -0.03 else ("++ ENHANCES" if d > 0.10 else "OK"))
        print(f"  {k}: Δ={d:+.3f}  {marker}")
    if "ablate_L0_L14_all" in all_results:
        d = all_results["ablate_L0_L14_all"]["delta_auroc_vs_baseline"]
        print(f"  ablate_L0_L14_all: Δ={d:+.3f}  {'BREAKS' if d < -0.10 else 'OK'}")

    # Gemma analogy
    print("\n=== GEMMA ANALOGY ===")
    print("Gemma pattern (N=30, retracted):")
    print("  L4 ablation: -0.13 (writer)")
    print("  L5 ablation: +0.29 (modulator/noise)")
    print("  L6 is reader")
    print(f"\nQwen L15 (this run):")
    if "ablate_L13" in all_results:
        d = all_results["ablate_L13"]["delta_auroc_vs_baseline"]
        print(f"  L13 ablation: {d:+.3f}  ({'writer analog' if d < -0.10 else 'not writer'})")
    if "ablate_L14" in all_results:
        d = all_results["ablate_L14"]["delta_auroc_vs_baseline"]
        print(f"  L14 ablation: {d:+.3f}  ({'modulator analog' if d > 0.10 else 'not modulator'})")
    if "ablate_L12" in all_results:
        d = all_results["ablate_L12"]["delta_auroc_vs_baseline"]
        print(f"  L12 ablation (control): {d:+.3f}")

    (out_dir / "ablation_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
