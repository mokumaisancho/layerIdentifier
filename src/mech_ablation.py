"""Mechanistic ablation: what is Gemma L6 reading?

Tests three hypotheses for what feeds the L6 probe signal:
  (A) Embedding-layer direct: L6 reads features that bypass L0-L5 entirely
      (residual stream carries embeddings forward).
  (B) Distributed across L0-L5: each layer contributes a small piece; ablating
      any single layer doesn't break the probe.
  (C) Residual-stream magnitude: the probe is just reading ||residual||,
      which is boringly correlated with confidence.

Tests:
  1. Identity-ablate layers 0-5 one at a time → measure L6 probe AUROC drop
  2. Identity-ablate ALL of layers 0-5 simultaneously → measure drop
  3. Replace L6 input with ||residual_5|| * sign-pattern (controls for magnitude)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/apple/Downloads/Py/layerIdentifier')

from src.model_loader import load_model, quick_summary
from src.probe_dataset import build_probe_dataset, is_correct
from src.layer_capture import install_layer_captures, restore_layers, resolve_layers
from src.entropy_pipeline import generate_with_full_capture
from src.features import layer_norm_features


class IdentityLayer:
    """Replace a layer's computation with input passthrough.
    Preserves tuple structure by calling original first."""
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


def run_captures(lm, examples, target_layer_for_probe=6):
    """Run captures on `examples`, return per-problem L6 features + correctness."""
    results = []
    for i, ex in enumerate(examples, 1):
        np.random.seed(ex.seed)
        prompt = wrap_prompt(lm, ex.prompt)
        trace = generate_with_full_capture(
            lm.model, lm.tokenizer, prompt,
            max_tokens=80, temperature=0.1,
            layer_indices=None,
            stop_sequences=["<turn|>", "<|end|>"],
        )
        correct = is_correct(trace.text, ex.ground_truth)
        l6_norms = trace.per_layer_norms.get(target_layer_for_probe, [])
        feats = layer_norm_features(l6_norms, target_layer_for_probe)
        results.append({
            "problem_id": ex.problem_id,
            "seed": ex.seed,
            "correct": correct,
            "l6_norms": l6_norms,
            "l6_features": feats,
        })
        if i % 5 == 0:
            print(f"  [{i}/{len(examples)}]", flush=True)
    return results


def measure_l6_signal(results, feature_name="convergence_slope"):
    """Return AUROC of L6 feature for predicting correctness."""
    scores = [r["l6_features"].get(f"L6_{feature_name}", 0.0) for r in results]
    labels = [1 if r["correct"] else 0 for r in results]
    return auroc(scores, labels)


def main():
    print("[mech] loading model")
    lm = load_model('/Volumes/BUF_2T_02/models/gemma-4-E4B-it-MLX-8bit')
    print(f"[mech] {quick_summary(lm)}")

    layers = resolve_layers(lm.model)
    print(f"[mech] {len(layers)} layers")

    # Use 30 examples for speed
    examples = build_probe_dataset(seeds=(7, 42))[:30]  # 30 problems
    print(f"[mech] {len(examples)} problems")

    out_dir = Path('/Users/apple/Downloads/Py/layerIdentifier/results/mechanistic_L6')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # === BASELINE ===
    print("\n=== BASELINE ===")
    t0 = time.perf_counter()
    baseline = run_captures(lm, examples)
    baseline_elapsed = time.perf_counter() - t0
    baseline_auroc = measure_l6_signal(baseline)
    print(f"  L6 convergence_slope AUROC: {baseline_auroc:.3f}  ({baseline_elapsed:.1f}s)")
    all_results["baseline"] = {
        "auroc_l6_convergence_slope": baseline_auroc,
        "elapsed_s": baseline_elapsed,
        "n_correct": sum(1 for r in baseline if r["correct"]),
        "n_problems": len(baseline),
    }

    # === SINGLE-LAYER ABLATIONS (L0..L5) ===
    print("\n=== SINGLE-LAYER IDENTITY ABLATIONS (L0..L5) ===")
    for abl_idx in range(6):
        original = layers[abl_idx]
        layers[abl_idx] = IdentityLayer(original)
        try:
            t0 = time.perf_counter()
            res = run_captures(lm, examples)
            elapsed = time.perf_counter() - t0
            auroc = measure_l6_signal(res)
            n_correct = sum(1 for r in res if r["correct"])
            delta_correct = n_correct - all_results["baseline"]["n_correct"]
            delta_auroc = auroc - baseline_auroc
            print(f"  Ablate L{abl_idx}: AUROC={auroc:.3f} (Δ={delta_auroc:+.3f})  "
                  f"correct={n_correct}/{len(res)} (Δ={delta_correct:+d})  ({elapsed:.1f}s)")
            all_results[f"ablate_L{abl_idx}"] = {
                "auroc_l6_convergence_slope": auroc,
                "delta_auroc_vs_baseline": delta_auroc,
                "n_correct": n_correct,
                "delta_correct_vs_baseline": delta_correct,
                "elapsed_s": elapsed,
            }
        finally:
            layers[abl_idx] = original

    # === ALL-LAYERS-0-5 ABLATION ===
    print("\n=== ALL LAYERS L0..L5 IDENTITY ABLATION ===")
    originals_05 = [layers[i] for i in range(6)]
    for i in range(6):
        layers[i] = IdentityLayer(originals_05[i])
    try:
        t0 = time.perf_counter()
        res = run_captures(lm, examples)
        elapsed = time.perf_counter() - t0
        auroc = measure_l6_signal(res)
        n_correct = sum(1 for r in res if r["correct"])
        delta_correct = n_correct - all_results["baseline"]["n_correct"]
        delta_auroc = auroc - baseline_auroc
        print(f"  Ablate L0-L5: AUROC={auroc:.3f} (Δ={delta_auroc:+.3f})  "
              f"correct={n_correct}/{len(res)} (Δ={delta_correct:+d})  ({elapsed:.1f}s)")
        all_results["ablate_L0_L5_all"] = {
            "auroc_l6_convergence_slope": auroc,
            "delta_auroc_vs_baseline": delta_auroc,
            "n_correct": n_correct,
            "delta_correct_vs_baseline": delta_correct,
            "elapsed_s": elapsed,
        }
    finally:
        for i, orig in enumerate(originals_05):
            layers[i] = orig

    # === VERDICT ===
    print("\n=== VERDICT ===")
    print(f"Baseline L6 AUROC: {baseline_auroc:.3f}")
    print(f"\nSingle-layer ablation impact on L6 AUROC:")
    for k in [f"ablate_L{i}" for i in range(6)]:
        d = all_results[k]["delta_auroc_vs_baseline"]
        marker = "*** BREAKS" if d < -0.10 else ("** weakens" if d < -0.03 else "OK")
        print(f"  {k}: Δ={d:+.3f}  {marker}")

    print(f"\nAll L0-L5 ablation: Δ={all_results['ablate_L0_L5_all']['delta_auroc_vs_baseline']:+.3f}")

    # Classify hypothesis
    single_breaks = [all_results[f"ablate_L{i}"]["delta_auroc_vs_baseline"] < -0.10 for i in range(6)]
    all_breaks = all_results["ablate_L0_L5_all"]["delta_auroc_vs_baseline"] < -0.10
    any_single_breaks = any(single_breaks)

    print("\n=== HYPOTHESIS VERDICT ===")
    if all_breaks:
        print("L6 probe DEPENDS on L0-L5 computation — refutes pure embedding-direct hypothesis (A)")
        if any_single_breaks:
            print("Single-layer ablation also breaks — suggests a CRITICAL layer exists")
            for i, broke in enumerate(single_breaks):
                if broke:
                    print(f"  → L{i} is critical for L6 probe")
        else:
            print("Single-layer ablations don't break — DISTRIBUTED signal (Hypothesis B confirmed)")
    else:
        print("L6 probe SURVIVES even L0-L5 ablation — strong evidence for embedding-direct (A) or magnitude (C)")

    out_path = out_dir / "ablation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[mech] wrote {out_path}")


if __name__ == "__main__":
    main()
