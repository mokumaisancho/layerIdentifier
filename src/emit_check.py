"""Emit check — does the model emit entropy at all?

Cheap probe:
  * Load model.
  * Generate on 5 problems.
  * Capture per-token entropy vec.
  * Pass if mean entropy variance > epsilon AND n_tokens > 10 per problem.
  * Report TPS.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model_loader import load_model, quick_summary
from .probe_dataset import emit_check_subset, is_correct
from .entropy_pipeline import generate_with_full_capture


EMIT_VARIANCE_EPS = 1e-3
EMIT_MIN_TOKENS = 10
EMIT_MIN_TPS = 5.0


def run_emit_check(model_path: str, out_dir: Path,
                    max_tokens: int = 256, temperature: float = 0.1) -> dict:
    """Run emit-check, write results to out_dir/emit_check.json, return summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[emit-check] loading {model_path}")
    lm = load_model(model_path)
    print(f"[emit-check] {quick_summary(lm)}")

    examples = emit_check_subset(k=5)
    print(f"[emit-check] {len(examples)} problems")

    captures = []
    for ex in examples:
        # Set seed via mlx if needed; temperature already in sampler
        np.random.seed(ex.seed)
        print(f"  {ex.problem_id} seed={ex.seed} ...", end=" ", flush=True)
        trace = generate_with_full_capture(
            lm.model, lm.tokenizer, ex.prompt,
            max_tokens=max_tokens, temperature=temperature,
            layer_indices=None,  # all layers
        )
        correct = is_correct(trace.text, ex.ground_truth)
        captures.append({
            "problem_id": ex.problem_id,
            "seed": ex.seed,
            "ground_truth": ex.ground_truth,
            "model_answer": trace.text.strip().splitlines()[-1][:80] if trace.text else "",
            "final_correct": correct,
            "n_tokens": trace.n_tokens,
            "tps": trace.tps,
            "elapsed_s": trace.elapsed_s,
            "entropy_vec": trace.entropy_vec.tolist(),
            "entropy_features": trace.entropy_features,
            "per_layer_norms": {str(k): v for k, v in trace.per_layer_norms.items()},
        })
        print(f"tokens={trace.n_tokens}  tps={trace.tps:.1f}  correct={correct}")

    # Aggregate emit verdict
    mean_entropy_var = float(np.mean([
        np.var(c["entropy_vec"]) if c["entropy_vec"] else 0.0
        for c in captures
    ]))
    min_tokens = min((c["n_tokens"] for c in captures), default=0)
    mean_tps = float(np.mean([c["tps"] for c in captures]))

    verdict = {
        "model_path": model_path,
        "n_layers": lm.n_layers,
        "arch": lm.arch,
        "n_problems": len(captures),
        "mean_entropy_variance": mean_entropy_var,
        "min_tokens_generated": min_tokens,
        "mean_tps": mean_tps,
        "emits_entropy": mean_entropy_var > EMIT_VARIANCE_EPS,
        "tokens_usable": min_tokens >= EMIT_MIN_TOKENS,
        "tps_usable": mean_tps >= EMIT_MIN_TPS,
        "captures": captures,
    }

    out = out_dir / "emit_check.json"
    out.write_text(json.dumps(verdict, indent=2))
    print(f"[emit-check] wrote {out}")
    print(f"[emit-check] emits_entropy={verdict['emits_entropy']}  "
          f"tokens_usable={verdict['tokens_usable']}  "
          f"tps_usable={verdict['tps_usable']}  "
          f"mean_tps={mean_tps:.1f}")
    return verdict
