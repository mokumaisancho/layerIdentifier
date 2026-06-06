"""Full layer-ID sweep — run all problems, capture all layers, rank.

Usage:
    python -m layerIdentifier.src.sweep \\
        --model /path/to/your/model.mlx \\
        --out results/your-model/ \\
        --max-tokens 150 \\
        --temperature 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .model_loader import load_model, quick_summary
from .probe_dataset import build_probe_dataset, is_correct
from .entropy_pipeline import generate_with_full_capture
from .features import all_layers_features, entropy_features
from .layer_ranker import rank_layers_from_captures, write_report, write_json


def run_sweep(model_path: str, out_dir: Path,
              max_tokens: int = 150, temperature: float = 0.1,
              limit: int | None = None,
              use_chat_template: bool = False,
              enable_thinking: bool = True,
              stop_sequences: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default stop sequences for Gemma-4 chat-template mode.
    if stop_sequences is None and use_chat_template:
        stop_sequences = ["<turn|>", "<|end|>"]

    print(f"[sweep] loading {model_path}")
    lm = load_model(model_path)
    print(f"[sweep] {quick_summary(lm)}")

    examples = build_probe_dataset(seeds=(7, 42, 123))
    if limit:
        examples = examples[:limit]
    print(f"[sweep] {len(examples)} problem-seed runs  "
          f"chat_template={use_chat_template} thinking={enable_thinking} "
          f"stops={stop_sequences}")

    def wrap_prompt(prompt: str) -> str:
        if not use_chat_template:
            return prompt
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not enable_thinking:
            kwargs["enable_thinking"] = False
        return lm.tokenizer.apply_chat_template(msgs, **kwargs)

    captures = []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples, 1):
        np.random.seed(ex.seed)
        print(f"[{i}/{len(examples)}] {ex.problem_id} seed={ex.seed} ...", end=" ", flush=True)
        try:
            trace = generate_with_full_capture(
                lm.model, lm.tokenizer, wrap_prompt(ex.prompt),
                max_tokens=max_tokens, temperature=temperature,
                layer_indices=None,  # ALL layers
                stop_sequences=stop_sequences,
            )
            correct = is_correct(trace.text, ex.ground_truth)
            layer_feats = all_layers_features(trace.per_layer_norms)
            captures.append({
                "problem_id": ex.problem_id,
                "seed": ex.seed,
                "ground_truth": ex.ground_truth,
                "model_answer_text": trace.text[:300],
                "final_correct": correct,
                "n_tokens": trace.n_tokens,
                "tps": trace.tps,
                "entropy_features": trace.entropy_features,
                "layer_features": layer_feats,
                "per_layer_norms": {str(k): v for k, v in trace.per_layer_norms.items()},
            })
            print(f"tok={trace.n_tokens} tps={trace.tps:.1f} correct={correct}")
        except Exception as e:
            print(f"ERROR: {e}")
            captures.append({
                "problem_id": ex.problem_id,
                "seed": ex.seed,
                "error": str(e),
                "final_correct": False,
                "layer_features": {},
            })

    elapsed = time.perf_counter() - t0
    n_correct = sum(1 for c in captures if c.get("final_correct"))

    # Save raw captures (large)
    cap_path = out_dir / "captures.json"
    cap_path.write_text(json.dumps(captures, indent=2))
    print(f"[sweep] wrote {cap_path} ({len(captures)} captures, {n_correct} correct)")

    # Rank
    rankings = rank_layers_from_captures(captures)
    write_report(rankings, out_dir / "layer_ranking.md",
                 model_label=Path(model_path).name,
                 n_layers=lm.n_layers, n_runs=len(captures))
    write_json(rankings, out_dir / "layer_ranking.json")
    print(f"[sweep] wrote layer_ranking.md / .json")

    return dict(
        n_layers=lm.n_layers,
        n_runs=len(captures),
        n_correct=n_correct,
        accuracy=n_correct / max(1, len(captures)),
        elapsed_s=elapsed,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        stop_sequences=stop_sequences,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path to MLX-format model")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of problem-seed runs (for quick tests)")
    ap.add_argument("--chat-template", action="store_true",
                    help="apply tokenizer chat_template before generation")
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable thinking mode (set enable_thinking=False in chat template)")
    args = ap.parse_args()

    summary = run_sweep(
        args.model, Path(args.out),
        max_tokens=args.max_tokens, temperature=args.temperature,
        limit=args.limit, use_chat_template=args.chat_template,
        enable_thinking=not args.no_thinking,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
