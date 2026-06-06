"""Benchmark early-exit vs baseline on a captures.json dataset.

Trains a probe on a (different) captures set, then runs early-exit on each
problem and measures:
  - Wall-clock speedup vs baseline
  - Accuracy on completed-only
  - Abort rate
  - False-abort rate (aborted but actually correct)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .model_loader import load_model
from .probe_dataset import build_probe_dataset, is_correct
from .early_exit import (
    train_probe_from_captures,
    generate_with_early_exit,
)


# Default probe feature config — top-N layers + most-discriminative features.
# These come from FINAL_FINDINGS_2026-06-06.md for each architecture.
PROBE_CONFIGS = {
    "gemma": {
        "layers": [6, 15, 41, 28, 21, 27, 2, 34, 23, 11],
        "features": ["convergence_slope", "n_spikes", "std_norm", "mean_norm",
                     "mid_delta_sum", "max_positive_delta", "max_negative_delta",
                     "delta_variance", "delta_std", "mid_spike_ratio",
                     "delta_mean", "n_delta_spikes"],
    },
    "qwen": {
        "layers": [15, 7, 5, 6, 4, 2, 14, 24, 26, 27],
        "features": ["convergence_slope", "n_spikes", "std_norm", "mean_norm",
                     "mid_delta_sum", "max_positive_delta", "max_negative_delta",
                     "delta_variance", "delta_std", "mid_spike_ratio",
                     "delta_mean", "n_delta_spikes"],
    },
}


def wrap_prompt(lm, prompt: str, use_chat_template: bool, enable_thinking: bool,
                stop_sequences: list[str] | None) -> str:
    if not use_chat_template:
        return prompt
    msgs = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    return lm.tokenizer.apply_chat_template(msgs, **kwargs)


def run_bench(model_path: str, captures_path: Path, arch: str,
              out_dir: Path, max_tokens: int = 150, temperature: float = 0.1,
              cutoff: int = 5, threshold: float = 0.5,
              abort_consecutive: int = 2, limit: int | None = None):
    print(f"[bench] loading probe-training captures from {captures_path}")
    train_caps = json.loads(captures_path.read_text())
    print(f"[bench] {len(train_caps)} training captures")

    cfg = PROBE_CONFIGS[arch]
    probe = train_probe_from_captures(train_caps, cfg["layers"], cfg["features"])

    # 5-fold CV check of probe quality on training set (sanity)
    from sklearn.model_selection import cross_val_score
    feature_names = [f"L{ly}_{fk}" for ly in cfg["layers"] for fk in cfg["features"]]
    X = np.array([[c.get("layer_features", {}).get(fn, 0.0) for fn in feature_names]
                  for c in train_caps])
    y = np.array([1 if c.get("final_correct") else 0 for c in train_caps])
    cv_auc = cross_val_score(probe, X, y, cv=5, scoring="roc_auc")
    print(f"[bench] probe 5-fold CV AUROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # Load model
    print(f"[bench] loading model {model_path}")
    lm = load_model(model_path)

    # Build dataset (same as original sweep)
    examples = build_probe_dataset(seeds=(7, 42, 123))
    if limit:
        examples = examples[:limit]
    print(f"[bench] running early-exit on {len(examples)} problems "
          f"(cutoff={cutoff}, threshold={threshold}, abort_consec={abort_consecutive})")

    results = []
    total_baseline_tokens = 0
    total_exit_tokens = 0
    total_baseline_time = 0.0
    total_exit_time = 0.0

    for i, ex in enumerate(examples, 1):
        np.random.seed(ex.seed)

        # Use training captures' settings
        train_sample = train_caps[0]
        use_chat = train_sample.get("chat_template", True) if hasattr(train_sample, "get") else True
        # Best-effort: assume chat-template + no-thinking for CLEAN runs
        prompt = wrap_prompt(lm, ex.prompt, use_chat_template=True, enable_thinking=False,
                             stop_sequences=None)

        trace = generate_with_early_exit(
            lm.model, lm.tokenizer, prompt,
            probe=probe,
            target_layers=cfg["layers"],
            feature_keys=cfg["features"],
            max_tokens=max_tokens,
            temperature=temperature,
            cutoff=cutoff,
            threshold=threshold,
            abort_consecutive=abort_consecutive,
            stop_sequences=["<turn|>", "<|end|>"] if arch == "gemma" else ["<|im_end|>"],
        )

        # Use baseline n_tokens from training captures (mean) as comparison
        correct = is_correct(trace.text, ex.ground_truth)
        aborted = trace.aborted

        results.append({
            "problem_id": ex.problem_id,
            "seed": ex.seed,
            "ground_truth": ex.ground_truth,
            "model_answer_text": trace.text[:200],
            "final_correct": correct,
            "n_tokens": trace.n_tokens,
            "tps": trace.tps,
            "aborted": aborted,
            "abort_token": trace.abort_token,
            "p_correct_at_abort": trace.p_correct_at_abort,
        })

        total_exit_tokens += trace.n_tokens
        total_exit_time += trace.elapsed_s

        marker = "ABORT" if aborted else "OK   "
        print(f"  [{i}/{len(examples)}] {ex.problem_id}: {marker} tok={trace.n_tokens} "
              f"tps={trace.tps:.1f} correct={correct}")

    # Compute baseline stats from training captures (same problems, no abort)
    # Match by problem_id+seed
    baseline_by_key = {}
    for c in train_caps:
        key = (c.get("problem_id"), c.get("seed"))
        baseline_by_key[key] = (c.get("n_tokens", 0), c.get("tps", 0), c.get("final_correct", False))

    matched = 0
    for r in results:
        key = (r["problem_id"], r["seed"])
        if key in baseline_by_key:
            bt, btps, bc = baseline_by_key[key]
            total_baseline_tokens += bt
            total_baseline_time += bt / btps if btps > 0 else 0
            matched += 1

    print(f"\n[bench] matched {matched}/{len(results)} to baseline")

    n_aborted = sum(1 for r in results if r["aborted"])
    n_correct = sum(1 for r in results if r["final_correct"])
    n_aborted_correct = sum(1 for r in results if r["aborted"] and r["final_correct"])

    avg_baseline_tps = (total_baseline_tokens / total_baseline_time) if total_baseline_time > 0 else 0
    avg_exit_tps = (total_exit_tokens / total_exit_time) if total_exit_time > 0 else 0

    print("\n=== EARLY-EXIT BENCH RESULTS ===")
    print(f"N problems:        {len(results)}")
    print(f"N aborted:         {n_aborted} ({n_aborted*100/len(results):.1f}%)")
    print(f"N correct (any):   {n_correct} ({n_correct*100/len(results):.1f}%)")
    print(f"False aborts:      {n_aborted_correct} (aborted but actually correct)")
    print(f"Avg baseline TPS:  {avg_baseline_tps:.2f}")
    print(f"Avg early-exit TPS:{avg_exit_tps:.2f}")
    print(f"TPS gain:          {avg_exit_tps/avg_baseline_tps:.2f}×")
    print(f"Token savings:     {total_baseline_tokens - total_exit_tokens} tokens "
          f"({(total_baseline_tokens - total_exit_tokens)*100/max(1,total_baseline_tokens):.1f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"early_exit_bench_c{cutoff}_t{threshold}_a{abort_consecutive}.json"
    out_path.write_text(json.dumps({
        "config": {
            "arch": arch,
            "model_path": model_path,
            "cutoff": cutoff,
            "threshold": threshold,
            "abort_consecutive": abort_consecutive,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "probe_cv_auc": {"mean": float(cv_auc.mean()), "std": float(cv_auc.std())},
        "summary": {
            "n_problems": len(results),
            "n_aborted": n_aborted,
            "abort_rate": n_aborted / len(results),
            "n_correct": n_correct,
            "accuracy": n_correct / len(results),
            "false_aborts": n_aborted_correct,
            "false_abort_rate": n_aborted_correct / max(1, n_aborted),
            "baseline_total_tokens": total_baseline_tokens,
            "exit_total_tokens": total_exit_tokens,
            "baseline_total_time_s": total_baseline_time,
            "exit_total_time_s": total_exit_time,
            "avg_baseline_tps": avg_baseline_tps,
            "avg_exit_tps": avg_exit_tps,
            "tps_gain": avg_exit_tps / max(0.01, avg_baseline_tps),
        },
        "per_problem": results,
    }, indent=2))
    print(f"\n[bench] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--captures", required=True, help="path to baseline captures.json (used to train probe)")
    ap.add_argument("--arch", required=True, choices=["gemma", "qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--abort-consecutive", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_bench(
        model_path=args.model,
        captures_path=Path(args.captures),
        arch=args.arch,
        out_dir=Path(args.out),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        cutoff=args.cutoff,
        threshold=args.threshold,
        abort_consecutive=args.abort_consecutive,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
