"""Wave 1 MVP pipeline: RMT-cleaned probe vs raw probe comparison.

Orchestrates: load → standardize → LedoitWolf + eigendecomp → MP filter →
project → CV LR probe → bootstrap CI → save results + cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_features
from src.eigendecomp import (
    standardize,
    compute_signal_decomp,
    project_onto_signal,
    save_cache,
)
from src.probe import cross_val_auroc
from src.bootstrap import bootstrap_ci


def run_mvp(
    arch_name: str,
    captures_path: str,
    out_dir: str,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Run Wave 1 MVP for one architecture.

    Returns dict with:
      arch, T, N, k_signal, mp_lambda_max, shrinkage,
      auroc_rmt, auroc_raw, ci_rmt, ci_raw
    """
    X, y, feature_names = load_features(captures_path)
    T, N = X.shape
    Q = N / T
    mp_lambda_max = float((1 + np.sqrt(Q)) ** 2)

    X_std = standardize(X)

    decomp = compute_signal_decomp(X_std, mp_lambda_max)
    signal_mask = decomp["signal_mask"]
    V_signal = decomp["eigvecs"][:, signal_mask]
    k_signal = int(signal_mask.sum())

    X_proj = project_onto_signal(X_std, V_signal)

    auroc_rmt = cross_val_auroc(X_proj, y, n_folds=n_folds, seed=seed)
    auroc_raw = cross_val_auroc(X_std, y, n_folds=n_folds, seed=seed)

    ci_rmt = bootstrap_ci(X_proj, y, n_resamples=n_bootstrap, seed=seed)
    ci_raw = bootstrap_ci(X_std, y, n_resamples=n_bootstrap, seed=seed)

    result = {
        "arch": arch_name,
        "T": T,
        "N": N,
        "Q": float(Q),
        "k_signal": k_signal,
        "mp_lambda_max": mp_lambda_max,
        "shrinkage": decomp["shrinkage"],
        "auroc_rmt": auroc_rmt,
        "auroc_raw": auroc_raw,
        "ci_rmt": ci_rmt,
        "ci_raw": ci_raw,
        "n_bootstrap": n_bootstrap,
        "n_folds": n_folds,
        "seed": seed,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "mvp_results.json").write_text(json.dumps(result, indent=2))
    save_cache(
        str(out_path / "eigvec_cache.npz"),
        decomp,
        feature_names=feature_names,
    )
    return result


def main():
    """Wave 1: run MVP for both Gemma and Qwen."""
    base = Path("/Users/apple/Downloads/Py/layerIdentifier/results")
    archs = [
        ("gemma", base / "gemma-4-E4B-nothinking" / "captures.json"),
        ("qwen", base / "qwen3.5-4B-nothinking" / "captures.json"),
    ]
    out_root = Path(__file__).parent.parent / "results"

    print("=" * 72)
    print("WAVE 1: MVP — RMT probe vs raw probe (5-fold CV + bootstrap CI)")
    print("=" * 72)

    all_results = {}
    for name, path in archs:
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        print(f"\n--- {name} ---")
        out_dir = out_root / f"wave1_{name}"
        result = run_mvp(
            arch_name=name,
            captures_path=str(path),
            out_dir=str(out_dir),
            n_folds=5,
            n_bootstrap=1000,
            seed=42,
        )
        all_results[name] = result
        print(f"  T={result['T']}, N={result['N']}, k_signal={result['k_signal']}")
        print(f"  RMT-k AUROC: {result['auroc_rmt']['mean']:.3f} ± {result['auroc_rmt']['std']:.3f}  "
              f"CI [{result['ci_rmt']['lower']:.3f}, {result['ci_rmt']['upper']:.3f}]")
        print(f"  Raw  AUROC: {result['auroc_raw']['mean']:.3f} ± {result['auroc_raw']['std']:.3f}  "
              f"CI [{result['ci_raw']['lower']:.3f}, {result['ci_raw']['upper']:.3f}]")

    # Save combined summary
    (out_root / "wave1_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_root}/wave1_summary.json")

    # Gate 1 check
    print("\n=== GATE 1 ===")
    pass_count = 0
    for name, r in all_results.items():
        auroc_ok = r["auroc_rmt"]["mean"] >= 0.950
        ci_ok = r["ci_rmt"]["half_width"] <= 0.050
        baseline_ok = (
            (name == "gemma" and abs(r["auroc_raw"]["mean"] - 0.981) <= 0.05) or
            (name == "qwen" and abs(r["auroc_raw"]["mean"] - 0.996) <= 0.05)
        )
        print(f"  {name}: auroc_rmt={r['auroc_rmt']['mean']:.3f} (need ≥0.950) {('✓' if auroc_ok else '✗')}")
        print(f"  {name}: ci_half_width={r['ci_rmt']['half_width']:.3f} (need ≤0.050) {('✓' if ci_ok else '✗')}")
        print(f"  {name}: baseline={r['auroc_raw']['mean']:.3f} (within ±0.05 of reference) {('✓' if baseline_ok else '✗')}")
        if auroc_ok and ci_ok:
            pass_count += 1

    print(f"\nGate 1: {pass_count}/{len(all_results)} architectures pass")
    if pass_count == len(all_results):
        print("PROCEED to Wave 2/3")
    else:
        print("STOP — write up negative result")


if __name__ == "__main__":
    main()
