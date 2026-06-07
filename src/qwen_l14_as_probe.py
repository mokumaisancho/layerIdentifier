"""L14-as-probe check: does L14 carry the discriminative feature itself?

Question raised by Move #5 Qwen: L14 is the dominant WRITER of L15's signal
(ablation Δ=-0.360), but does L14 also HOLD the signal at the feature level?

Method:
  For each layer L in {L12, L13, L14, L15}, train a probe using ONLY that
  layer's 14 features. Compare AUROC.

Interpretation:
  - L14 AUROC ≈ L15 AUROC → L14 carries the signal (passive holder + writer)
  - L14 AUROC ≪ L15 AUROC → L14 writes via a non-feature mechanism (routing)
  - L14 AUROC between → L14 has partial signal; L15 amplifies

Also reports L14 single-feature AUROCs (which of the 14 features is most
discriminative at L14).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/apple/Downloads/Py/layerIdentifier")
CAPTURES = ROOT / "results/qwen3.5-4B-nothinking/captures.json"
OUT = ROOT / "results/qwen_l14_as_probe.json"

LAYERS = [12, 13, 14, 15]
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42


def load_layer_features(layer: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for the given layer only."""
    with open(CAPTURES) as f:
        data = json.load(f)
    feature_names = sorted(
        k for k in data[0]["layer_features"].keys() if k.startswith(f"L{layer}_")
    )
    T = len(data)
    N = len(feature_names)
    X = np.zeros((T, N), dtype=np.float64)
    y = np.zeros(T, dtype=np.int64)
    for i, ex in enumerate(data):
        for j, k in enumerate(feature_names):
            X[i, j] = float(ex["layer_features"].get(k, 0.0))
        y[i] = int(bool(ex["final_correct"]))
    return X, y, feature_names


def cv_auroc(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            per_fold.append(0.5)
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        clf.fit(X[tr], y[tr])
        scores = clf.predict_proba(X[te])[:, 1]
        per_fold.append(float(roc_auc_score(y[te], scores)))
    return {
        "mean": float(np.mean(per_fold)),
        "std": float(np.std(per_fold)),
        "per_fold": per_fold,
    }


def bootstrap_ci(X: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    T = len(y)
    means = []
    for _ in range(n):
        idx = rng.integers(0, T, T)
        if len(np.unique(y[idx])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        clf.fit(X[idx], y[idx])
        scores = clf.predict_proba(X)[:, 1]
        means.append(float(roc_auc_score(y, scores)))
    arr = np.array(means)
    return {
        "point": float(np.mean(arr)),
        "lower": float(np.percentile(arr, 2.5)),
        "upper": float(np.percentile(arr, 97.5)),
        "half_width": float((np.percentile(arr, 97.5) - np.percentile(arr, 2.5)) / 2),
        "n_resamples": len(means),
    }


def single_feature_auroc(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    """For each feature column, compute AUROC (no CV — single feature)."""
    out = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        if np.std(col) == 0:
            out.append({"feature": name, "auroc": 0.5})
            continue
        # Direction-agnostic: AUROC of col or 1-col, whichever is higher
        a1 = float(roc_auc_score(y, col))
        a2 = float(roc_auc_score(y, -col))
        a = max(a1, a2)
        out.append({"feature": name, "auroc": a, "direction": "+" if a1 >= a2 else "-"})
    return sorted(out, key=lambda d: -d["auroc"])


def main():
    print(f"Loading captures from {CAPTURES}")
    print(f"T = 150 (Qwen), feature families per layer = 14")
    print()

    results = {
        "config": {
            "n_folds": N_FOLDS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "baseline_L15_n_spikes_auroc": 0.9189938601703308,
        },
        "per_layer": {},
    }

    for L in LAYERS:
        print(f"--- L{L} ---")
        X, y, names = load_layer_features(L)
        print(f"  X shape: {X.shape}, positives: {int(y.sum())}/{len(y)}")

        cv = cv_auroc(X, y)
        ci = bootstrap_ci(X, y)
        single = single_feature_auroc(X, y, names)

        results["per_layer"][f"L{L}"] = {
            "n_features": len(names),
            "cv_auroc": cv,
            "bootstrap_ci": ci,
            "top_features": single[:5],
        }
        print(f"  CV AUROC: {cv['mean']:.4f} ± {cv['std']:.4f}")
        print(f"  Bootstrap 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]  (hw={ci['half_width']:.4f})")
        print(f"  Top features:")
        for f in single[:3]:
            d = f.get("direction", "?")
            print(f"    {f['feature']:35s}  AUROC={f['auroc']:.4f} ({d})")
        print()

    # Verdict
    l14 = results["per_layer"]["L14"]["cv_auroc"]["mean"]
    l15 = results["per_layer"]["L15"]["cv_auroc"]["mean"]
    diff = l15 - l14
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"L14 CV AUROC: {l14:.4f}")
    print(f"L15 CV AUROC: {l15:.4f}")
    print(f"Difference (L15 - L14): {diff:+.4f}")
    if diff < 0.05:
        verdict = "CARRIER+WRITER: L14 holds the signal at the feature level"
    elif diff < 0.20:
        verdict = "PARTIAL: L14 has signal but L15 amplifies"
    else:
        verdict = "WRITER-ONLY: L14 writes via non-feature mechanism (routing/attention)"
    print(f"Interpretation: {verdict}")
    results["verdict"] = {"L14_auroc": l14, "L15_auroc": l15, "diff": diff, "interpretation": verdict}

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
