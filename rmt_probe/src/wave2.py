"""Wave 2: Robustness characterization of RMT-cleaned probe.

Four tasks, all reading from Wave 1 eigvec caches and raw captures.json:
  T2.1 — k-sweep:        AUROC vs k top-eigenvalue modes (no MP filter)
  T2.2 — 10-fold CV:     per-fold variance at MP-selected k
  T2.3 — PCA-k baseline: sklearn PCA top-k vs RMT top-k
  T2.4 — Bonferroni:     top-k individual features by t-test rank

Writes a single JSON: results/wave2_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eigendecomp import standardize
from src.loader import load_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
WAVE1_DIR = {arch: RESULTS_DIR / f"wave1_{arch}" for arch in ("gemma", "qwen")}
CAPTURES = {
    "gemma": Path("/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json"),
    "qwen": Path("/Users/apple/Downloads/Py/layerIdentifier/results/qwen3.5-4B-nothinking/captures.json"),
}

K_SWEEP: tuple[int, ...] = (3, 5, 8, 10, 12, 14, 20, 30, 50, 100)
K_BONF: tuple[int, ...] = (3, 5, 10, 14, 20, 30)
SEED = 42
N_FOLDS_KSWEEP = 5
N_FOLDS_VARIANCE = 10

# Match Wave 1 reference k (MP-selected)
MP_K = {"gemma": 14, "qwen": 12}


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------

def _cv_auroc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict:
    """Stratified k-fold CV L2-LR AUROC. Mirrors src.probe.cross_val_auroc but
    kept local so Wave 2 has no implicit dependency on internal module state."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold: list[float] = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            per_fold.append(0.5)
            continue
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
        )
        clf.fit(X[tr], y[tr])
        scores = clf.predict_proba(X[te])[:, 1]
        per_fold.append(float(roc_auc_score(y[te], scores)))
    return {
        "mean": float(np.mean(per_fold)),
        "std": float(np.std(per_fold)),
        "per_fold": per_fold,
    }


def _project_topk(X_std: np.ndarray, eigvecs: np.ndarray, k: int) -> np.ndarray:
    """Project onto the top-k eigenvectors (already sorted descending by eigval).

    Returns array of shape (T, k). Clip k to available columns.
    """
    k_eff = min(k, eigvecs.shape[1])
    return X_std @ eigvecs[:, :k_eff]


def _pca_topk(X_std: np.ndarray, k: int) -> np.ndarray:
    """sklearn PCA projection onto top-k singular-value components."""
    k_eff = min(k, X_std.shape[1], X_std.shape[0])
    pca = PCA(n_components=k_eff, random_state=SEED)
    return pca.fit_transform(X_std)


def _bonferroni_rank(X_std: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of features sorted by Bonferroni-style t-test rank.

    For each feature column, compute Welch's t-test (correct vs incorrect).
    Rank by ascending p-value (most discriminative first). This matches the
    Move #3 reference: Bonferroni-corrected survival at α < 0.05/N is the
    canonical filter, and ranking by raw p-value is the equivalent continuous
    ordering.
    """
    n_features = X_std.shape[1]
    pvals = np.empty(n_features, dtype=np.float64)
    pos_mask = y == 1
    neg_mask = y == 0
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    for j in range(n_features):
        col = X_std[:, j]
        pos_vals = col[pos_mask]
        neg_vals = col[neg_mask]
        if pos_count < 2 or neg_count < 2:
            pvals[j] = 1.0
            continue
        # Welch's t-test, nan-safe
        t_stat, p_val = ttest_ind(pos_vals, neg_vals, equal_var=False)
        if np.isnan(p_val):
            pvals[j] = 1.0
        else:
            pvals[j] = float(p_val)
    # Smallest p-value first
    return np.argsort(pvals, kind="stable")


# ---------------------------------------------------------------------------
# Per-task drivers
# ---------------------------------------------------------------------------

def task_t21_ksweep(
    arch: str,
    X_std: np.ndarray,
    y: np.ndarray,
    eigvecs: np.ndarray,
) -> dict:
    """T2.1 — AUROC vs k for top-k eigenvalue projection (5-fold CV)."""
    out: dict = {"arch": arch, "n_folds": N_FOLDS_KSWEEP, "k_values": list(K_SWEEP), "auroc": {}}
    for k in K_SWEEP:
        X_proj = _project_topk(X_std, eigvecs, k)
        auroc = _cv_auroc(X_proj, y, n_folds=N_FOLDS_KSWEEP, seed=SEED)
        out["auroc"][str(k)] = auroc
    return out


def task_t22_variance(
    arch: str,
    X_std: np.ndarray,
    y: np.ndarray,
    eigvecs: np.ndarray,
) -> dict:
    """T2.2 — 10-fold CV variance at MP-selected k."""
    k = MP_K[arch]
    X_proj = _project_topk(X_std, eigvecs, k)
    auroc = _cv_auroc(X_proj, y, n_folds=N_FOLDS_VARIANCE, seed=SEED)
    return {
        "arch": arch,
        "k": k,
        "n_folds": N_FOLDS_VARIANCE,
        "auroc": auroc,
    }


def task_t23_pca(
    arch: str,
    X_std: np.ndarray,
    y: np.ndarray,
) -> dict:
    """T2.3 — PCA top-k baseline (sklearn PCA, same k values as T2.1)."""
    out: dict = {"arch": arch, "n_folds": N_FOLDS_KSWEEP, "k_values": list(K_SWEEP), "auroc": {}}
    for k in K_SWEEP:
        X_pca = _pca_topk(X_std, k)
        auroc = _cv_auroc(X_pca, y, n_folds=N_FOLDS_KSWEEP, seed=SEED)
        out["auroc"][str(k)] = auroc
    return out


def task_t24_bonferroni(
    arch: str,
    X_std: np.ndarray,
    y: np.ndarray,
) -> dict:
    """T2.4 — top-k Bonferroni-ranked raw features baseline."""
    ranked_idx = _bonferroni_rank(X_std, y)
    out: dict = {
        "arch": arch,
        "n_folds": N_FOLDS_KSWEEP,
        "k_values": list(K_BONF),
        "auroc": {},
        "top_features_at_k14": [],
    }
    for k in K_BONF:
        sel = ranked_idx[:k]
        X_sel = X_std[:, sel]
        auroc = _cv_auroc(X_sel, y, n_folds=N_FOLDS_KSWEEP, seed=SEED)
        out["auroc"][str(k)] = auroc
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_arch(arch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load (X_std, y, eigvecs, wave1_meta) for one arch. Reuses Wave 1 cache."""
    cache_path = WAVE1_DIR[arch] / "eigvec_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing eigvec cache: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    eigvecs = np.asarray(cache["eigvecs"])

    cap_path = CAPTURES[arch]
    if not cap_path.exists():
        raise FileNotFoundError(f"Missing captures: {cap_path}")
    X, y, _ = load_features(str(cap_path))
    X_std = standardize(X)

    mvp_path = WAVE1_DIR[arch] / "mvp_results.json"
    with open(mvp_path) as f:
        wave1_meta = json.load(f)
    return X_std, y, eigvecs, wave1_meta


def main() -> None:
    print("=" * 72)
    print("WAVE 2: Robustness (k-sweep, 10-fold CV, PCA baseline, Bonferroni)")
    print("=" * 72)

    results: dict[str, dict] = {
        "T21_ksweep": {},
        "T22_variance": {},
        "T23_pca": {},
        "T24_bonferroni": {},
    }

    for arch in ("gemma", "qwen"):
        print(f"\n--- {arch} ---")
        X_std, y, eigvecs, _wave1 = _load_arch(arch)
        T, N = X_std.shape
        print(f"  T={T}, N={N}, k_mp={MP_K[arch]}")

        # T2.1 — k-sweep
        t21 = task_t21_ksweep(arch, X_std, y, eigvecs)
        results["T21_ksweep"][arch] = t21
        print("  T2.1 k-sweep (5-fold CV AUROC):")
        for k in K_SWEEP:
            m = t21["auroc"][str(k)]["mean"]
            s = t21["auroc"][str(k)]["std"]
            marker = " *" if arch == "qwen" and m >= 0.95 else ""
            print(f"    k={k:>3d}  AUROC = {m:.3f} ± {s:.3f}{marker}")

        # T2.2 — 10-fold variance
        t22 = task_t22_variance(arch, X_std, y, eigvecs)
        results["T22_variance"][arch] = t22
        print(
            f"  T2.2 10-fold @ k={t22['k']}: AUROC = "
            f"{t22['auroc']['mean']:.3f} ± {t22['auroc']['std']:.3f}"
        )
        print(f"    per-fold: {[round(v, 3) for v in t22['auroc']['per_fold']]}")

        # T2.3 — PCA-k baseline
        t23 = task_t23_pca(arch, X_std, y)
        results["T23_pca"][arch] = t23
        print("  T2.3 PCA-k (5-fold CV AUROC):")
        for k in K_SWEEP:
            m = t23["auroc"][str(k)]["mean"]
            s = t23["auroc"][str(k)]["std"]
            rmt_m = t21["auroc"][str(k)]["mean"]
            delta = rmt_m - m
            print(f"    k={k:>3d}  PCA = {m:.3f} ± {s:.3f}  (RMT-PCA = {delta:+.3f})")

        # T2.4 — Bonferroni baseline
        t24 = task_t24_bonferroni(arch, X_std, y)
        results["T24_bonferroni"][arch] = t24
        print("  T2.4 top-k Bonferroni features (5-fold CV AUROC):")
        for k in K_BONF:
            m = t24["auroc"][str(k)]["mean"]
            s = t24["auroc"][str(k)]["std"]
            print(f"    k={k:>3d}  Bonf = {m:.3f} ± {s:.3f}")

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "wave2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # ---------------------------------------------------------------
    # Acceptance criteria summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ACCEPTANCE CRITERIA")
    print("=" * 72)

    # AC2.1 — k-sweep coverage + Qwen crossing
    qwen_ksweep = results["T21_ksweep"]["qwen"]["auroc"]
    qwen_first_ge_95: int | None = None
    for k in K_SWEEP:
        if qwen_ksweep[str(k)]["mean"] >= 0.95 and qwen_first_ge_95 is None:
            qwen_first_ge_95 = k
    ac21_pass = all(str(k) in qwen_ksweep for k in K_SWEEP) and all(
        str(k) in results["T21_ksweep"]["gemma"]["auroc"] for k in K_SWEEP
    )
    print(
        f"  AC2.1 (k-sweep coverage, both archs, 10 k values): "
        f"{'PASS' if ac21_pass else 'FAIL'}"
    )
    print(
        f"        Qwen first reaches AUROC >= 0.95 at k = "
        f"{qwen_first_ge_95 if qwen_first_ge_95 is not None else 'NEVER'}"
    )

    # AC2.2 — 10-fold std <= 0.040 on both archs
    stds = {
        a: results["T22_variance"][a]["auroc"]["std"] for a in ("gemma", "qwen")
    }
    ac22_pass = all(v <= 0.040 for v in stds.values())
    print(
        f"  AC2.2 (10-fold std <= 0.040): "
        f"{'PASS' if ac22_pass else 'FAIL'}  "
        f"(gemma={stds['gemma']:.3f}, qwen={stds['qwen']:.3f})"
    )

    # AC2.3 — RMT >= PCA at k=14 (gemma) and k=12 (qwen)
    rmt_g = results["T21_ksweep"]["gemma"]["auroc"]["14"]["mean"]
    pca_g = results["T23_pca"]["gemma"]["auroc"]["14"]["mean"]
    rmt_q = results["T21_ksweep"]["qwen"]["auroc"]["12"]["mean"]
    pca_q = results["T23_pca"]["qwen"]["auroc"]["12"]["mean"]
    ac23_pass = (rmt_g >= pca_g) and (rmt_q >= pca_q)
    print(
        f"  AC2.3 (RMT >= PCA @ k=14 Gemma, k=12 Qwen): "
        f"{'PASS' if ac23_pass else 'FAIL'}  "
        f"(Gemma RMT={rmt_g:.3f} vs PCA={pca_g:.3f}; "
        f"Qwen RMT={rmt_q:.3f} vs PCA={pca_q:.3f})"
    )

    # AC2.4 — Bonferroni numbers present at all 6 k values for both archs
    ac24_pass = all(
        all(str(k) in results["T24_bonferroni"][a]["auroc"] for k in K_BONF)
        for a in ("gemma", "qwen")
    )
    bonf_g_14 = results["T24_bonferroni"]["gemma"]["auroc"]["14"]["mean"]
    bonf_q_14 = results["T24_bonferroni"]["qwen"]["auroc"]["14"]["mean"] if "14" in results["T24_bonferroni"]["qwen"]["auroc"] else float("nan")
    print(
        f"  AC2.4 (Bonferroni baseline, k in {{3,5,10,14,20,30}}): "
        f"{'PASS' if ac24_pass else 'FAIL'}  "
        f"(Bonf@14 Gemma={bonf_g_14:.3f}, Qwen={bonf_q_14:.3f})"
    )


if __name__ == "__main__":
    main()
