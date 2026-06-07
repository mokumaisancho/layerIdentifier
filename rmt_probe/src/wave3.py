"""Wave 3: Interpretability of RMT signal eigenvectors.

Four tasks on the cached eigenvectors from Wave 1:
  T3.1 — Layer-module clustering (k-means on loading profiles, silhouette).
  T3.2 — Per-mode semantic interpretation (top-8 features, dominant family).
  T3.3 — Cross-architecture mode matching (3x3 cosine matrix on common subset).
  T3.4 — Per-mode single-feature correlation (top-3 |Pearson r|).

Loads each eigvec_cache.npz and the corresponding captures.json exactly once.
Writes one combined dict to results/wave3_results.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_features
from src.eigendecomp import standardize

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ARCHS: List[Tuple[str, str, str]] = [
    (
        "gemma",
        "/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json",
        "/Users/apple/Downloads/Py/layerIdentifier/rmt_probe/results/wave1_gemma/eigvec_cache.npz",
    ),
    (
        "qwen",
        "/Users/apple/Downloads/Py/layerIdentifier/results/qwen3.5-4B-nothinking/captures.json",
        "/Users/apple/Downloads/Py/layerIdentifier/rmt_probe/results/wave1_qwen/eigvec_cache.npz",
    ),
]

FAMILIES: List[str] = [
    "mean_norm", "std_norm", "n_spikes",
    "delta_mean", "delta_std", "delta_variance",
    "convergence_slope", "mid_delta_sum",
    "max_positive_delta", "max_negative_delta",
    "early_spike_ratio", "mid_spike_ratio", "late_spike_ratio",
    "n_delta_spikes",
]

LAYER_RE = re.compile(r"^L(\d+)_")
FAMILY_RE = re.compile(r"^L\d+_(.+)$")

# Gate thresholds (from plan.md acceptance criteria)
SILHOUETTE_GATE = 0.40
XARCH_COSINE_GATE = 0.50
CORR_GATE = 0.70


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _layer_of(name: str) -> int:
    m = LAYER_RE.match(name)
    if m is None:
        raise ValueError(f"unparseable feature name: {name}")
    return int(m.group(1))


def _family_of(name: str) -> str:
    m = FAMILY_RE.match(name)
    if m is None:
        raise ValueError(f"unparseable feature name: {name}")
    return m.group(1)


def _load_arch(captures_path: str, cache_path: str) -> dict:
    """Load captures + eigvec cache; standardize X once and reuse."""
    X, y, feature_names = load_features(captures_path)
    cache = np.load(cache_path, allow_pickle=True)
    cache_names = list(cache["feature_names"])
    # Defensive: cache names should match loader names; align if needed.
    if cache_names != feature_names:
        # Reorder X to match cache order.
        idx = [feature_names.index(n) for n in cache_names]
        X = X[:, idx]
        feature_names = cache_names
    X_std = standardize(X)
    eigvals = np.asarray(cache["eigvals"])
    eigvecs = np.asarray(cache["eigvecs"])
    signal_mask = np.asarray(cache["signal_mask"]).astype(bool)
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_std": X_std,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "signal_mask": signal_mask,
    }


# -----------------------------------------------------------------------------
# T3.1 — Layer-module clustering
# -----------------------------------------------------------------------------

def task31_layer_clustering(arch_data: Dict[str, dict]) -> dict:
    """K-means on per-feature loading profiles; silhouette at k in {2,3,4}.

    Each feature f has a row V_signal[f, :] of length k_signal — its "loading
    profile" across modes. We cluster these profiles, then check whether
    clusters localize to discrete layer ranges.
    """
    out: dict = {}
    for arch, d in arch_data.items():
        V_signal = d["eigvecs"][:, d["signal_mask"]]  # (N, k_signal)
        feature_names = d["feature_names"]
        layers = np.array([_layer_of(n) for n in feature_names])

        per_k: dict = {}
        rng = np.random.RandomState(42)
        for k in (2, 3, 4):
            if V_signal.shape[0] <= k:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(V_signal)
            sil = float(silhouette_score(V_signal, labels))
            cluster_layers: Dict[int, dict] = {}
            for c in range(k):
                mask = labels == c
                cluster_layers[c] = {
                    "n_features": int(mask.sum()),
                    "layer_min": int(layers[mask].min()) if mask.any() else -1,
                    "layer_max": int(layers[mask].max()) if mask.any() else -1,
                    "layer_median": float(np.median(layers[mask])) if mask.any() else -1,
                    "layer_p10": float(np.percentile(layers[mask], 10)) if mask.any() else -1,
                    "layer_p90": float(np.percentile(layers[mask], 90)) if mask.any() else -1,
                    "unique_layer_set": sorted(set(int(x) for x in layers[mask].tolist())) if mask.any() else [],
                }
            per_k[str(k)] = {
                "silhouette": sil,
                "cluster_layers": cluster_layers,
                "labels": labels.tolist(),
            }
        out[arch] = {
            "k_signal": int(d["signal_mask"].sum()),
            "per_k": per_k,
        }
    return out


# -----------------------------------------------------------------------------
# T3.2 — Per-mode semantic interpretation
# -----------------------------------------------------------------------------

def task32_mode_families(arch_data: Dict[str, dict], top_n_features: int = 8) -> dict:
    """For each arch and top-3 signal modes, report dominant feature family.

    Mode index = column index in V_signal (sorted by descending eigval).
    """
    out: dict = {}
    for arch, d in arch_data.items():
        V_signal = d["eigvecs"][:, d["signal_mask"]]  # (N, k_signal)
        feature_names = d["feature_names"]
        families = np.array([_family_of(n) for n in feature_names])
        n_modes = min(3, V_signal.shape[1])
        modes: Dict[str, dict] = {}
        for m in range(n_modes):
            v = V_signal[:, m]
            order = np.argsort(-np.abs(v))  # descending |loading|
            top_idx = order[:top_n_features]
            top_features = [feature_names[i] for i in top_idx]
            top_loadings = [float(v[i]) for i in top_idx]
            top_families = [families[i] for i in top_idx]
            # Count dominant family among the top-N
            fam_counts: Dict[str, int] = {}
            for fam in top_families:
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
            dominant_family, dominant_count = max(fam_counts.items(), key=lambda kv: kv[1])
            modes[f"mode_{m}"] = {
                "top_features": top_features,
                "top_loadings": top_loadings,
                "top_families": top_families,
                "family_counts": fam_counts,
                "dominant_family": dominant_family,
                "dominant_count": dominant_count,
                "dominant_fraction": dominant_count / top_n_features,
            }
        out[arch] = modes
    return out


# -----------------------------------------------------------------------------
# T3.3 — Cross-architecture mode matching
# -----------------------------------------------------------------------------

def task33_xarch_cosine(arch_data: Dict[str, dict]) -> dict:
    """3x3 cosine similarity matrix between Gemma and Qwen top-3 modes.

    Restrict each arch to the common feature subset (intersection of names).
    Cosine is computed on the loading vectors restricted to that subset.
    """
    g = arch_data["gemma"]
    q = arch_data["qwen"]
    g_names = list(g["feature_names"])
    q_names = list(q["feature_names"])
    common = sorted(set(g_names) & set(q_names))
    if len(common) == 0:
        return {"error": "no common features", "common_count": 0}

    # Reindex each arch's V_signal columns (top-3 modes) onto `common`.
    g_V = g["eigvecs"][:, g["signal_mask"]][:, :3]  # (N_g, 3)
    q_V = q["eigvecs"][:, q["signal_mask"]][:, :3]  # (N_q, 3)
    g_idx = np.array([g_names.index(n) for n in common])
    q_idx = np.array([q_names.index(n) for n in common])
    g_V_common = g_V[g_idx, :]  # (N_common, 3)
    q_V_common = q_V[q_idx, :]  # (N_common, 3)

    # cosine_similarity expects (n_samples_X, n_features); here each column is a
    # vector we want to compare, so transpose: shape (3, N_common).
    cos = cosine_similarity(g_V_common.T, q_V_common.T)  # (3, 3)
    cos_list = cos.tolist()
    # Statistics for gate check
    flat = cos.flatten()
    return {
        "common_count": len(common),
        "matrix": cos_list,  # rows = gemma modes, cols = qwen modes
        "max_offdiag_abs": float(np.max(np.abs(cos))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "diag_mean": float(np.mean(np.diag(cos))),
    }


# -----------------------------------------------------------------------------
# T3.4 — Per-mode single-feature correlation
# -----------------------------------------------------------------------------

def task34_per_mode_corr(arch_data: Dict[str, dict], top_n: int = 3) -> dict:
    """For each arch and each top-3 mode, compute |Pearson r| between the
    mode's projection (X_std @ v_i) and each individual feature.

    Report the top-3 features by |corr|.
    """
    out: dict = {}
    for arch, d in arch_data.items():
        X_std = d["X_std"]  # (T, N)
        V_signal = d["eigvecs"][:, d["signal_mask"]]  # (N, k_signal)
        feature_names = d["feature_names"]
        T = X_std.shape[0]
        n_modes = min(3, V_signal.shape[1])
        modes: Dict[str, dict] = {}

        # Standardize columns of X_std again is unnecessary (already z-scored),
        # but center+scale projections to compute Pearson.
        # corr(a, b) = (a·b) / (||a|| ||b||) for centered a, b.
        X_centered = X_std - X_std.mean(axis=0, keepdims=True)
        X_norm = np.linalg.norm(X_centered, axis=0)  # (N,)

        for m in range(n_modes):
            v = V_signal[:, m]
            proj = X_std @ v  # (T,)
            pc = proj - proj.mean()
            pn = float(np.linalg.norm(pc))
            if pn < 1e-12:
                # Degenerate projection; report zeros.
                modes[f"mode_{m}"] = {
                    "top_features": [],
                    "top_corrs": [],
                    "all_below_gate": True,
                }
                continue
            # |corr| with each feature
            numer = X_centered.T @ pc  # (N,)
            denom = X_norm * pn + 1e-12
            corr = np.abs(numer / denom)
            order = np.argsort(-corr)
            top_idx = order[:top_n]
            modes[f"mode_{m}"] = {
                "top_features": [feature_names[i] for i in top_idx],
                "top_corrs": [float(corr[i]) for i in top_idx],
                "max_corr": float(corr[order[0]]),
                "all_above_gate": bool(all(corr[i] >= CORR_GATE for i in top_idx)),
            }
        out[arch] = modes
    return out


# -----------------------------------------------------------------------------
# Gate checks
# -----------------------------------------------------------------------------

def evaluate_ac(results: dict) -> dict:
    """Evaluate acceptance criteria; return PASS/FAIL summary."""
    ac: dict = {}

    # AC3.1 — silhouette >= 0.40 for some k, for both archs.
    ac31_per_arch: Dict[str, dict] = {}
    for arch, body in results["T31_clusters"].items():
        best_sil = -1.0
        best_k = None
        for k_str, k_body in body["per_k"].items():
            if k_body["silhouette"] > best_sil:
                best_sil = k_body["silhouette"]
                best_k = int(k_str)
        ac31_per_arch[arch] = {
            "best_silhouette": best_sil,
            "best_k": best_k,
            "pass": best_sil >= SILHOUETTE_GATE,
        }
    ac["AC3.1_silhouette"] = {
        "gate": f">= {SILHOUETTE_GATE}",
        "per_arch": ac31_per_arch,
        "pass": all(v["pass"] for v in ac31_per_arch.values()),
    }

    # AC3.2 — dominant family reported for top-3 modes for both archs.
    n_modes_reported: Dict[str, int] = {}
    for arch, body in results["T32_modes"].items():
        n_modes_reported[arch] = len(body)
    ac["AC3.2_dominant_family"] = {
        "gate": "dominant family reported for top-3 modes per arch",
        "per_arch": n_modes_reported,
        "pass": all(n >= 3 for n in n_modes_reported.values()),
    }

    # AC3.3 — cross-arch cosine <= 0.50 confirms transfer failure.
    max_abs = results["T33_xarch"]["max_offdiag_abs"]
    ac["AC3.3_xarch_cosine"] = {
        "gate": f"max |cos| <= {XARCH_COSINE_GATE} (transfer failure confirmed)",
        "max_abs_cosine": max_abs,
        "mean_abs_cosine": results["T33_xarch"]["mean_abs"],
        "diag_mean": results["T33_xarch"]["diag_mean"],
        "pass": max_abs <= XARCH_COSINE_GATE,
    }

    # AC3.4 — top-3 features per mode with |corr| >= 0.7.
    ac34_per_arch: Dict[str, dict] = {}
    for arch, body in results["T34_corr"].items():
        all_ok = True
        per_mode_max: Dict[str, float] = {}
        for mode_key, mode_body in body.items():
            per_mode_max[mode_key] = mode_body.get("max_corr", 0.0)
            if not mode_body.get("all_above_gate", False):
                all_ok = False
        ac34_per_arch[arch] = {
            "per_mode_max_corr": per_mode_max,
            "pass": all_ok,
        }
    ac["AC3.4_top_corr"] = {
        "gate": f"top-3 features all with |corr| >= {CORR_GATE}",
        "per_arch": ac34_per_arch,
        "pass": all(v["pass"] for v in ac34_per_arch.values()),
    }

    return ac


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).parent.parent
    out_path = base / "results" / "wave3_results.json"

    print("=" * 72)
    print("WAVE 3: Interpretability of RMT signal eigenvectors")
    print("=" * 72)

    # Load both architectures once.
    arch_data: Dict[str, dict] = {}
    for name, cap_path, cache_path in ARCHS:
        cap_p = Path(cap_path)
        cache_p = Path(cache_path)
        if not cap_p.exists() or not cache_p.exists():
            print(f"  [skip] {name}: missing inputs")
            continue
        print(f"  loading {name} ...")
        arch_data[name] = _load_arch(str(cap_p), str(cache_p))
        d = arch_data[name]
        print(
            f"    T={d['X'].shape[0]}, N={d['X'].shape[1]}, "
            f"k_signal={int(d['signal_mask'].sum())}"
        )

    if len(arch_data) < 2:
        print("FAIL: need both archs loaded")
        return

    # Run all four tasks.
    print("\n[T3.1] layer-module clustering ...")
    t31 = task31_layer_clustering(arch_data)
    for arch, body in t31.items():
        for k_str, k_body in body["per_k"].items():
            print(
                f"  {arch} k={k_str}: silhouette={k_body['silhouette']:.3f}"
            )

    print("\n[T3.2] per-mode semantic interpretation ...")
    t32 = task32_mode_families(arch_data)
    for arch, body in t32.items():
        for mode_key, mode_body in body.items():
            print(
                f"  {arch} {mode_key}: dominant='{mode_body['dominant_family']}' "
                f"({mode_body['dominant_count']}/{8} top features)"
            )

    print("\n[T3.3] cross-architecture mode matching ...")
    t33 = task33_xarch_cosine(arch_data)
    print(f"  common features: {t33['common_count']}")
    print(f"  max |cos|: {t33['max_offdiag_abs']:.3f}")
    print(f"  mean |cos|: {t33['mean_abs']:.3f}")
    print("  matrix (rows=gemma modes, cols=qwen modes):")
    for row in t33["matrix"]:
        print("    " + "  ".join(f"{v:+.3f}" for v in row))

    print("\n[T3.4] per-mode single-feature correlation ...")
    t34 = task34_per_mode_corr(arch_data)
    for arch, body in t34.items():
        for mode_key, mode_body in body.items():
            feats = mode_body.get("top_features", [])
            corrs = mode_body.get("top_corrs", [])
            if feats:
                pairs = ", ".join(f"{f}({c:.2f})" for f, c in zip(feats, corrs))
                print(
                    f"  {arch} {mode_key}: max|r|={mode_body['max_corr']:.3f}  "
                    f"top3={pairs}"
                )

    # Assemble + evaluate ACs.
    results = {
        "T31_clusters": t31,
        "T32_modes": t32,
        "T33_xarch": t33,
        "T34_corr": t34,
    }
    ac = evaluate_ac(results)
    results["acceptance_criteria"] = ac

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== ACCEPTANCE CRITERIA ===")
    for ac_key, ac_body in ac.items():
        status = "PASS" if ac_body["pass"] else "FAIL"
        print(f"  [{status}] {ac_key}  (gate: {ac_body['gate']})")
        if ac_key == "AC3.1_silhouette":
            for a, b in ac_body["per_arch"].items():
                print(f"    {a}: best silhouette={b['best_silhouette']:.3f} at k={b['best_k']}")
        elif ac_key == "AC3.3_xarch_cosine":
            print(f"    max |cos|={ac_body['max_abs_cosine']:.3f}, "
                  f"mean |cos|={ac_body['mean_abs_cosine']:.3f}")
        elif ac_key == "AC3.4_top_corr":
            for a, b in ac_body["per_arch"].items():
                pmm = b["per_mode_max_corr"]
                print(f"    {a}: per-mode max|r| = " +
                      ", ".join(f"{m}={v:.3f}" for m, v in pmm.items()))


if __name__ == "__main__":
    main()
