"""Feature computation — IDENTICAL formulas to parallel_l4_tot reference.

Features per Qwen SIGNAL_CATALOG.md:
  - entropy vec: mean, max, first, std, n_spike, spike_positions
  - entropy delta: max_positive, max_negative, variance, mean, std
  - per-layer L2 norm: mid_spike_ratio, late_spike_ratio, early_spike_ratio
  - per-layer delta: max_negative_delta, mid_delta_sum, convergence_slope, n_spikes

We add per-layer versions of every entropy-delta feature so ranking covers
both Qwen's published + discovered-not-wired signals.
"""

from __future__ import annotations

import numpy as np


ENTROPY_SPIKE_SIGMA = 1.5


# ---------------------------------------------------------------------------
# Per-vector (entropy or L2 norm) features
# ---------------------------------------------------------------------------
def spike_positions(vec: np.ndarray, sigma: float = ENTROPY_SPIKE_SIGMA) -> np.ndarray:
    if len(vec) == 0:
        return np.array([], dtype=int)
    threshold = vec.mean() + sigma * (vec.std() + 1e-12)
    return np.where(vec > threshold)[0]


def zone_spike_ratio(vec: np.ndarray, zone: str) -> float:
    """Same convention as Qwen reference:
       early: <15% | mid: 15-60% | late: >60%
       Returns 0.5 (neutral) when no spikes or empty vec.
    """
    if len(vec) == 0:
        return 0.5
    sp = spike_positions(vec)
    if len(sp) == 0:
        return 0.5
    n = len(vec)
    if zone == "early":
        mask = sp < n * 0.15
    elif zone == "mid":
        mask = (sp >= n * 0.15) & (sp < n * 0.60)
    elif zone == "late":
        mask = sp >= n * 0.60
    else:
        raise ValueError(f"unknown zone: {zone}")
    return float(mask.sum()) / float(len(sp))


def delta_features(vec: np.ndarray) -> dict[str, float]:
    if len(vec) < 2:
        return dict(max_positive_delta=0.0, max_negative_delta=0.0,
                    delta_variance=0.0, delta_mean=0.0, delta_std=0.0,
                    convergence_slope=0.0, mid_delta_sum=0.0, n_delta_spikes=0.0)
    d = np.diff(vec)
    n = len(d)
    mid_mask = (np.arange(n) >= n * 0.15) & (np.arange(n) < n * 0.60)
    return dict(
        max_positive_delta=float(d.max()),
        max_negative_delta=float(d.min()),
        delta_variance=float(d.var()),
        delta_mean=float(d.mean()),
        delta_std=float(d.std()),
        convergence_slope=float(np.polyfit(np.arange(n), d, 1)[0]) if n >= 2 else 0.0,
        mid_delta_sum=float(d[mid_mask].sum()) if mid_mask.any() else 0.0,
        n_delta_spikes=float((np.abs(d) > (np.abs(d).mean() + ENTROPY_SPIKE_SIGMA * (d.std() + 1e-12))).sum()),
    )


def entropy_features(ent_vec: np.ndarray) -> dict[str, float]:
    """Entropy-vec features matching Qwen names exactly (so they're comparable)."""
    if len(ent_vec) == 0:
        return dict(mean_entropy=0.0, max_entropy=0.0, first_token_entropy=0.0,
                    entropy_std=0.0, n_spike_tokens=0, n_tokens=0)
    sp = spike_positions(ent_vec)
    feats = dict(
        mean_entropy=float(ent_vec.mean()),
        max_entropy=float(ent_vec.max()),
        first_token_entropy=float(ent_vec[0]),
        entropy_std=float(ent_vec.std()),
        n_spike_tokens=int(len(sp)),
        n_tokens=int(len(ent_vec)),
    )
    feats.update(delta_features(ent_vec))
    # Qwen-style delta renames
    feats["delta_variance_qwen"] = feats["delta_variance"]
    return feats


def layer_norm_features(norms: list[float], layer_idx: int) -> dict[str, float]:
    """Per-layer L2-norm features. Keys are prefixed with L{idx}_ to avoid collision.

    Computes everything Qwen computed + everything Qwen 'discovered but not wired':
    mid_spike_ratio, late_spike_ratio, early_spike_ratio, plus all delta features.
    """
    vec = np.array(norms, dtype=np.float32) if len(norms) else np.array([], dtype=np.float32)
    if len(vec) == 0:
        return {}
    out = {
        f"L{layer_idx}_mid_spike_ratio": zone_spike_ratio(vec, "mid"),
        f"L{layer_idx}_late_spike_ratio": zone_spike_ratio(vec, "late"),
        f"L{layer_idx}_early_spike_ratio": zone_spike_ratio(vec, "early"),
        f"L{layer_idx}_mean_norm": float(vec.mean()),
        f"L{layer_idx}_std_norm": float(vec.std()),
        f"L{layer_idx}_n_spikes": int(len(spike_positions(vec))),
    }
    d = delta_features(vec)
    for k, v in d.items():
        out[f"L{layer_idx}_{k}"] = v
    return out


def all_layers_features(per_layer_norms: dict[int, list[float]]) -> dict[str, float]:
    """Compute the full per-layer feature dict for every captured layer."""
    out = {}
    for idx in sorted(per_layer_norms.keys()):
        out.update(layer_norm_features(per_layer_norms[idx], idx))
    return out


def partial_layer_features(norms: list[float], layer_idx: int) -> dict[str, float]:
    """Same as layer_norm_features but for a prefix of the trajectory.

    Used during early-exit: compute features from norms seen so far.
    """
    return layer_norm_features(norms, layer_idx)
