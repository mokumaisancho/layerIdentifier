"""Temporal entropy feature extraction for L4 gate improvement.

Extracts dynamical features from spike_positions that the current probe discards.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def extract_temporal_features(pass1_features: dict[str, Any]) -> dict[str, float]:
    """Extract temporal/dynamical features from entropy spike data.

    Args:
        pass1_features: Dict with keys mean_entropy, max_entropy, first_token_entropy,
            entropy_std, n_spike_tokens, n_tokens, spike_positions

    Returns:
        Dict of temporal features that can be concatenated with scalar features.
    """
    spikes = pass1_features.get("spike_positions", [])
    n_tokens = pass1_features.get("n_tokens", 200)
    max_ent = pass1_features.get("max_entropy", 0)
    mean_ent = pass1_features.get("mean_entropy", 0)
    n_spikes = pass1_features.get("n_spike_tokens", 0)

    feats: dict[str, float] = {}

    # ── A1: Early / Late Spike Ratios ──
    # Single-pass count (was 3 passes)
    early_threshold = n_tokens * 0.15
    late_threshold = n_tokens * 0.60
    early = mid = late = 0
    for s in spikes:
        if s < early_threshold:
            early += 1
        elif s < late_threshold:
            mid += 1
        else:
            late += 1

    feats["early_spike_ratio"] = early / n_spikes if n_spikes > 0 else 0.0
    feats["mid_spike_ratio"] = mid / n_spikes if n_spikes > 0 else 0.0
    feats["late_spike_ratio"] = late / n_spikes if n_spikes > 0 else 0.0

    # ── A2: Recovery Indicator ──
    # If late spikes > early spikes, model may be recovering from early uncertainty
    feats["recovery_indicator"] = 1.0 if late > early else 0.0
    feats["late_minus_early"] = (late - early) / max(n_spikes, 1)

    # ── A3: Spike Amplitude Proxy ──
    # Big spikes vs many spikes: max_entropy / n_spikes
    feats["spike_amplitude_proxy"] = max_ent / max(n_spikes, 1)
    feats["spike_density"] = n_spikes / max(n_tokens, 1)

    # ── A4: Entropy Trajectory Shape ──
    # Concentration: are spikes clustered or spread out?
    if len(spikes) >= 2:
        spike_spread = np.std(spikes) if hasattr(np, 'std') else _manual_std(spikes)
        feats["spike_spread"] = float(spike_spread)
        # Coefficient of variation of spike positions
        feats["spike_cv"] = float(spike_spread / np.mean(spikes)) if np.mean(spikes) > 0 else 0.0
    else:
        feats["spike_spread"] = 0.0
        feats["spike_cv"] = 0.0

    # ── A5: Önden-style Temporal Entropy Dynamics ──
    # Divide generation into 4 quarters, estimate entropy per quarter from spikes
    q_size = n_tokens / 4
    quarter_counts = [0, 0, 0, 0]
    for s in spikes:
        q = min(int(s / q_size), 3)
        quarter_counts[q] += 1

    # Normalize by quarter length to get spike density
    for i, cnt in enumerate(quarter_counts):
        feats[f"q{i+1}_spike_density"] = cnt / max(q_size, 1)

    # Trend: entropy increasing or decreasing across generation?
    # Fit linear trend to quarter densities
    x = np.array([0, 1, 2, 3])
    y = np.array([quarter_counts[i] / max(q_size, 1) for i in range(4)])
    if len(set(y)) > 1:
        slope = _linear_regression_slope(x, y)
        feats["entropy_trend_slope"] = float(slope)
    else:
        feats["entropy_trend_slope"] = 0.0

    # ── A6: Confidence vs Uncertainty Mix ──
    # High max but low mean = few big doubts (impulse pattern)
    # High max and high mean = sustained uncertainty (pink pattern)
    feats["max_mean_ratio"] = max_ent / max(mean_ent, 0.001)
    feats["entropy_concentration"] = (max_ent - mean_ent) / max(mean_ent, 0.001)

    # ── A7: First-Token Signal ──
    # Very high first-token entropy often indicates the model is "thinking" about multiple approaches
    fte = pass1_features.get("first_token_entropy", 0)
    feats["first_token_relative"] = fte / max(max_ent, 0.001)

    return feats


def _manual_std(values: list[float]) -> float:
    """Compute std dev without numpy dependency."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute slope of best-fit line — vectorized."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = x - x_mean
    numerator = float(np.dot(dx, y - y_mean))
    denominator = float(np.dot(dx, dx))
    return numerator / denominator if denominator != 0 else 0.0


def load_cycle_data(pattern: str = "results/entropy_retry/cycle_*_details.jsonl") -> list[dict]:
    """Load all cycle detail entries with features."""
    entries = []
    for f in sorted(Path().glob(pattern)):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    if "pass1_features" in e and e["pass1_features"]:
                        e["_source"] = str(f)
                        entries.append(e)
    return entries


def build_temporal_dataset(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix and labels from cycle data.

    Returns:
        X: (n_samples, n_temporal_features) array
        y: (n_samples,) binary array — 1 = wrong, 0 = correct
        feature_names: list of temporal feature names
    """
    rows = []
    labels = []

    sample_feats = extract_temporal_features(entries[0]["pass1_features"])
    feature_names = list(sample_feats.keys())

    for e in entries:
        tf = extract_temporal_features(e["pass1_features"])
        rows.append([tf[name] for name in feature_names])
        labels.append(0 if e.get("pass1_correct") else 1)

    return np.array(rows), np.array(labels), feature_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/entropy_retry/cycle_*_details.jsonl")
    parser.add_argument("--save", default="results/temporal_features.jsonl")
    args = parser.parse_args()

    print("Loading cycle data...")
    entries = load_cycle_data(args.data)
    print(f"Loaded {len(entries)} entries")

    print("Extracting temporal features...")
    X, y, names = build_temporal_dataset(entries)
    print(f"Features: {names}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Quick analysis: which features separate correct from wrong?
    print("\n=== Feature Separation ===")
    correct_mask = y == 0
    wrong_mask = y == 1

    for i, name in enumerate(names):
        c_mean = np.mean(X[correct_mask, i])
        w_mean = np.mean(X[wrong_mask, i])
        diff = abs(c_mean - w_mean)
        print(f"{name:30} | correct={c_mean:8.4f} | wrong={w_mean:8.4f} | diff={diff:8.4f}")

    # Save
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        for e in entries:
            tf = extract_temporal_features(e["pass1_features"])
            record = {
                "question_idx": e.get("question_idx"),
                "cycle": e.get("cycle"),
                "pass1_correct": e.get("pass1_correct"),
                **tf,
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved to {args.save}")
