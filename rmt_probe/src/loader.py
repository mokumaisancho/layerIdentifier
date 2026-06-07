"""Load captures.json into feature matrix (T, N) + labels + feature names."""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np


def load_features(path: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load captures.json and return (X, y, feature_names).

    X shape: (T, N) where T = number of problems, N = number of layer_features.
    y shape: (T,) — int labels (0/1) from final_correct.
    feature_names: sorted list of all keys observed across problems.
    """
    with open(path) as f:
        data = json.load(f)
    all_keys = sorted(set().union(*(ex["layer_features"].keys() for ex in data)))
    T, N = len(data), len(all_keys)
    X = np.zeros((T, N), dtype=np.float64)
    y = np.zeros(T, dtype=np.int64)
    for i, ex in enumerate(data):
        for j, k in enumerate(all_keys):
            X[i, j] = float(ex["layer_features"].get(k, 0.0))
        y[i] = int(bool(ex["final_correct"]))
    return X, y, all_keys
