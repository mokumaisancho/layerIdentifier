"""Novelty probe -- detects recalled vs novel reasoning from hidden states.

Trains a LogisticRegression on 12-dim aggregated features extracted from
captured hidden states. Uses balanced class weights and C=0.01 regularization
for small-sample robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    """Compute skewness of a flattened array."""
    if len(x) < 3:
        return 0.0
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of a flattened array."""
    if len(x) < 4:
        return 0.0
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4) - 3.0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Single-sample prediction from the novelty probe."""

    recall_score: float  # P(recall) in [0, 1]
    feature_vector: np.ndarray  # 12-dim features used
    feature_names: list[str]


# ---------------------------------------------------------------------------
# NoveltyProbe
# ---------------------------------------------------------------------------

class NoveltyProbe:
    """Probe that classifies hidden-state captures as recall or novel.

    Features are 12 dimensional: mean / std / skewness / kurtosis computed
    on layers 0, 15, and 31 of a captured hidden-state tensor.
    """

    FEATURE_NAMES: list[str] = [
        "L0_mean", "L0_std", "L0_skew", "L0_kurtosis",
        "L15_mean", "L15_std", "L15_skew", "L15_kurtosis",
        "L31_mean", "L31_std", "L31_skew", "L31_kurtosis",
    ]
    FEATURE_LAYERS: list[int] = [0, 15, 31]

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.model = LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self._trained = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(capture_path: str | Path) -> np.ndarray:
        """Extract 12-dim aggregated features from a capture ``.npz`` file.

        Returns an array of shape ``(12,)`` with
        mean / std / skewness / kurtosis per layer.
        """
        data = np.load(capture_path, allow_pickle=True)
        features: list[float] = []
        for layer_idx in NoveltyProbe.FEATURE_LAYERS:
            key = f"layer_{layer_idx}"
            if key not in data:
                raise KeyError(
                    f"Layer {layer_idx} not found in {capture_path}. "
                    f"Available keys: {list(data.keys())}"
                )
            arr = data[key]  # shape: (n_tokens, hidden_dim)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            flat = arr.flatten()
            features.extend([
                float(np.mean(flat)),
                float(np.std(flat)),
                _skewness(flat),
                _kurtosis(flat),
            ])
        return np.array(features, dtype=np.float32)

    @staticmethod
    def extract_features_from_arrays(layer_arrays: dict[int, np.ndarray]) -> np.ndarray:
        """Extract 12-dim features from in-memory layer arrays.

        Parameters
        ----------
        layer_arrays: dict mapping layer index to ndarray of shape (n_tokens, hidden_dim)

        Returns
        -------
        ndarray of shape (12,) with mean/std/skew/kurtosis per layer.
        """
        features: list[float] = []
        for layer_idx in NoveltyProbe.FEATURE_LAYERS:
            if layer_idx not in layer_arrays:
                raise KeyError(f"Layer {layer_idx} not in layer_arrays. Available: {list(layer_arrays.keys())}")
            arr = layer_arrays[layer_idx]
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            flat = arr.flatten().astype(np.float32)
            features.extend([
                float(np.mean(flat)),
                float(np.std(flat)),
                _skewness(flat),
                _kurtosis(flat),
            ])
        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
        """Train the probe and return evaluation metrics.

        Parameters
        ----------
        features:
            ``(n_samples, 12)`` feature matrix.
        labels:
            ``(n_samples,)`` label vector.  ``1`` = recall, ``0`` = novel.

        Returns
        -------
        dict with ``loo_auroc``, ``n_samples``, ``n_recall``, ``n_novel``,
        and ``feature_importance``.
        """
        self.model.fit(features, labels)
        self._trained = True

        # LOO-AUROC evaluation
        loo = LeaveOneOut()
        probs = np.zeros(len(labels))
        for train_idx, test_idx in loo.split(features):
            m = LogisticRegression(
                C=0.01,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            m.fit(features[train_idx], labels[train_idx])
            probs[test_idx] = m.predict_proba(features[test_idx])[:, 1]

        auroc = roc_auc_score(labels, probs)

        # Feature importance (absolute coefficient magnitudes)
        importance = dict(
            zip(self.FEATURE_NAMES, np.abs(self.model.coef_[0]).tolist())
        )

        return {
            "loo_auroc": float(auroc),
            "n_samples": int(len(labels)),
            "n_recall": int(labels.sum()),
            "n_novel": int((1 - labels).sum()),
            "feature_importance": importance,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> ProbeResult:
        """Predict recall score for a single sample.

        Parameters
        ----------
        features:
            ``(12,)`` feature vector.

        Raises
        ------
        RuntimeError
            If the probe has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError("Probe not trained. Call train() first.")
        prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        return ProbeResult(
            recall_score=float(prob),
            feature_vector=features,
            feature_names=self.FEATURE_NAMES,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained probe to a joblib file."""
        joblib.dump(
            {"model": self.model, "threshold": self.threshold},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> NoveltyProbe:
        """Load a trained probe from a joblib file."""
        data = joblib.load(path)
        probe = cls(threshold=data["threshold"])
        probe.model = data["model"]
        probe._trained = True
        return probe
