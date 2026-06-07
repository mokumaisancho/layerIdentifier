"""TDD tests for loader.py — load captures.json into feature matrix."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_captures(tmp_path):
    """Build a small captures.json fixture: 20 problems, 5 layer features each."""
    problems = []
    rng = np.random.default_rng(42)
    for i in range(20):
        # 2 layers × 3 features = 6 features
        layer_features = {
            f"L{L}_{feat}": float(rng.normal(0, 1))
            for L in (0, 1)
            for feat in ("mean_norm", "std_norm", "n_spikes")
        }
        problems.append({
            "problem_id": i,
            "seed": 7,
            "ground_truth": str(i * 2),
            "model_answer_text": str(i * 2) if i % 2 == 0 else str(i * 2 + 1),
            "final_correct": bool(i % 2 == 0),
            "n_tokens": 30,
            "tps": 50.0,
            "entropy_features": {},
            "layer_features": layer_features,
            "per_layer_norms": {},
        })
    path = tmp_path / "captures.json"
    path.write_text(json.dumps(problems))
    return path


def test_load_features_returns_ndarray(synthetic_captures):
    from src.loader import load_features
    X, y, names = load_features(str(synthetic_captures))
    assert isinstance(X, np.ndarray), "X must be np.ndarray"
    assert isinstance(y, np.ndarray), "y must be np.ndarray"


def test_load_features_correct_shape(synthetic_captures):
    from src.loader import load_features
    X, y, names = load_features(str(synthetic_captures))
    assert X.ndim == 2, "X must be 2D"
    assert X.shape[0] == 20, "T must be 20"
    assert X.shape[1] == 6, "N must be 6 (2 layers × 3 features)"
    assert y.shape == (20,), "y must be (T,)"


def test_load_features_label_values(synthetic_captures):
    from src.loader import load_features
    X, y, names = load_features(str(synthetic_captures))
    assert set(np.unique(y)).issubset({0, 1}), "y must be binary"
    assert (y == 0).sum() > 0, "must have negatives"
    assert (y == 1).sum() > 0, "must have positives"


def test_load_features_names_length(synthetic_captures):
    from src.loader import load_features
    X, y, names = load_features(str(synthetic_captures))
    assert isinstance(names, list), "names must be list"
    assert len(names) == X.shape[1], "names length must equal N"


def test_load_features_real_gemma_shape():
    """Real captures.json: T=150, N=588 for Gemma."""
    path = Path("/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json")
    if not path.exists():
        pytest.skip("Gemma captures.json not available")
    from src.loader import load_features
    X, y, names = load_features(str(path))
    assert X.shape == (150, 588), f"Gemma must be (150, 588), got {X.shape}"
    assert y.shape == (150,)
    assert len(names) == 588
