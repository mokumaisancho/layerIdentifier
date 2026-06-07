"""TDD tests for probe.py — cross-validated LR probe."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def separable_data():
    """Easy binary problem: positives at +x direction, negatives at -x."""
    rng = np.random.default_rng(0)
    n = 60
    X_pos = rng.normal(2, 0.5, (n // 2, 5))
    X_neg = rng.normal(-2, 0.5, (n // 2, 5))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return X, y


@pytest.fixture
def random_data():
    """Random labels (AUROC should be ~0.5)."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 5))
    y = rng.integers(0, 2, 60)
    return X, y


def test_cross_val_auroc_returns_dict_with_required_keys(separable_data):
    from src.probe import cross_val_auroc
    X, y = separable_data
    result = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert isinstance(result, dict)
    for key in ("mean", "std", "per_fold"):
        assert key in result


def test_cross_val_auroc_in_valid_range(separable_data):
    from src.probe import cross_val_auroc
    X, y = separable_data
    result = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert 0.5 <= result["mean"] <= 1.0
    assert all(0.5 <= a <= 1.0 for a in result["per_fold"])


def test_cross_val_auroc_high_for_separable(separable_data):
    from src.probe import cross_val_auroc
    X, y = separable_data
    result = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert result["mean"] >= 0.95, f"separable data must give ≥0.95, got {result['mean']}"


def test_cross_val_auroc_near_half_for_random(random_data):
    """Random labels → AUROC near 0.5. Allow wide range due to small-sample variance."""
    from src.probe import cross_val_auroc
    X, y = random_data
    result = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert 0.20 <= result["mean"] <= 0.80, f"random data must be near 0.5, got {result['mean']}"


def test_cross_val_auroc_reproducible(separable_data):
    from src.probe import cross_val_auroc
    X, y = separable_data
    r1 = cross_val_auroc(X, y, n_folds=5, seed=42)
    r2 = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert r1["mean"] == r2["mean"], "same seed must give same result"
    assert r1["per_fold"] == r2["per_fold"]


def test_cross_val_auroc_per_fold_length(separable_data):
    from src.probe import cross_val_auroc
    X, y = separable_data
    result = cross_val_auroc(X, y, n_folds=5, seed=42)
    assert len(result["per_fold"]) == 5
