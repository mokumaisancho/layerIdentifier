"""TDD tests for bootstrap.py — 95% CIs on AUROC."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def easy_dataset():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 5))
    X[:n // 2] += 2.0  # shift positives
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return X, y


def test_bootstrap_ci_returns_dict_with_required_keys(easy_dataset):
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    result = bootstrap_ci(X, y, n_resamples=200, seed=42)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "point", "half_width"):
        assert key in result


def test_bootstrap_ci_lower_lt_point_lt_upper(easy_dataset):
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    result = bootstrap_ci(X, y, n_resamples=200, seed=42)
    assert result["lower"] <= result["point"] <= result["upper"], \
        "point estimate must lie inside CI"


def test_bootstrap_ci_half_width_positive(easy_dataset):
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    result = bootstrap_ci(X, y, n_resamples=200, seed=42)
    assert result["half_width"] > 0


def test_bootstrap_ci_reproducible(easy_dataset):
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    r1 = bootstrap_ci(X, y, n_resamples=200, seed=42)
    r2 = bootstrap_ci(X, y, n_resamples=200, seed=42)
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]


def test_bootstrap_ci_easy_data_high_point(easy_dataset):
    """Easy data → point AUROC near 1.0."""
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    result = bootstrap_ci(X, y, n_resamples=500, seed=42)
    assert result["point"] >= 0.95


def test_bootstrap_ci_half_width_for_1000_resamples(easy_dataset):
    """For N=100 with reasonable effect size, half-width should be ≤ 0.10 with 1000 resamples."""
    from src.bootstrap import bootstrap_ci
    X, y = easy_dataset
    result = bootstrap_ci(X, y, n_resamples=1000, seed=42)
    assert result["half_width"] <= 0.10, f"CI too wide: {result['half_width']}"
