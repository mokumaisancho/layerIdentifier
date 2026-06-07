"""TDD tests for eigendecomp.py — Ledoit-Wolf + MP filtering + projection."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_std_matrix():
    """T=200, N=50 standardized matrix with embedded 3-dim signal structure."""
    rng = np.random.default_rng(42)
    T, N = 200, 50
    # 3 signal directions
    V_signal = rng.normal(0, 1, (N, 3))
    Q, _ = np.linalg.qr(V_signal)  # orthonormalize
    V_signal = Q
    z = rng.normal(0, 1, (T, 3)) * 5.0  # signal scores, large magnitude
    noise = rng.normal(0, 1, (T, N))
    X = z @ V_signal.T + noise
    # Standardize
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X_std, T, N


def test_standardize_zero_mean_unit_std():
    from src.eigendecomp import standardize
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, (100, 10))
    X_std = standardize(X)
    assert np.allclose(X_std.mean(0), 0, atol=1e-7), "mean must be 0"
    assert np.allclose(X_std.std(0), 1, atol=1e-7), "std must be 1"


def test_compute_signal_decomp_returns_dict_with_required_keys(synthetic_std_matrix):
    from src.eigendecomp import compute_signal_decomp
    X_std, T, N = synthetic_std_matrix
    # MP lambda_max for Q = N/T = 0.25
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    assert isinstance(result, dict)
    for key in ("eigvals", "eigvecs", "signal_mask", "shrinkage"):
        assert key in result, f"missing key: {key}"


def test_eigenvalues_sorted_descending(synthetic_std_matrix):
    from src.eigendecomp import compute_signal_decomp
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    eigvals = result["eigvals"]
    assert np.all(np.diff(eigvals) <= 1e-10), "eigvals must be descending"


def test_eigenvectors_orthogonal(synthetic_std_matrix):
    """Eigenvectors of symmetric matrix must be orthonormal."""
    from src.eigendecomp import compute_signal_decomp
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    V = result["eigvecs"]
    # V is (N, N); columns should be orthonormal
    gram = V.T @ V
    assert np.allclose(gram, np.eye(N), atol=1e-6), "V must be orthonormal"


def test_signal_mask_at_least_3_for_synthetic_data(synthetic_std_matrix):
    """We planted 3 strong signal directions; signal_mask must capture ≥3."""
    from src.eigendecomp import compute_signal_decomp
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    n_signal = int(result["signal_mask"].sum())
    assert n_signal >= 3, f"expected ≥3 signal modes, got {n_signal}"


def test_shrinkage_in_valid_range(synthetic_std_matrix):
    """Ledoit-Wolf shrinkage must be in [0, 1]."""
    from src.eigendecomp import compute_signal_decomp
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    s = result["shrinkage"]
    assert 0.0 <= s <= 1.0, f"shrinkage must be in [0,1], got {s}"


def test_project_onto_signal_reduces_dimension(synthetic_std_matrix):
    from src.eigendecomp import compute_signal_decomp, project_onto_signal
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    V_signal = result["eigvecs"][:, result["signal_mask"]]
    X_proj = project_onto_signal(X_std, V_signal)
    assert X_proj.shape == (T, V_signal.shape[1])
    assert V_signal.shape[1] < N, "must reduce dimension"


def test_save_and_load_cache_roundtrip(tmp_path, synthetic_std_matrix):
    from src.eigendecomp import compute_signal_decomp, save_cache, load_cache
    X_std, T, N = synthetic_std_matrix
    Q = N / T
    mp_lambda_max = (1 + np.sqrt(Q)) ** 2
    result = compute_signal_decomp(X_std, mp_lambda_max)
    cache_path = tmp_path / "cache.npz"
    save_cache(str(cache_path), result, feature_names=[f"f{i}" for i in range(N)])
    loaded = load_cache(str(cache_path))
    assert np.allclose(loaded["eigvals"], result["eigvals"])
    assert np.allclose(loaded["eigvecs"], result["eigvecs"])
    assert loaded["feature_names"] == [f"f{i}" for i in range(N)]
