"""Ledoit-Wolf shrinkage + Marchenko-Pastur filtering + projection.

Eigendecomposition of standardized feature covariance, MP-based signal
eigenvalue selection, and projection onto the signal subspace.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score per column → mean 0, std 1."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-12
    return (X - mu) / sigma


def compute_signal_decomp(
    X_std: np.ndarray,
    mp_lambda_max: float,
) -> dict:
    """Compute Ledoit-Wolf shrunk covariance + eigendecomp + MP signal mask.

    Returns dict with:
      eigvals: (N,) sorted descending
      eigvecs: (N, N) columns are eigenvectors, sorted by eigvals descending
      signal_mask: (N,) bool — True if eigval > mp_lambda_max
      shrinkage: float in [0, 1] — Ledoit-Wolf shrinkage intensity
    """
    lw = LedoitWolf().fit(X_std)
    cov = lw.covariance_
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    signal_mask = eigvals > mp_lambda_max
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "signal_mask": signal_mask,
        "shrinkage": float(lw.shrinkage_),
    }


def project_onto_signal(
    X_std: np.ndarray,
    V_signal: np.ndarray,
) -> np.ndarray:
    """Project standardized features onto signal eigenvectors.

    X_proj = X_std @ V_signal. Shape (T, k).
    """
    return X_std @ V_signal


def save_cache(
    path: str,
    decomp: dict,
    feature_names: Optional[list[str]] = None,
) -> None:
    """Save eigendecomp result + feature names to .npz."""
    np.savez(
        path,
        eigvals=decomp["eigvals"],
        eigvecs=decomp["eigvecs"],
        signal_mask=decomp["signal_mask"],
        shrinkage=decomp["shrinkage"],
        feature_names=np.array(feature_names if feature_names is not None else [], dtype=object),
    )


def load_cache(path: str) -> dict:
    """Load eigvec cache."""
    data = np.load(path, allow_pickle=True)
    return {
        "eigvals": data["eigvals"],
        "eigvecs": data["eigvecs"],
        "signal_mask": data["signal_mask"],
        "shrinkage": float(data["shrinkage"]),
        "feature_names": list(data["feature_names"]),
    }
