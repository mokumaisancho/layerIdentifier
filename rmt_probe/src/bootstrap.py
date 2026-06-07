"""Bootstrap 95% CIs on AUROC point estimate."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI on AUROC of L2 LR fit on full data.

    Resample (X, y) with replacement n_resamples times. For each, fit LR on
    the bootstrap sample and evaluate AUROC on the SAME bootstrap sample
    (percentile bootstrap — straightforward, matches Move #3 convention).

    Returns:
      lower: float — lower CI bound
      upper: float — upper CI bound
      point: float — point estimate (AUROC of LR fit on full data)
      half_width: float — (upper - lower) / 2
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    # Point estimate
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    point = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    aurocs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        clf_b = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf_b.fit(xb, yb)
        aurocs.append(float(roc_auc_score(yb, clf_b.predict_proba(xb)[:, 1])))

    if not aurocs:
        return {"lower": 0.5, "upper": 0.5, "point": point, "half_width": 0.0}

    alpha = (1 - ci) / 2
    lower = float(np.quantile(aurocs, alpha))
    upper = float(np.quantile(aurocs, 1 - alpha))
    return {
        "lower": lower,
        "upper": upper,
        "point": point,
        "half_width": (upper - lower) / 2.0,
    }
