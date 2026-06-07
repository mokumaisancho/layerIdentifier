"""Cross-validated L2 logistic regression probe for AUROC."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def cross_val_auroc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Stratified k-fold CV with L2 LR. Returns mean/std/per-fold AUROC.

    If a fold has only one class, its AUROC is set to 0.5.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            per_fold.append(0.5)
            continue
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        scores = clf.predict_proba(X[te])[:, 1]
        per_fold.append(float(roc_auc_score(y[te], scores)))
    return {
        "mean": float(np.mean(per_fold)),
        "std": float(np.std(per_fold)),
        "per_fold": per_fold,
    }
