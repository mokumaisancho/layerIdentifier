"""Hypothesis C test: does the probe just read residual-stream magnitude?

If yes, a probe using ONLY ||residual at layer N|| (a single scalar per token)
should match the full L6 probe AUROC. If no, the L6 probe reads something more
specific than magnitude.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def auroc(scores, labels):
    s = np.array(scores); l = np.array(labels)
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    rs = np.sum([(neg < x).sum() + 0.5 * (neg == x).sum() for x in pos])
    a = rs / (len(pos) * len(neg))
    return max(a, 1 - a)


caps = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json').read_text())
labels = np.array([1 if c['final_correct'] else 0 for c in caps])
n = len(caps)
print(f"N: {n} (correct={labels.sum()}, wrong={n - labels.sum()})")

# === Per-layer single-feature AUROC ===
print("\n=== Best single feature per layer ===")
print(f"{'Layer':<8}{'Best Feature':<30}{'AUROC':<10}")
print("-" * 50)
for layer in range(7):
    feat_names = ['mid_spike_ratio', 'mean_norm', 'std_norm', 'n_spikes',
                  'max_positive_delta', 'max_negative_delta',
                  'convergence_slope', 'mid_delta_sum', 'n_delta_spikes']
    best_fn = None
    best_auc = 0
    for fn in feat_names:
        key = f"L{layer}_{fn}"
        vals = np.array([c.get('layer_features', {}).get(key, 0.0) for c in caps])
        auc = auroc(vals, labels)
        if auc > best_auc:
            best_auc = auc
            best_fn = fn
    print(f"L{layer:<7}{best_fn:<30}{best_auc:<10.3f}")

# === Multi-feature probe (5-fold CV) per layer ===
print("\n=== Multi-feature probe (5-fold CV) per layer ===")
print(f"{'Layer':<8}{'CV AUROC':<20}")
print("-" * 30)
pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=0.1, max_iter=1000))])
feat_names_4 = ['convergence_slope', 'mean_norm', 'std_norm', 'n_spikes']

cv_aucs = {}
for layer in range(7):
    feature_keys = [f"L{layer}_{fn}" for fn in feat_names_4]
    rows = []
    for c in caps:
        row = [c.get('layer_features', {}).get(fk, 0.0) for fk in feature_keys]
        rows.append(row)
    X = np.array(rows)
    cv_auc = cross_val_score(pipe, X, labels, cv=5, scoring='roc_auc')
    cv_aucs[layer] = cv_auc
    print(f"L{layer:<7}{cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# === L0-L5 stacked vs L6 alone ===
print("\n=== Does L6 add info BEYOND L0-L5? ===")
pre_rows = []
for c in caps:
    row = []
    for l in range(6):
        for fn in feat_names_4:
            row.append(c.get('layer_features', {}).get(f"L{l}_{fn}", 0.0))
    pre_rows.append(row)
X_pre = np.array(pre_rows)

l6_rows = []
for c in caps:
    row = [c.get('layer_features', {}).get(f"L6_{fn}", 0.0) for fn in feat_names_4]
    l6_rows.append(row)
X_l6 = np.array(l6_rows)

X_combined = np.hstack([X_pre, X_l6])

cv_pre = cross_val_score(pipe, X_pre, labels, cv=5, scoring='roc_auc')
cv_l6 = cross_val_score(pipe, X_l6, labels, cv=5, scoring='roc_auc')
cv_combined = cross_val_score(pipe, X_combined, labels, cv=5, scoring='roc_auc')

print(f"L0-L5 only (24 features):  {cv_pre.mean():.3f} ± {cv_pre.std():.3f}")
print(f"L6 only (4 features):      {cv_l6.mean():.3f} ± {cv_l6.std():.3f}")
print(f"L0-L5 + L6 combined (28):  {cv_combined.mean():.3f} ± {cv_combined.std():.3f}")
print(f"L6 marginal over L0-L5:    Δ={cv_combined.mean() - cv_pre.mean():+.3f}")

print("\n=== VERDICT (Hypothesis C) ===")
if cv_l6.mean() > cv_pre.mean() + 0.05:
    print(f"REFUTED: L6 ({cv_l6.mean():.3f}) carries UNIQUE signal beyond L0-L5 magnitude ({cv_pre.mean():.3f})")
    print("The L6 probe reads something MORE than just magnitude.")
elif cv_pre.mean() > cv_l6.mean() + 0.05:
    print(f"CONFIRMED: L0-L5 magnitude ({cv_pre.mean():.3f}) is BETTER than L6 ({cv_l6.mean():.3f})")
    print("L6 is a degraded version of earlier magnitude signal.")
else:
    print(f"AMBIGUOUS: L0-L5 ({cv_pre.mean():.3f}) ≈ L6 ({cv_l6.mean():.3f})")
    print("Could be that L6 reads the same magnitude as L0-L5, just in a slightly different form.")
