"""Qwen early-exit threshold simulation.

Same logic as /tmp/probe_threshold_tune.py but for Qwen3.5-4B.
No GPU needed — uses existing captures.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

sys.path.insert(0, '/Users/apple/Downloads/Py/layerIdentifier')
from src.early_exit_bench import PROBE_CONFIGS
from src.features import partial_layer_features

caps = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/qwen3.5-4B-nothinking/captures.json').read_text())
print(f"Loaded {len(caps)} Qwen captures")

cfg = PROBE_CONFIGS['qwen']
feature_names = [f"L{ly}_{fk}" for ly in cfg['layers'] for fk in cfg['features']]

# Train probe on full features
X = np.array([[c.get('layer_features', {}).get(fn, 0.0) for fn in feature_names] for c in caps])
y = np.array([1 if c['final_correct'] else 0 for c in caps])
probe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'))])
probe.fit(X, y)

# Sanity check probe quality
from sklearn.model_selection import cross_val_score
cv_auc = cross_val_score(probe, X, y, cv=5, scoring='roc_auc')
print(f"Probe 5-fold CV AUROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# Simulate per-token probe eval
print("\nSimulating per-token probe evaluations...")
sim_data = []
for c in caps:
    label = 1 if c['final_correct'] else 0
    # Use first target layer for n_tokens reference
    n_tokens = len(c.get('per_layer_norms', {}).get(str(cfg['layers'][0]), []))
    if n_tokens == 0: continue
    p_per_token = []
    for t in range(1, n_tokens + 1):
        row = []
        for layer in cfg['layers']:
            full_norms = c.get('per_layer_norms', {}).get(str(layer), [])
            partial = full_norms[:t]
            feats = partial_layer_features(partial, layer)
            for fk in cfg['features']:
                row.append(feats.get(f"L{layer}_{fk}", 0.0))
        x_t = np.array(row).reshape(1, -1)
        p = float(probe.predict_proba(x_t)[0, 1])
        p_per_token.append(p)
    sim_data.append({'label': label, 'n_tokens': n_tokens, 'p': p_per_token})

print(f"Simulated {len(sim_data)} Qwen problems")

# Sweep thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
abort_consecs = [1, 2, 3, 4]
cutoffs = [3, 5, 8, 10]

print(f"\n{'cutoff':<8}{'thresh':<8}{'abortN':<8}{'abort%':<8}{'FALSE%':<8}{'sens%':<8}{'tokSaved%':<10}{'netWin':<10}")
print("-" * 75)

results = []
for cutoff in cutoffs:
    for thresh in thresholds:
        for ac in abort_consecs:
            tp_aborts = fp_aborts = tn = tp = 0
            tok_full = tok_used = 0
            for d in sim_data:
                p_traj = d['p']; label = d['label']; n_tok = d['n_tokens']
                tok_full += n_tok
                abort_token = None
                cons_low = 0
                for k, p in enumerate(p_traj):
                    if (k+1) < cutoff: continue
                    if p < thresh:
                        cons_low += 1
                        if cons_low >= ac:
                            abort_token = k+1
                            break
                    else:
                        cons_low = 0
                if abort_token is not None:
                    tok_used += abort_token
                    if label == 0: tp_aborts += 1
                    else: fp_aborts += 1
                else:
                    tok_used += n_tok
                    if label == 0: tn += 1
                    else: tp += 1
            n = len(sim_data)
            n_aborted = tp_aborts + fp_aborts
            n_bw = tp_aborts + tn
            n_bc = tp + fp_aborts
            abort_rate = n_aborted / n
            false_abort_rate = fp_aborts / max(1, n_aborted)
            sensitivity = tp_aborts / max(1, n_bw)
            tok_saved_pct = (tok_full - tok_used) / tok_full
            net_win = tok_saved_pct - 2 * (fp_aborts / n)

            print(f"{cutoff:<8}{thresh:<8}{ac:<8}{abort_rate*100:<8.1f}{false_abort_rate*100:<8.1f}{sensitivity*100:<8.1f}{tok_saved_pct*100:<10.1f}{net_win*100:<10.2f}")

            results.append({
                'cutoff': cutoff, 'threshold': thresh, 'abort_consec': ac,
                'abort_rate': abort_rate, 'false_abort_rate': false_abort_rate,
                'sensitivity': sensitivity, 'tokens_saved_pct': tok_saved_pct,
                'net_win': net_win,
                'tp_aborts': tp_aborts, 'fp_aborts': fp_aborts,
            })

best = max(results, key=lambda r: r['net_win'])
print(f"\n=== BEST QWEN CONFIG ===")
print(f"cutoff={best['cutoff']}, threshold={best['threshold']}, abort_consec={best['abort_consec']}")
print(f"  Abort rate:        {best['abort_rate']*100:.1f}%")
print(f"  False-abort rate:  {best['false_abort_rate']*100:.1f}%")
print(f"  Sensitivity:       {best['sensitivity']*100:.1f}%")
print(f"  Tokens saved:      {best['tokens_saved_pct']*100:.1f}%")
print(f"  Net win:           {best['net_win']*100:.2f}")

# Find zero-false-abort configs
zero_fa = [r for r in results if r['fp_aborts'] == 0]
if zero_fa:
    best_zf = max(zero_fa, key=lambda r: r['tokens_saved_pct'])
    print(f"\n=== BEST ZERO-FALSE-ABORT QWEN CONFIG ===")
    print(f"cutoff={best_zf['cutoff']}, threshold={best_zf['threshold']}, abort_consec={best_zf['abort_consec']}")
    print(f"  Abort rate:        {best_zf['abort_rate']*100:.1f}%")
    print(f"  False aborts:      0")
    print(f"  Sensitivity:       {best_zf['sensitivity']*100:.1f}%")
    print(f"  Tokens saved:      {best_zf['tokens_saved_pct']*100:.1f}%")

# Save
out = Path('/Users/apple/Downloads/Py/layerIdentifier/results/qwen3.5-4B-earlyexit-sim')
out.mkdir(exist_ok=True)
(out / 'threshold_sweep.json').write_text(json.dumps({
    'probe_cv_auc': {'mean': float(cv_auc.mean()), 'std': float(cv_auc.std())},
    'best_config': best,
    'best_zero_false_abort': best_zf if zero_fa else None,
    'all_configs': results,
}, indent=2))
print(f"\nWrote {out}/threshold_sweep.json")
