"""Out-of-sample early-exit evaluation.

Split Gemma captures by problem_id into train/test. Train probe on train,
simulate early-exit on test. Compare to in-sample numbers.

If out-of-sample false-abort rate is similar to in-sample, the probe generalizes.
If much higher, the probe is overfit to specific problems.
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

caps = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json').read_text())
print(f"Loaded {len(caps)} Gemma captures")

# Split by problem_id (not by capture, to avoid leakage)
from collections import defaultdict
by_problem = defaultdict(list)
for c in caps:
    by_problem[c['problem_id']].append(c)
problem_ids = sorted(by_problem.keys())
print(f"Total unique problems: {len(problem_ids)}")

# 80/20 split by problem_id (50 problems → 40 train, 10 test)
np.random.seed(2024)
np.random.shuffle(problem_ids)
n_train = int(len(problem_ids) * 0.8)
train_pids = set(problem_ids[:n_train])
test_pids = set(problem_ids[n_train:])

train_caps = [c for c in caps if c['problem_id'] in train_pids]
test_caps = [c for c in caps if c['problem_id'] in test_pids]
print(f"Train: {len(train_caps)} captures ({len(train_pids)} problems)")
print(f"Test:  {len(test_caps)} captures ({len(test_pids)} problems)")

# Train probe on TRAIN only
cfg = PROBE_CONFIGS['gemma']
feature_names = [f"L{ly}_{fk}" for ly in cfg['layers'] for fk in cfg['features']]
X_train = np.array([[c.get('layer_features', {}).get(fn, 0.0) for fn in feature_names] for c in train_caps])
y_train = np.array([1 if c['final_correct'] else 0 for c in train_caps])
probe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'))])
probe.fit(X_train, y_train)

# Eval probe quality on TEST
X_test = np.array([[c.get('layer_features', {}).get(fn, 0.0) for fn in feature_names] for c in test_caps])
y_test = np.array([1 if c['final_correct'] else 0 for c in test_caps])
from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])
print(f"\nProbe AUROC on TEST: {test_auc:.3f}")
print(f"(in-sample CV was 0.951 ± 0.049)")

# Simulate per-token probe eval on TEST
def simulate(caps_list, probe, cfg, cutoff, thresh, ac):
    tp_aborts = fp_aborts = tn = tp = 0
    tok_full = tok_used = 0
    for c in caps_list:
        label = 1 if c['final_correct'] else 0
        n_tokens = len(c.get('per_layer_norms', {}).get(str(cfg['layers'][0]), []))
        if n_tokens == 0: continue
        tok_full += n_tokens
        abort_token = None
        cons_low = 0
        for k in range(n_tokens):
            t = k + 1
            if t < cutoff: continue
            row = []
            for layer in cfg['layers']:
                full_norms = c.get('per_layer_norms', {}).get(str(layer), [])
                partial = full_norms[:t]
                feats = partial_layer_features(partial, layer)
                for fk in cfg['features']:
                    row.append(feats.get(f"L{layer}_{fk}", 0.0))
            x_t = np.array(row).reshape(1, -1)
            p = float(probe.predict_proba(x_t)[0, 1])
            if p < thresh:
                cons_low += 1
                if cons_low >= ac:
                    abort_token = t
                    break
            else:
                cons_low = 0
        if abort_token is not None:
            tok_used += abort_token
            if label == 0: tp_aborts += 1
            else: fp_aborts += 1
        else:
            tok_used += n_tokens
            if label == 0: tn += 1
            else: tp += 1
    n = len(caps_list)
    n_aborted = tp_aborts + fp_aborts
    n_bw = tp_aborts + tn
    return {
        'n': n,
        'n_aborted': n_aborted,
        'abort_rate': n_aborted / n,
        'false_abort_rate': fp_aborts / max(1, n_aborted),
        'sensitivity': tp_aborts / max(1, n_bw),
        'tokens_saved_pct': (tok_full - tok_used) / tok_full,
        'tp_aborts': tp_aborts,
        'fp_aborts': fp_aborts,
    }

# Test the same configs as published
print(f"\n=== OUT-OF-SAMPLE EVAL (test problems only, N={len(test_caps)}) ===")
print(f"\n{'Config':<35}{'abort%':<10}{'FALSE%':<10}{'sens%':<10}{'tokSaved%':<12}")
print("-" * 80)

configs = [
    ('Aggressive c5_t0.5_a2', 5, 0.5, 2),
    ('Tuned c10_t0.2_a4', 10, 0.2, 4),
    ('Best sim c3_t0.2_a3', 3, 0.2, 3),
    ('Conservative c10_t0.1_a4', 10, 0.1, 4),
]

results = {}
for name, cutoff, thresh, ac in configs:
    r = simulate(test_caps, probe, cfg, cutoff, thresh, ac)
    print(f"{name:<35}{r['abort_rate']*100:<10.1f}{r['false_abort_rate']*100:<10.1f}{r['sensitivity']*100:<10.1f}{r['tokens_saved_pct']*100:<12.1f}")
    results[name] = r

# Also run on FULL set (in-sample comparison)
print(f"\n=== IN-SAMPLE (all 150 captures, same probe trained on train only) ===")
print(f"\n{'Config':<35}{'abort%':<10}{'FALSE%':<10}{'sens%':<10}{'tokSaved%':<12}")
print("-" * 80)
for name, cutoff, thresh, ac in configs:
    r = simulate(caps, probe, cfg, cutoff, thresh, ac)
    print(f"{name:<35}{r['abort_rate']*100:<10.1f}{r['false_abort_rate']*100:<10.1f}{r['sensitivity']*100:<10.1f}{r['tokens_saved_pct']*100:<12.1f}")

# Verdict
print(f"\n=== VERDICT ===")
tuned_oos = results['Tuned c10_t0.2_a4']
if tuned_oos['false_abort_rate'] < 0.20:
    print(f"Out-of-sample false-abort rate {tuned_oos['false_abort_rate']*100:.1f}% is acceptable (< 20%)")
    print(f"Probe GENERALIZES — not just memorizing training problems")
else:
    print(f"Out-of-sample false-abort rate {tuned_oos['false_abort_rate']*100:.1f}% is HIGH")
    print(f"Probe may be overfitting to specific problem patterns")

# Save
Path('/Users/apple/Downloads/Py/layerIdentifier/results/out_of_sample_eval.json').write_text(
    json.dumps({
        'split': {'n_train_problems': len(train_pids), 'n_test_problems': len(test_pids),
                  'n_train_captures': len(train_caps), 'n_test_captures': len(test_caps)},
        'test_probe_auroc': test_auc,
        'configs': results,
    }, indent=2))
print("\nWrote results/out_of_sample_eval.json")
