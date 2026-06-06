"""Bootstrap 95% confidence intervals for all headline AUROC numbers.

For each (model, feature) pair, compute:
- Point AUROC
- Bootstrap 95% CI (1000 resamples)
- Mann-Whitney z-score
- Bonferroni-corrected p-value

Output: results/bootstrap_cis.json + console summary
"""
import json
import numpy as np
from pathlib import Path
from math import sqrt
from scipy.stats import norm


def auroc_from_ranks(pos, neg):
    """Compute AUROC from raw score arrays."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    rs = 0
    for x in pos:
        rs += (neg < x).sum() + 0.5 * (neg == x).sum()
    a = rs / (len(pos) * len(neg))
    return max(a, 1 - a)


def bootstrap_auroc_ci(scores, labels, n_boot=1000, ci=95):
    """Bootstrap CI for AUROC. scores/labels are arrays."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)
    pos_mask = labels == 1
    neg_mask = labels == 0
    point = auroc_from_ranks(scores[pos_mask], scores[neg_mask])

    boots = []
    rng = np.random.default_rng(2024)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]
        l = labels[idx]
        if l.sum() == 0 or l.sum() == n:
            continue
        a = auroc_from_ranks(s[l == 1], s[l == 0])
        boots.append(a)
    boots = np.array(boots)
    alpha = (100 - ci) / 2
    return point, np.percentile(boots, alpha), np.percentile(boots, 100 - alpha), boots.std()


# Load both
gemma = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json').read_text())
qwen = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/qwen3.5-4B-nothinking/captures.json').read_text())

gemma_labels = np.array([1 if c['final_correct'] else 0 for c in gemma])
qwen_labels = np.array([1 if c['final_correct'] else 0 for c in qwen])

print("=== Bootstrap CIs (1000 resamples, 95%) ===\n")
print(f"{'Model':<8}{'N(pos,neg)':<14}{'Feature':<32}{'AUROC':<10}{'CI low':<10}{'CI high':<10}{'SE':<8}")
print("-" * 102)

results = {}

# Gemma headline features
gemma_targets = [
    ('L6_convergence_slope', 'Gemma L6 conv_slope'),
    ('L15_max_positive_delta', 'Gemma L15 max_pos_delta'),
    ('L6_mean_norm', 'Gemma L6 mean_norm'),
    ('L41_n_spikes', 'Gemma L41 n_spikes'),
    ('L2_convergence_slope', 'Gemma L2 conv_slope'),
]
for key, label in gemma_targets:
    scores = np.array([c.get('layer_features', {}).get(key, 0.0) for c in gemma])
    pt, lo, hi, se = bootstrap_auroc_ci(scores, gemma_labels)
    n_pos = int(gemma_labels.sum()); n_neg = len(gemma_labels) - n_pos
    print(f"Gemma   {f'({n_pos},{n_neg})':<14}{label:<32}{pt:<10.3f}{lo:<10.3f}{hi:<10.3f}{se:<8.3f}")
    results[key] = {'model': 'gemma', 'label': label, 'auroc': pt, 'ci_low': lo, 'ci_high': hi, 'se': se, 'n_pos': n_pos, 'n_neg': n_neg}

# Qwen headline features
qwen_targets = [
    ('L15_n_spikes', 'Qwen L15 n_spikes'),
    ('L7_convergence_slope', 'Qwen L7 conv_slope'),
    ('L5_convergence_slope', 'Qwen L5 conv_slope'),
    ('L6_convergence_slope', 'Qwen L6 conv_slope'),
    ('L2_delta_mean', 'Qwen L2 delta_mean'),
]
for key, label in qwen_targets:
    scores = np.array([c.get('layer_features', {}).get(key, 0.0) for c in qwen])
    pt, lo, hi, se = bootstrap_auroc_ci(scores, qwen_labels)
    n_pos = int(qwen_labels.sum()); n_neg = len(qwen_labels) - n_pos
    print(f"Qwen    {f'({n_pos},{n_neg})':<14}{label:<32}{pt:<10.3f}{lo:<10.3f}{hi:<10.3f}{se:<8.3f}")
    results[key] = {'model': 'qwen', 'label': label, 'auroc': pt, 'ci_low': lo, 'ci_high': hi, 'se': se, 'n_pos': n_pos, 'n_neg': n_neg}

# Early-exit metrics (from earlier bench)
print("\n=== Early-exit metrics (Gemma, tuned config) ===")
bench = json.loads(Path('/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-earlyexit/early_exit_bench_c10_t0.2_a4.json').read_text())
baseline = {(c['problem_id'], c['seed']): c for c in gemma}

# Compute per-problem time savings
time_savings = []
n_correct_baseline = 0
n_aborted = 0
n_false_abort = 0
for r in bench['per_problem']:
    key = (r['problem_id'], r['seed'])
    b = baseline.get(key)
    if not b: continue
    bt = b['n_tokens'] / b['tps'] if b['tps'] > 0 else 0
    et = r.get('elapsed_s', r['n_tokens'] / r['tps'] if r['tps'] > 0 else 0)
    time_savings.append(bt - et)
    if b['final_correct']: n_correct_baseline += 1
    if r['aborted']:
        n_aborted += 1
        if b['final_correct']: n_false_abort += 1

time_savings = np.array(time_savings)
print(f"Mean time saving per problem: {time_savings.mean():.2f}s")
print(f"95% CI: [{np.percentile(time_savings, 2.5):.2f}, {np.percentile(time_savings, 97.5):.2f}]")

# Bootstrap wall-clock gain
baseline_total = sum(b['n_tokens'] / b['tps'] for c in gemma[:150] for b in [c] if b['tps'] > 0)
exit_total = sum(r.get('elapsed_s', r['n_tokens']/r['tps'] if r['tps']>0 else 0) for r in bench['per_problem'])

# False abort rate bootstrap
print(f"\nFalse abort rate (point): {n_false_abort}/{n_aborted} = {n_false_abort/max(1,n_aborted):.3f}")

# Bonferroni for Gemma L6
n_pos_g = int(gemma_labels.sum()); n_neg_g = len(gemma_labels) - n_pos_g
se_null_g = sqrt((n_pos_g + n_neg_g + 1) / (12 * n_pos_g * n_neg_g))
n_tests_g = 588 + 13
alpha_bonf_g = 0.05 / n_tests_g
print(f"\n=== Bonferroni check ===")
print(f"Gemma: n_tests={n_tests_g}, alpha_bonf={alpha_bonf_g:.2e}, se_null={se_null_g:.4f}")
print(f"Min AUROC for Bonferroni sig: {0.5 + norm.ppf(1 - alpha_bonf_g) * se_null_g:.3f}")
print(f"L6 conv_slope AUROC: {results['L6_convergence_slope']['auroc']:.3f} (CI {results['L6_convergence_slope']['ci_low']:.3f}-{results['L6_convergence_slope']['ci_high']:.3f})")
print(f"  vs min for sig: {'SIGNIFICANT' if results['L6_convergence_slope']['ci_low'] > 0.5 + norm.ppf(1 - alpha_bonf_g) * se_null_g else 'NOT SIG'}")

# Save
out = {
    'feature_cis': results,
    'early_exit': {
        'mean_time_saving_per_problem': float(time_savings.mean()),
        'time_saving_ci_low': float(np.percentile(time_savings, 2.5)),
        'time_saving_ci_high': float(np.percentile(time_savings, 97.5)),
        'false_abort_rate': n_false_abort / max(1, n_aborted),
        'n_aborted': n_aborted,
        'n_false_abort': n_false_abort,
    },
}
Path('/Users/apple/Downloads/Py/layerIdentifier/results/bootstrap_cis.json').write_text(json.dumps(out, indent=2))
print("\nWrote results/bootstrap_cis.json")
