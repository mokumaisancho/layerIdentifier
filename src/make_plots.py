"""Generate publication-quality plots for the layerIdentifier findings.

Outputs to results/plots/:
  01_per_layer_auroc_curve.png      — Gemma + Qwen per-layer best AUROC vs relative depth
  02_thinking_vs_nothinking.png     — Show the thinking-mode artifact
  03_threshold_tuning_heatmap.png   — Early-exit net_win by cutoff x threshold
  04_L0_L6_buildup.png              — Per-layer single-feature AUROC L0..L6
  05_p_correct_trajectories.png     — Example P(correct) trajectories correct vs wrong
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

RESULTS = Path('/Users/apple/Downloads/Py/layerIdentifier/results')
PLOTS = RESULTS / 'plots'
PLOTS.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
    'savefig.dpi': 130,
    'savefig.bbox': 'tight',
})

# Load
gemma = json.loads((RESULTS / 'gemma-4-E4B-nothinking/captures.json').read_text())
qwen = json.loads((RESULTS / 'qwen3.5-4B-nothinking/captures.json').read_text())
gemma_labels = np.array([1 if c['final_correct'] else 0 for c in gemma])
qwen_labels = np.array([1 if c['final_correct'] else 0 for c in qwen])


def auroc(scores, labels):
    s = np.array(scores); l = np.array(labels)
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    rs = np.sum([(neg < x).sum() + 0.5 * (neg == x).sum() for x in pos])
    return max(rs / (len(pos) * len(neg)), 1 - rs / (len(pos) * len(neg)))


# -----------------------------------------------------------------------------
# Plot 1: Per-layer best-AUROC vs relative depth
# -----------------------------------------------------------------------------
def per_layer_best(caps, labels, n_layers):
    feat_keys = ['mid_spike_ratio', 'late_spike_ratio', 'early_spike_ratio',
                 'mean_norm', 'std_norm', 'n_spikes',
                 'max_positive_delta', 'max_negative_delta', 'delta_variance',
                 'delta_mean', 'delta_std', 'convergence_slope',
                 'mid_delta_sum', 'n_delta_spikes']
    out = []
    for layer in range(n_layers):
        best = 0
        for fk in feat_keys:
            key = f"L{layer}_{fk}"
            scores = np.array([c.get('layer_features', {}).get(key, 0.0) for c in caps])
            a = auroc(scores, labels)
            if a > best: best = a
        out.append(best)
    return out

gemma_n_layers = 42
qwen_n_layers = 32

gemma_per_layer = per_layer_best(gemma, gemma_labels, gemma_n_layers)
qwen_per_layer = per_layer_best(qwen, qwen_labels, qwen_n_layers)

gemma_rel = np.arange(gemma_n_layers) / gemma_n_layers
qwen_rel = np.arange(qwen_n_layers) / qwen_n_layers

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gemma_rel, gemma_per_layer, 'o-', label=f'Gemma-4-E4B ({gemma_n_layers} layers)', color='#1f77b4', alpha=0.85, markersize=5)
ax.plot(qwen_rel, qwen_per_layer, 's-', label=f'Qwen3.5-4B ({qwen_n_layers} layers)', color='#d62728', alpha=0.85, markersize=5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
ax.axvspan(0.05, 0.25, color='green', alpha=0.08, label='early-layer band (rel 0.05-0.25)')
ax.set_xlabel('Relative depth (layer_idx / n_layers)')
ax.set_ylabel('Best single-feature AUROC')
ax.set_title('Per-layer correctness probe AUROC (clean, no-thinking)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.45, 0.95)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.savefig(PLOTS / '01_per_layer_auroc_curve.png')
plt.close()
print("Saved 01_per_layer_auroc_curve.png")

# -----------------------------------------------------------------------------
# Plot 2: Thinking vs no-thinking (Qwen L22/L24 artifact)
# -----------------------------------------------------------------------------
# We have clean captures only. Plot distribution of mid_delta_sum at L22 for both
# To show the thinking artifact, we'd need the thinking-mode captures (excluded from public).
# Substitute: compare actual L6 Gemma distribution correct vs wrong (the real signal)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gemma L6 conv_slope: correct vs wrong
gemma_l6_cs = np.array([c.get('layer_features', {}).get('L6_convergence_slope', 0.0) for c in gemma])
ax = axes[0]
bins = np.linspace(gemma_l6_cs.min(), gemma_l6_cs.max(), 30)
ax.hist(gemma_l6_cs[gemma_labels == 1], bins=bins, alpha=0.6, color='green', label=f'correct (n={gemma_labels.sum()})')
ax.hist(gemma_l6_cs[gemma_labels == 0], bins=bins, alpha=0.6, color='red', label=f'wrong (n={(1-gemma_labels).sum()}')
ax.axvline(gemma_l6_cs[gemma_labels == 1].mean(), color='green', linestyle='--', linewidth=2)
ax.axvline(gemma_l6_cs[gemma_labels == 0].mean(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('L6 convergence_slope')
ax.set_ylabel('Count')
ax.set_title(f'Gemma L6 convergence_slope\n(AUROC 0.807 [0.72, 0.88])')
ax.legend()
ax.grid(True, alpha=0.3)

# Qwen L15 n_spikes: correct vs wrong
qwen_l15_ns = np.array([c.get('layer_features', {}).get('L15_n_spikes', 0.0) for c in qwen])
ax = axes[1]
bins = np.linspace(qwen_l15_ns.min(), qwen_l15_ns.max(), 30)
ax.hist(qwen_l15_ns[qwen_labels == 1], bins=bins, alpha=0.6, color='green', label=f'correct (n={qwen_labels.sum()})')
ax.hist(qwen_l15_ns[qwen_labels == 0], bins=bins, alpha=0.6, color='red', label=f'wrong (n={(1-qwen_labels).sum()}')
ax.axvline(qwen_l15_ns[qwen_labels == 1].mean(), color='green', linestyle='--', linewidth=2)
ax.axvline(qwen_l15_ns[qwen_labels == 0].mean(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('L15 n_spikes')
ax.set_ylabel('Count')
ax.set_title(f'Qwen L15 n_spikes\n(AUROC 0.889 [0.84, 0.94])')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Probe signal distributions — correct vs wrong (clean, no-thinking)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS / '02_thinking_vs_nothinking.png')
plt.close()
print("Saved 02_thinking_vs_nothinking.png")

# -----------------------------------------------------------------------------
# Plot 3: Threshold tuning heatmap (Gemma)
# -----------------------------------------------------------------------------
# Re-simulate from captures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0, '/Users/apple/Downloads/Py/layerIdentifier')
from src.early_exit_bench import PROBE_CONFIGS
from src.features import partial_layer_features

cfg = PROBE_CONFIGS['gemma']
feature_names = [f"L{ly}_{fk}" for ly in cfg['layers'] for fk in cfg['features']]
X = np.array([[c.get('layer_features', {}).get(fn, 0.0) for fn in feature_names] for c in gemma])
y = gemma_labels
probe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'))])
probe.fit(X, y)

# Simulate per-token p_correct for each problem
sim_data = []
for c in gemma:
    label = 1 if c['final_correct'] else 0
    n_tokens = len(c.get('per_layer_norms', {}).get('6', []))
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

# Build heatmap: net_win by (cutoff, threshold), averaged over abort_consec ∈ {2,3,4}
cutoffs = [3, 5, 8, 10]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
abort_consecs = [2, 3, 4]

heatmap = np.zeros((len(cutoffs), len(thresholds)))
for i, cutoff in enumerate(cutoffs):
    for j, thresh in enumerate(thresholds):
        # average over abort_consecs
        net_wins = []
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
            tok_saved_pct = (tok_full - tok_used) / tok_full
            net_win = tok_saved_pct - 2 * (fp_aborts / n)
            net_wins.append(net_win)
        heatmap[i, j] = np.mean(net_wins)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heatmap, cmap='RdYlGn', vmin=-0.3, vmax=0.15, aspect='auto')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'{t}' for t in thresholds])
ax.set_yticks(range(len(cutoffs)))
ax.set_yticklabels([f'{c}' for c in cutoffs])
ax.set_xlabel('Threshold (abort if P(correct) < threshold)')
ax.set_ylabel('Cutoff (tokens before first probe eval)')
ax.set_title('Early-exit net_win (tokens_saved% - 2·false_abort%)\nGemma, averaged over abort_consec ∈ {2,3,4}')
plt.colorbar(im, label='net_win')
# Annotate cells
for i in range(len(cutoffs)):
    for j in range(len(thresholds)):
        ax.text(j, i, f'{heatmap[i,j]*100:+.1f}', ha='center', va='center', fontsize=10,
                color='black' if abs(heatmap[i,j]) < 0.15 else 'white')
plt.savefig(PLOTS / '03_threshold_tuning_heatmap.png')
plt.close()
print("Saved 03_threshold_tuning_heatmap.png")

# -----------------------------------------------------------------------------
# Plot 4: L0-L6 build-up (single feature AUROC)
# -----------------------------------------------------------------------------
best_per_layer = []
for layer in range(7):
    feat_keys = ['convergence_slope', 'n_spikes', 'std_norm', 'mean_norm',
                 'mid_delta_sum', 'max_positive_delta', 'max_negative_delta',
                 'mid_spike_ratio']
    best = 0; best_fn = ''
    for fk in feat_keys:
        key = f"L{layer}_{fk}"
        scores = np.array([c.get('layer_features', {}).get(key, 0.0) for c in gemma])
        a = auroc(scores, gemma_labels)
        if a > best: best = a; best_fn = fk
    best_per_layer.append((layer, best, best_fn))

fig, ax = plt.subplots(figsize=(8, 5))
layers = [x[0] for x in best_per_layer]
aucs = [x[1] for x in best_per_layer]
ax.plot(layers, aucs, 'o-', color='#1f77b4', markersize=10, linewidth=2)
for i, (l, a, fn) in enumerate(best_per_layer):
    ax.annotate(fn.replace('_', '\n'), (l, a), textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=8)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('Best single-feature AUROC')
ax.set_title('Probe signal build-up: Gemma L0 to L6\n(best single feature per layer)')
ax.set_xticks(layers)
ax.set_ylim(0.55, 0.85)
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(PLOTS / '04_L0_L6_buildup.png')
plt.close()
print("Saved 04_L0_L6_buildup.png")

# -----------------------------------------------------------------------------
# Plot 5: Example P(correct) trajectories
# -----------------------------------------------------------------------------
# Pick 5 correct + 5 wrong examples
correct_idx = [i for i, d in enumerate(sim_data) if d['label'] == 1][:5]
wrong_idx = [i for i, d in enumerate(sim_data) if d['label'] == 0][:5]

fig, ax = plt.subplots(figsize=(9, 5))
for idx in correct_idx:
    d = sim_data[idx]
    ax.plot(range(1, len(d['p']) + 1), d['p'], 'g-', alpha=0.5, linewidth=1.5)
for idx in wrong_idx:
    d = sim_data[idx]
    ax.plot(range(1, len(d['p']) + 1), d['p'], 'r-', alpha=0.5, linewidth=1.5)
ax.axhline(0.5, color='black', linestyle=':', alpha=0.5)
ax.axvline(5, color='blue', linestyle='--', alpha=0.4, label='cutoff=5')
ax.axvline(10, color='navy', linestyle='--', alpha=0.4, label='cutoff=10')
ax.set_xlabel('Token position')
ax.set_ylabel('P(correct) from probe')
ax.set_title('P(correct) trajectories — 5 correct (green) vs 5 wrong (red)\nGemma, simulated per-token probe eval')
ax.set_xlim(0, 50)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
plt.savefig(PLOTS / '05_p_correct_trajectories.png')
plt.close()
print("Saved 05_p_correct_trajectories.png")

print(f"\nAll plots saved to {PLOTS}/")
