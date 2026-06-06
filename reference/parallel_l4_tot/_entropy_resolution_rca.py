"""Entropy resolution analysis: do correct paths show different spike resolution patterns?

Key features from the data:
- spike_positions: WHERE in the generation sequence entropy spiked
- delta_std: HOW MUCH entropy varies between consecutive tokens
- max_positive_delta / max_negative_delta: largest entropy jumps
- L12_late_spike_ratio: fraction of spikes in the last 1/3 of generation
- L10_mid_spike_ratio: fraction of spikes in the middle 1/3 of generation

The fusion hypothesis: at fusion points (combining info), entropy spikes.
Correct paths RESOLVE these spikes (high delta_std = alternating effort/resolution).
Wrong paths GLOSS OVER (low delta_std = smooth but wrong generation).
"""

import json
import numpy as np

records = []
with open("results/bench_max_tokens_512/benchmark_parallel.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        records.append(json.loads(line))

TARGET = [103, 105, 110, 112, 136, 146, 147]

print("=== ENTROPY RESOLUTION PATTERNS ===\n")

# Per-problem comparison
for r in records:
    if r["question_idx"] not in TARGET:
        continue
    idx = r["question_idx"]
    gt = r.get("ground_truth")
    paths = r.get("parallel", {}).get("paths", [])

    print(f"Q{idx} (GT={gt}):")
    for i, p in enumerate(paths):
        ans = p.get("answer")
        feat = p.get("features", {})
        score = p.get("score", {})
        is_correct = ans is not None and abs(ans - gt) < 0.01
        is_selected = score.get("selected", False)

        delta_std = feat.get("delta_std", 0)
        n_spikes = feat.get("n_spike_tokens", 0)
        spikes = feat.get("spike_positions", [])
        L12 = feat.get("L12_late_spike_ratio", 0)
        L10 = feat.get("L10_mid_spike_ratio", 0)
        mean_ent = feat.get("mean_entropy", 0)

        tag = "CORRECT" if is_correct else "wrong"
        sel = " SELECTED" if is_selected else ""

        # Spike clustering: are spikes concentrated or spread?
        if spikes and len(spikes) > 2:
            spike_gaps = [spikes[j+1] - spikes[j] for j in range(len(spikes)-1)]
            mean_gap = np.mean(spike_gaps)
            std_gap = np.std(spike_gaps)
            # Coefficient of variation of spike gaps — low = uniform spacing, high = clustered
            gap_cv = std_gap / max(mean_gap, 1)
        else:
            gap_cv = 0

        print(f"  P{i}: ans={ans} ({tag}{sel})")
        print(f"       delta_std={delta_std:.3f} n_spikes={n_spikes} gap_cv={gap_cv:.2f} L12={L12:.3f} L10={L10:.3f}")
    print()

# Aggregate across ALL 50 problems: does delta_std discriminate?
print("=== AGGREGATE DELTA_STD ANALYSIS ===\n")

correct_delta_std = []
wrong_delta_std = []
correct_L12 = []
wrong_L12 = []

for r in records:
    gt = r.get("ground_truth")
    paths = r.get("parallel", {}).get("paths", [])

    for p in paths:
        ans = p.get("answer")
        feat = p.get("features", {})
        is_correct = ans is not None and abs(ans - gt) < 0.01

        if is_correct:
            correct_delta_std.append(feat.get("delta_std", 0))
            correct_L12.append(feat.get("L12_late_spike_ratio", 0))
        else:
            wrong_delta_std.append(feat.get("delta_std", 0))
            wrong_L12.append(feat.get("L12_late_spike_ratio", 0))

print(f"              Correct (n={len(correct_delta_std)})    Wrong (n={len(wrong_delta_std)})")
print(f"delta_std:    {np.mean(correct_delta_std):.3f} ± {np.std(correct_delta_std):.3f}      {np.mean(wrong_delta_std):.3f} ± {np.std(wrong_delta_std):.3f}")
print(f"L12_late:     {np.mean(correct_L12):.3f} ± {np.std(correct_L12):.3f}      {np.mean(wrong_L12):.3f} ± {np.std(wrong_L12):.3f}")

# Within-problem: does the correct path have higher delta_std?
print(f"\nWithin-problem delta_std comparison:")
d_higher = 0
d_lower = 0
d_same = 0
for r in records:
    gt = r.get("ground_truth")
    paths = r.get("parallel", {}).get("paths", [])
    c_ds = [p.get("features", {}).get("delta_std", 0) for p in paths
            if p.get("answer") is not None and abs(p["answer"] - gt) < 0.01]
    w_ds = [p.get("features", {}).get("delta_std", 0) for p in paths
            if p.get("answer") is None or abs(p["answer"] - gt) >= 0.01]
    if c_ds and w_ds:
        if np.mean(c_ds) > np.mean(w_ds):
            d_higher += 1
        elif np.mean(c_ds) < np.mean(w_ds):
            d_lower += 1
        else:
            d_same += 1

print(f"  Correct has higher delta_std: {d_higher}/47")
print(f"  Correct has lower delta_std:  {d_lower}/47")
print(f"  Same:                         {d_same}/47")

# Would selecting by highest delta_std work?
print(f"\n=== SELECT BY HIGHEST DELTA_STD ===\n")
correct_selected_by_deltastd = 0
wrong_selected_by_deltastd = 0
for r in records:
    gt = r.get("ground_truth")
    paths = r.get("parallel", {}).get("paths", [])
    best_idx = max(range(len(paths)),
                   key=lambda i: paths[i].get("features", {}).get("delta_std", 0))
    ans = paths[best_idx].get("answer")
    if ans is not None and abs(ans - gt) < 0.01:
        correct_selected_by_deltastd += 1
    else:
        wrong_selected_by_deltastd += 1

print(f"Selecting path with highest delta_std: {correct_selected_by_deltastd}/{correct_selected_by_deltastd + wrong_selected_by_deltastd} correct")

# Combine: highest delta_std AND #### present AND entropy < 2.0
print(f"\n=== COMBINED: delta_std highest among valid (ent<2.0, #### present) ===\n")
correct_combined = 0
wrong_combined = 0
for r in records:
    gt = r.get("ground_truth")
    paths = r.get("parallel", {}).get("paths", [])

    valid = [(i, p) for i, p in enumerate(paths)
             if p.get("features", {}).get("mean_entropy", 0) < 2.0
             and "####" in p.get("text", "")]

    if valid:
        best_idx, best_p = max(valid, key=lambda x: x[1].get("features", {}).get("delta_std", 0))
    else:
        best_idx = max(range(len(paths)),
                       key=lambda i: paths[i].get("features", {}).get("delta_std", 0))
        best_p = paths[best_idx]

    ans = best_p.get("answer")
    if ans is not None and abs(ans - gt) < 0.01:
        correct_combined += 1
    else:
        wrong_combined += 1

print(f"Combined selector: {correct_combined}/{correct_combined + wrong_combined} correct")
print(f"(current selector: 32/47 correct)")
