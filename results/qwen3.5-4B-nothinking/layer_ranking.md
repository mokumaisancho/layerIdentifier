# Layer Ranking — Qwen3.5-4B-MLX-4bit

- Transformer layers: **32**
- Problem-seed runs: **150**

Methodology: identical to parallel_l4_tot (Qwen reference).
Per-layer L2 norms captured per generated token; features computed
with same spike thresholds (mean + 1.5σ) and zone boundaries (early/mid/late).

## Top-30 features by AUROC

| Rank | Layer | Feature | AUROC | Direction | |d| |
|---|---|---|---|---|---|
| 1 | L15 | L15_n_spikes | 0.889 | - | 1.86 |
| 2 | L7 | L7_std_norm | 0.873 | + | 1.59 |
| 3 | L7 | L7_convergence_slope | 0.872 | - | 0.98 |
| 4 | L5 | L5_convergence_slope | 0.870 | - | 1.14 |
| 5 | L6 | L6_convergence_slope | 0.869 | - | 1.18 |
| 6 | L4 | L4_convergence_slope | 0.864 | - | 1.07 |
| 7 | L2 | L2_delta_mean | 0.864 | - | 1.21 |
| 8 | L14 | L14_n_spikes | 0.857 | - | 1.44 |
| 9 | L7 | L7_delta_variance | 0.855 | + | 1.29 |
| 10 | L7 | L7_delta_std | 0.855 | + | 1.39 |
| 11 | L24 | L24_n_delta_spikes | 0.855 | - | 1.83 |
| 12 | L26 | L26_convergence_slope | 0.854 | - | 0.73 |
| 13 | L14 | L14_convergence_slope | 0.849 | - | 0.82 |
| 14 | L27 | L27_convergence_slope | 0.848 | - | 0.86 |
| 15 | L3 | L3_convergence_slope | 0.848 | - | 0.99 |
| 16 | L12 | L12_convergence_slope | 0.843 | - | 0.83 |
| 17 | L13 | L13_convergence_slope | 0.842 | - | 0.83 |
| 18 | L28 | L28_convergence_slope | 0.840 | - | 0.90 |
| 19 | L23 | L23_convergence_slope | 0.840 | - | 0.83 |
| 20 | L25 | L25_convergence_slope | 0.839 | - | 0.75 |
| 21 | L1 | L1_n_spikes | 0.839 | - | 1.37 |
| 22 | L29 | L29_convergence_slope | 0.837 | - | 0.92 |
| 23 | L8 | L8_convergence_slope | 0.836 | - | 0.97 |
| 24 | L11 | L11_convergence_slope | 0.833 | - | 0.82 |
| 25 | L9 | L9_convergence_slope | 0.831 | - | 0.83 |
| 26 | L12 | L12_mean_norm | 0.831 | - | 1.24 |
| 27 | L6 | L6_n_spikes | 0.826 | - | 1.46 |
| 28 | L17 | L17_mean_norm | 0.826 | - | 1.23 |
| 29 | L15 | L15_convergence_slope | 0.825 | - | 0.85 |
| 30 | L10 | L10_convergence_slope | 0.824 | - | 0.77 |

## Per-layer best feature

| Layer | Best feature | AUROC | Direction |
|---|---|---|---|
| L0 | L0_n_spikes | 0.774 | - |
| L1 | L1_n_spikes | 0.839 | - |
| L2 | L2_delta_mean | 0.864 | - |
| L3 | L3_convergence_slope | 0.848 | - |
| L4 | L4_convergence_slope | 0.864 | - |
| L5 | L5_convergence_slope | 0.870 | - |
| L6 | L6_convergence_slope | 0.869 | - |
| L7 | L7_std_norm | 0.873 | + |
| L8 | L8_convergence_slope | 0.836 | - |
| L9 | L9_convergence_slope | 0.831 | - |
| L10 | L10_convergence_slope | 0.824 | - |
| L11 | L11_convergence_slope | 0.833 | - |
| L12 | L12_convergence_slope | 0.843 | - |
| L13 | L13_convergence_slope | 0.842 | - |
| L14 | L14_n_spikes | 0.857 | - |
| L15 | L15_n_spikes | 0.889 | - |
| L16 | L16_convergence_slope | 0.817 | - |
| L17 | L17_mean_norm | 0.826 | - |
| L18 | L18_convergence_slope | 0.818 | - |
| L19 | L19_convergence_slope | 0.802 | - |
| L20 | L20_mean_norm | 0.734 | - |
| L21 | L21_mean_norm | 0.764 | - |
| L22 | L22_n_spikes | 0.805 | - |
| L23 | L23_convergence_slope | 0.840 | - |
| L24 | L24_n_delta_spikes | 0.855 | - |
| L25 | L25_convergence_slope | 0.839 | - |
| L26 | L26_convergence_slope | 0.854 | - |
| L27 | L27_convergence_slope | 0.848 | - |
| L28 | L28_convergence_slope | 0.840 | - |
| L29 | L29_convergence_slope | 0.837 | - |
| L30 | L30_convergence_slope | 0.792 | - |
| L31 | L31_n_spikes | 0.764 | - |