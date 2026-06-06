# Layer Ranking — gemma-4-E4B-it-MLX-8bit

- Transformer layers: **42**
- Problem-seed runs: **150**

Methodology: identical to parallel_l4_tot (Qwen reference).
Per-layer L2 norms captured per generated token; features computed
with same spike thresholds (mean + 1.5σ) and zone boundaries (early/mid/late).

## Top-30 features by AUROC

| Rank | Layer | Feature | AUROC | Direction | |d| |
|---|---|---|---|---|---|
| 1 | L6 | L6_convergence_slope | 0.807 | - | 0.90 |
| 2 | L15 | L15_max_positive_delta | 0.791 | - | 0.98 |
| 3 | L6 | L6_mean_norm | 0.777 | + | 0.96 |
| 4 | L41 | L41_n_spikes | 0.766 | - | 1.13 |
| 5 | L28 | L28_n_spikes | 0.763 | - | 0.99 |
| 6 | L21 | L21_mid_spike_ratio | 0.759 | - | 1.20 |
| 7 | L27 | L27_mid_delta_sum | 0.749 | + | 0.71 |
| 8 | L2 | L2_convergence_slope | 0.744 | - | 0.67 |
| 9 | L34 | L34_mean_norm | 0.743 | - | 0.69 |
| 10 | L23 | L23_mid_delta_sum | 0.743 | + | 0.88 |
| 11 | L11 | L11_mid_delta_sum | 0.741 | + | 0.86 |
| 12 | L7 | L7_mean_norm | 0.741 | + | 0.85 |
| 13 | L10 | L10_mid_delta_sum | 0.741 | + | 0.88 |
| 14 | L35 | L35_mid_delta_sum | 0.740 | + | 0.87 |
| 15 | L39 | L39_mid_delta_sum | 0.740 | + | 0.94 |
| 16 | L38 | L38_mid_delta_sum | 0.739 | + | 0.91 |
| 17 | L4 | L4_convergence_slope | 0.739 | - | 0.52 |
| 18 | L36 | L36_mid_delta_sum | 0.737 | + | 0.85 |
| 19 | L17 | L17_mid_spike_ratio | 0.736 | - | 1.09 |
| 20 | L19 | L19_delta_variance | 0.736 | - | 0.85 |
| 21 | L19 | L19_delta_std | 0.736 | - | 0.87 |
| 22 | L24 | L24_convergence_slope | 0.735 | - | 0.35 |
| 23 | L22 | L22_mean_norm | 0.732 | + | 0.87 |
| 24 | L10 | L10_delta_mean | 0.732 | + | 0.59 |
| 25 | L22 | L22_mid_delta_sum | 0.731 | + | 0.78 |
| 26 | L11 | L11_delta_mean | 0.730 | + | 0.56 |
| 27 | L23 | L23_delta_variance | 0.730 | + | 0.88 |
| 28 | L23 | L23_delta_std | 0.730 | + | 0.92 |
| 29 | L40 | L40_mid_delta_sum | 0.729 | + | 0.93 |
| 30 | L34 | L34_mid_delta_sum | 0.729 | + | 0.81 |

## Per-layer best feature

| Layer | Best feature | AUROC | Direction |
|---|---|---|---|
| L0 | L0_n_spikes | 0.653 | - |
| L1 | L1_n_spikes | 0.718 | - |
| L2 | L2_convergence_slope | 0.744 | - |
| L3 | L3_convergence_slope | 0.725 | - |
| L4 | L4_convergence_slope | 0.739 | - |
| L5 | L5_mean_norm | 0.716 | + |
| L6 | L6_convergence_slope | 0.807 | - |
| L7 | L7_mean_norm | 0.741 | + |
| L8 | L8_convergence_slope | 0.688 | - |
| L9 | L9_delta_mean | 0.727 | + |
| L10 | L10_mid_delta_sum | 0.741 | + |
| L11 | L11_mid_delta_sum | 0.741 | + |
| L12 | L12_delta_mean | 0.716 | + |
| L13 | L13_delta_mean | 0.709 | + |
| L14 | L14_delta_mean | 0.721 | + |
| L15 | L15_max_positive_delta | 0.791 | - |
| L16 | L16_n_spikes | 0.709 | - |
| L17 | L17_mid_spike_ratio | 0.736 | - |
| L18 | L18_max_negative_delta | 0.682 | + |
| L19 | L19_delta_variance | 0.736 | - |
| L20 | L20_mid_delta_sum | 0.683 | + |
| L21 | L21_mid_spike_ratio | 0.759 | - |
| L22 | L22_mean_norm | 0.732 | + |
| L23 | L23_mid_delta_sum | 0.743 | + |
| L24 | L24_convergence_slope | 0.735 | - |
| L25 | L25_convergence_slope | 0.715 | - |
| L26 | L26_mid_delta_sum | 0.729 | + |
| L27 | L27_mid_delta_sum | 0.749 | + |
| L28 | L28_n_spikes | 0.763 | - |
| L29 | L29_mean_norm | 0.684 | - |
| L30 | L30_mean_norm | 0.722 | - |
| L31 | L31_mean_norm | 0.719 | - |
| L32 | L32_mean_norm | 0.706 | - |
| L33 | L33_mean_norm | 0.725 | - |
| L34 | L34_mean_norm | 0.743 | - |
| L35 | L35_mid_delta_sum | 0.740 | + |
| L36 | L36_mid_delta_sum | 0.737 | + |
| L37 | L37_mid_delta_sum | 0.728 | + |
| L38 | L38_mid_delta_sum | 0.739 | + |
| L39 | L39_mid_delta_sum | 0.740 | + |
| L40 | L40_mid_delta_sum | 0.729 | + |
| L41 | L41_n_spikes | 0.766 | - |