"""Compare Gemma layer ranking to Qwen reference (parallel_l4_tot).

Qwen3.5-4B-MLX has 32 layers; Gemma-4-E4B has 42.
Direct index comparison is meaningless — we compare by RELATIVE DEPTH
(layer_idx / n_layers).

Qwen reference layers (from SIGNAL_CATALOG.md):
  L10  mid_spike_ratio   direction=-  (anti-signal)   rel_depth=0.312
  L12  late_spike_ratio  direction=+                  rel_depth=0.375
  L18  mid_delta_sum     direction=+  AUROC=0.593     rel_depth=0.562
  L22  mid_delta_sum     direction=+  AUROC=0.628     rel_depth=0.688
        convergence_slope              AUROC=0.622
  L24  max_negative_delta direction=+ AUROC=0.652     rel_depth=0.750
        convergence_slope              AUROC=0.623
        n_spikes (delta)               AUROC=0.616
        mid_delta_sum                  AUROC=0.604

For Gemma (n_layers=42), the equivalent relative-depth bands are:
  Qwen L10  ≈ Gemma L13
  Qwen L12  ≈ Gemma L16
  Qwen L18  ≈ Gemma L24
  Qwen L22  ≈ Gemma L29
  Qwen L24  ≈ Gemma L32

Question this script answers:
  Do Gemma's top-K layers by AUROC cluster in the same relative-depth bands
  as Qwen's, or somewhere else entirely?
"""

from __future__ import annotations

import json
from pathlib import Path

# (qwen_layer, qwen_n_layers, feature, auroc, direction)
QWEN_REF = [
    (10, 32, "mid_spike_ratio",     None,  "-"),
    (12, 32, "late_spike_ratio",    None,  "+"),
    (18, 32, "mid_delta_sum",       0.593, "+"),
    (22, 32, "mid_delta_sum",       0.628, "+"),
    (22, 32, "convergence_slope",   0.622, "+"),
    (24, 32, "max_negative_delta",  0.652, "+"),
    (24, 32, "convergence_slope",   0.623, "+"),
    (24, 32, "n_delta_spikes",      0.616, "+"),
    (24, 32, "mid_delta_sum",       0.604, "+"),
    (18, 32, "mid_delta_sum",       0.593, "+"),
]


def relative_depth(layer: int, n_layers: int) -> float:
    return layer / n_layers


def find_gemma_equivalent(qwen_layer: int, qwen_n_layers: int, gemma_n_layers: int) -> int:
    """Return the Gemma layer index closest to Qwen's relative depth."""
    rd = relative_depth(qwen_layer, qwen_n_layers)
    return round(rd * gemma_n_layers)


def load_ranking(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def top_per_layer(ranking: list[dict], top_k: int = 5) -> dict[int, list[dict]]:
    """Group by layer, return top-K features per layer by AUROC."""
    by_layer: dict[int, list[dict]] = {}
    for r in ranking:
        by_layer.setdefault(r["layer"], []).append(r)
    out = {}
    for layer, items in by_layer.items():
        items.sort(key=lambda r: -r["auroc"])
        out[layer] = items[:top_k]
    return out


def main(ranking_path: Path, n_layers_gemma: int = 42, out_path: Path | None = None) -> None:
    ranking = load_ranking(ranking_path)
    by_layer = top_per_layer(ranking, top_k=3)

    # Sort layers by their single best AUROC
    best_per_layer = [(l, items[0]) for l, items in by_layer.items()]
    best_per_layer.sort(key=lambda x: -x[1]["auroc"])

    # Top-10 layers by best AUROC
    print("\n## Gemma top-10 layers (best feature AUROC)\n")
    print("| Rank | Layer | Rel-depth | Best feature | AUROC | Direction |")
    print("|---|---|---|---|---|---|")
    for i, (layer, item) in enumerate(best_per_layer[:10], 1):
        rd = relative_depth(layer, n_layers_gemma)
        print(f"| {i} | L{layer} | {rd:.3f} | {item['feature']} | {item['auroc']:.3f} | {item['direction']} |")

    # For each Qwen reference layer, find Gemma's closest-layer AUROC for the same feature
    print("\n## Cross-model comparison (Qwen rel-depth → Gemma closest layer)\n")
    print("| Qwen layer | Qwen feature | Qwen AUROC | Qwen rel-depth | Gemma closest layer | Gemma same-feature AUROC | Gemma best-in-band AUROC |")
    print("|---|---|---|---|---|---|---|")
    for q_layer, q_n, q_feat, q_auroc, q_dir in QWEN_REF:
        rd = relative_depth(q_layer, q_n)
        gemma_layer_exact = find_gemma_equivalent(q_layer, q_n, n_layers_gemma)
        # Search the 3-layer band around gemma_layer_exact
        band = range(max(0, gemma_layer_exact - 2), min(n_layers_gemma, gemma_layer_exact + 3))
        # Same feature in band
        same_in_band = [
            r for l in band for r in by_layer.get(l, [])
            if r["feature"].endswith(q_feat) or r["feature"] == f"L{l}_{q_feat}"
        ]
        best_same = max((r["auroc"] for r in same_in_band), default=None)
        # Any feature in band
        any_in_band = [r for l in band for r in by_layer.get(l, [])]
        best_any = max((r["auroc"] for r in any_in_band), default=None)
        q_a_str = f"{q_auroc:.3f}" if q_auroc else "n/a"
        bs_str = f"{best_same:.3f}" if best_same is not None else "—"
        ba_str = f"{best_any:.3f}" if best_any is not None else "—"
        print(f"| L{q_layer} | {q_feat} | {q_a_str} | {rd:.3f} | L{gemma_layer_exact} (band L{min(band)}-L{max(band)}) | {bs_str} | {ba_str} |")

    # Verdict
    print("\n## Verdict\n")
    top5_layers = [l for l, _ in best_per_layer[:5]]
    top5_rel_depths = [relative_depth(l, n_layers_gemma) for l in top5_layers]
    # Qwen's top bands: 0.31-0.38 (L10/L12) and 0.56-0.75 (L18/L22/L24)
    qwen_band_a = any(0.20 <= rd <= 0.45 for rd in top5_rel_depths)
    qwen_band_b = any(0.50 <= rd <= 0.80 for rd in top5_rel_depths)
    if qwen_band_a and qwen_band_b:
        verdict = "**ARCHITECTURAL** — top Gemma layers fall in the same relative-depth bands as Qwen's L10-L12 and L18-L24. Phenomenon is not Qwen-specific."
    elif qwen_band_a or qwen_band_b:
        verdict = "**PARTIAL** — some overlap with Qwen bands but not both. Mixed evidence."
    else:
        verdict = "**QWEN-SPECIFIC** — Gemma's top layers fall outside Qwen's bands. Phenomenon may be architecture-specific."
    print(verdict)
    print(f"\nGemma top-5 layers: {top5_layers}  rel-depths: {[f'{r:.2f}' for r in top5_rel_depths]}")

    if out_path:
        out_path.write_text("\n".join([
            f"# Gemma vs Qwen comparison",
            f"",
            f"Gemma top-5 layers: {top5_layers}",
            f"Gemma top-5 relative depths: {[round(r, 3) for r in top5_rel_depths]}",
            f"",
            verdict,
        ]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True, help="path to layer_ranking.json")
    ap.add_argument("--n-layers", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(Path(args.ranking), n_layers_gemma=args.n_layers,
         out_path=Path(args.out) if args.out else None)
