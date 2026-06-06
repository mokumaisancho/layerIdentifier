"""Layer ranker — per-layer AUROC against correctness + ranked output.

For each layer × feature, compute AUROC(correct vs wrong) across the
problem-seed matrix. Rank by AUROC. Output a markdown report directly
comparable to reference/parallel_l4_tot/SIGNAL_CATALOG.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LayerRanking:
    layer: int
    feature: str
    auroc: float
    direction: str  # "+" = higher value → correct, "-" = higher value → wrong
    n_correct: int
    n_wrong: int
    effect_size: float  # mean(correct) - mean(wrong), in standardized units


def _auroc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    """Mann-Whitney U / AUROC. Returns (auroc, direction)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5, "+"
    # Mann-Whitney
    n = len(pos) + len(neg)
    rank_sum = 0
    for x in pos:
        rank_sum += (neg < x).sum() + 0.5 * (neg == x).sum()
    auroc_pos = rank_sum / (len(pos) * len(neg))
    direction = "+" if auroc_pos >= 0.5 else "-"
    # Always report AUROC in the "better-than-chance" direction
    auroc = max(auroc_pos, 1 - auroc_pos)
    return float(auroc), direction


def _cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    sp, sn = pos.std() + 1e-12, neg.std() + 1e-12
    pooled = np.sqrt(((len(pos) - 1) * sp ** 2 + (len(neg) - 1) * sn ** 2)
                     / max(1, len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / pooled)


def rank_layers_from_captures(captures: list[dict]) -> list[LayerRanking]:
    """Given a list of capture dicts (each from one problem-seed run), compute
    per-layer-feature AUROC and return a ranking.

    Each capture dict should have:
      - final_correct: bool
      - layer_features: dict[str, float] (from features.all_layers_features)
    """
    labels = np.array([1 if c["final_correct"] else 0 for c in captures], dtype=int)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    # Collect all layer-feature keys present
    feature_keys: set[str] = set()
    for c in captures:
        feature_keys.update(c.get("layer_features", {}).keys())

    rankings: list[LayerRanking] = []
    for k in feature_keys:
        vals = []
        for c in captures:
            v = c.get("layer_features", {}).get(k)
            vals.append(v if v is not None else np.nan)
        arr = np.array(vals, dtype=float)
        mask = ~np.isnan(arr)
        if mask.sum() < 5:
            continue
        auroc, direction = _auroc(arr[mask], labels[mask])
        d = _cohens_d(arr[mask], labels[mask])
        # Layer index is parsed from key prefix L<int>_
        try:
            layer_idx = int(k.split("_")[0][1:])
        except (ValueError, IndexError):
            layer_idx = -1
        rankings.append(LayerRanking(
            layer=layer_idx,
            feature=k,
            auroc=auroc,
            direction=direction,
            n_correct=n_pos,
            n_wrong=n_neg,
            effect_size=abs(d),
        ))

    rankings.sort(key=lambda r: (r.layer, -r.auroc))
    return rankings


def write_report(rankings: list[LayerRanking], out_path: Path, model_label: str,
                 n_layers: int, n_runs: int) -> None:
    """Write a markdown report comparable to SIGNAL_CATALOG.md."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Layer Ranking — {model_label}",
        "",
        f"- Transformer layers: **{n_layers}**",
        f"- Problem-seed runs: **{n_runs}**",
        "",
        "Methodology: identical to parallel_l4_tot (Qwen reference).",
        "Per-layer L2 norms captured per generated token; features computed",
        "with same spike thresholds (mean + 1.5σ) and zone boundaries (early/mid/late).",
        "",
        "## Top-30 features by AUROC",
        "",
        "| Rank | Layer | Feature | AUROC | Direction | |d| |",
        "|---|---|---|---|---|---|",
    ]
    top = sorted(rankings, key=lambda r: -r.auroc)[:30]
    for i, r in enumerate(top, 1):
        lines.append(f"| {i} | L{r.layer} | {r.feature} | {r.auroc:.3f} | {r.direction} | {r.effect_size:.2f} |")

    lines.extend([
        "",
        "## Per-layer best feature",
        "",
        "| Layer | Best feature | AUROC | Direction |",
        "|---|---|---|---|",
    ])
    best_per_layer: dict[int, LayerRanking] = {}
    for r in rankings:
        if r.layer < 0:
            continue
        if r.layer not in best_per_layer or r.auroc > best_per_layer[r.layer].auroc:
            best_per_layer[r.layer] = r
    for layer in sorted(best_per_layer.keys()):
        r = best_per_layer[layer]
        lines.append(f"| L{layer} | {r.feature} | {r.auroc:.3f} | {r.direction} |")

    out_path.write_text("\n".join(lines))


def write_json(rankings: list[LayerRanking], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in rankings], indent=2))
