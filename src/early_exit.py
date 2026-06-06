"""Early-exit generation: train a probe, abort generation when P(correct) < threshold.

Two pieces:
  1. train_probe(captures) -> sklearn Pipeline (StandardScaler + LogisticRegression)
     Trained on the top-N features per architecture.
  2. generate_with_early_exit(...) -> generation loop that evaluates the probe
     at every token past a cutoff and aborts if P(correct) < threshold for K
     consecutive tokens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("MLX required") from e

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .layer_capture import install_layer_captures, restore_layers
from .features import partial_layer_features


@dataclass
class EarlyExitTrace:
    text: str
    n_tokens: int
    aborted: bool
    abort_token: Optional[int]
    p_correct_at_abort: float
    p_correct_trajectory: list[float]
    elapsed_s: float
    tps: float


def train_probe_from_captures(
    captures: list[dict],
    target_layers: list[int],
    feature_keys: list[str],
    C: float = 0.1,
) -> Pipeline:
    """Train StandardScaler + LogisticRegression on captured layer features.

    Args:
        captures: list of capture dicts (must have 'layer_features' and 'final_correct').
        target_layers: layer indices whose features to use (e.g. [6, 15, 41] for Gemma).
        feature_keys: feature names (without L{idx}_ prefix) per layer.
        C: LR regularization inverse strength.

    Returns:
        Fitted sklearn Pipeline.
    """
    feature_names = []
    for layer in target_layers:
        for fk in feature_keys:
            feature_names.append(f"L{layer}_{fk}")

    X = np.zeros((len(captures), len(feature_names)))
    y = np.array([1 if c.get("final_correct") else 0 for c in captures])

    for i, c in enumerate(captures):
        lf = c.get("layer_features", {})
        for j, fn in enumerate(feature_names):
            X[i, j] = lf.get(fn, 0.0)

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def get_probe_feature_names(target_layers: list[int], feature_keys: list[str]) -> list[str]:
    out = []
    for layer in target_layers:
        for fk in feature_keys:
            out.append(f"L{layer}_{fk}")
    return out


def compute_partial_probe_features(
    per_layer_norms_partial: dict[int, list[float]],
    target_layers: list[int],
    feature_keys: list[str],
) -> np.ndarray:
    """Compute probe input vector from current (partial) per-layer norms."""
    row = []
    for layer in target_layers:
        partial_norms = per_layer_norms_partial.get(layer, [])
        feats = partial_layer_features(partial_norms, layer)
        for fk in feature_keys:
            row.append(feats.get(f"L{layer}_{fk}", 0.0))
    return np.array(row, dtype=np.float32)


def generate_with_early_exit(
    model,
    tokenizer,
    prompt: str,
    probe: Pipeline,
    target_layers: list[int],
    feature_keys: list[str],
    max_tokens: int = 150,
    temperature: float = 0.0,
    cutoff: int = 5,
    threshold: float = 0.5,
    abort_consecutive: int = 2,
    keep_raw: bool = False,
    max_kv_size: Optional[int] = None,
    stop_sequences: Optional[list[str]] = None,
    probe_eval_every: int = 1,
) -> EarlyExitTrace:
    """Generate with mid-generation probe evaluation and early abort.

    After token T (T >= cutoff), the probe is evaluated every `probe_eval_every`
    tokens. If P(correct) < threshold for `abort_consecutive` consecutive
    evaluations, generation aborts.

    Args:
        probe: fitted sklearn Pipeline.
        target_layers: layers whose features the probe expects.
        feature_keys: per-layer feature names (without L{idx}_ prefix).
        cutoff: minimum tokens before first probe evaluation.
        threshold: P(correct) threshold below which we count an abort vote.
        abort_consecutive: how many consecutive low-confidence votes to abort on.
        probe_eval_every: probe eval cadence (1 = every token after cutoff).

    Returns:
        EarlyExitTrace.
    """
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    # Only wrap layers the probe needs (saves overhead).
    wrappers, originals, layers = install_layer_captures(
        model, layer_indices=target_layers, keep_raw=keep_raw
    )

    try:
        if max_kv_size is None:
            prompt_cache = mlx_cache.make_prompt_cache(model)
        else:
            prompt_cache = mlx_cache.make_prompt_cache(model, max_kv_size=max_kv_size)

        prompt_tokens = mx.array(tokenizer.encode(prompt))
        sampler = make_sampler(temp=temperature)

        gen = generate_step(
            prompt_tokens,
            model,
            prompt_cache=prompt_cache,
            sampler=sampler,
            max_tokens=max_tokens,
            max_kv_size=max_kv_size,
        )

        tokens = []
        p_trajectory: list[float] = []
        consecutive_low = 0
        abort_token = None
        p_at_abort = 0.0
        stopped_text: Optional[str] = None
        t0 = time.perf_counter()

        for i, (token, logprobs) in enumerate(gen):
            tokens.append(token)

            # Stop sequences
            if stop_sequences and ((i + 1) % 4 == 0 or i < 16):
                text_check = tokenizer.decode(tokens)
                for s in stop_sequences:
                    if s in text_check:
                        idx = text_check.find(s)
                        stopped_text = text_check[:idx]
                        break
                if stopped_text is not None:
                    break

            # Probe evaluation
            if (i + 1) >= cutoff and ((i + 1 - cutoff) % probe_eval_every == 0):
                # Snapshot per-layer norms seen so far (token i+1 generated)
                partial = {idx: list(w.norms) for idx, w in wrappers.items()}
                x = compute_partial_probe_features(partial, target_layers, feature_keys).reshape(1, -1)
                p_correct = float(probe.predict_proba(x)[0, 1])
                p_trajectory.append(p_correct)

                if p_correct < threshold:
                    consecutive_low += 1
                    if consecutive_low >= abort_consecutive:
                        abort_token = i + 1
                        p_at_abort = p_correct
                        break
                else:
                    consecutive_low = 0

        mx.synchronize()
        elapsed = time.perf_counter() - t0

        text = stopped_text if stopped_text is not None else tokenizer.decode(tokens)
        aborted = abort_token is not None
        n_gen = len(tokens)
        tps = n_gen / elapsed if elapsed > 0 else 0.0

        return EarlyExitTrace(
            text=text,
            n_tokens=n_gen,
            aborted=aborted,
            abort_token=abort_token,
            p_correct_at_abort=p_at_abort,
            p_correct_trajectory=p_trajectory,
            elapsed_s=elapsed,
            tps=tps,
        )

    finally:
        if originals:
            restore_layers(layers, originals)
