"""Generation pipeline with per-token entropy + per-layer hidden-state capture.

Adapted from reference/parallel_l4_tot/entropy_capture.py:generate_with_entropy.
Differences from Qwen reference:
  * Captures ALL layers by default (configurable).
  * Returns full per-layer norm dict, not just L10/L12 features.
  * Same per-token entropy vector convention.
  * Same entropy spike threshold (mean + 1.5σ).
  * MAX_KV_SIZE lifted to None (unbounded) when possible — Qwen reference noted this
    matters for attention analysis (RotatingKVCache evicts old entries).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("MLX required") from e

from .layer_capture import install_layer_captures, restore_layers, resolve_layers
from .features import entropy_features, all_layers_features


@dataclass
class GenTrace:
    """Single generation trace.

    `per_layer_norms[idx]` is a list of L2 norms (one per generated token).
    """
    text: str
    n_tokens: int
    entropy_vec: np.ndarray
    per_layer_norms: dict[int, list[float]]
    entropy_features: dict[str, float]
    layer_features: dict[str, float]
    elapsed_s: float
    tps: float


def generate_with_full_capture(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    layer_indices: Optional[list[int]] = None,
    keep_raw: bool = False,
    max_kv_size: Optional[int] = None,
    stop_sequences: Optional[list[str]] = None,
):
    """Generate one continuation, capturing per-token entropy + per-layer L2 norms.

    Args:
        model: MLX-loaded model.
        tokenizer: MLX tokenizer.
        prompt: prompt text.
        max_tokens: cap on generated tokens.
        temperature: 0.0 = greedy; >0 = sampled.
        layer_indices: which layers to wrap (None = all).
        keep_raw: also keep raw last-token hidden state (memory-heavy).
        max_kv_size: None = unbounded KV cache (recommended for layer analysis).
        stop_sequences: optional list of decode-string stops (e.g. ["<turn|>", "<|end|>"]).
            Checked every 4 generated tokens; truncates text at first match.

    Returns:
        GenTrace dataclass.
    """
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    wrappers, originals, layers = install_layer_captures(
        model, layer_indices=layer_indices, keep_raw=keep_raw
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
        entropies = []
        stopped_text: Optional[str] = None
        t0 = time.perf_counter()
        for i, (token, logprobs) in enumerate(gen):
            tokens.append(token)
            probs = mx.softmax(logprobs.astype(mx.float32))
            ent = -mx.sum(probs * mx.log(probs + 1e-10))
            entropies.append(float(ent))

            # Stop sequences: check every 4 tokens (cheap), and every token early-on.
            # Decode overhead is ~0.3ms per call; total ~5% of one forward pass.
            if stop_sequences and ((i + 1) % 4 == 0 or i < 16):
                text_check = tokenizer.decode(tokens)
                for s in stop_sequences:
                    if s in text_check:
                        idx = text_check.find(s)
                        stopped_text = text_check[:idx]
                        break
                if stopped_text is not None:
                    break

        mx.synchronize()
        elapsed = time.perf_counter() - t0

        text = stopped_text if stopped_text is not None else tokenizer.decode(tokens)
        ent_arr = np.array(entropies, dtype=np.float32)

        per_layer_norms = {idx: list(w.norms) for idx, w in wrappers.items()}

        ent_feats = entropy_features(ent_arr)
        layer_feats = all_layers_features(per_layer_norms)

        n_gen = len(tokens)
        tps = n_gen / elapsed if elapsed > 0 else 0.0

        return GenTrace(
            text=text,
            n_tokens=n_gen,
            entropy_vec=ent_arr,
            per_layer_norms=per_layer_norms,
            entropy_features=ent_feats,
            layer_features=layer_feats,
            elapsed_s=elapsed,
            tps=tps,
        )

    finally:
        if originals:
            restore_layers(layers, originals)


def count_layers(model) -> int:
    """Return the transformer layer count for any MLX model."""
    return len(resolve_layers(model))
