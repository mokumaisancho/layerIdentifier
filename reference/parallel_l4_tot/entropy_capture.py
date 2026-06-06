"""Entropy capture and hidden-state layer capture for MLX model generation.

Provides LayerCaptureWrapper for intercepting hidden-state norms at specific
transformer layers, entropy-based spike detection, and generation with
per-token entropy capture.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import MAX_TOKENS

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_KV_SIZE = 256
ENTROPY_SPIKE_SIGMA = 1.5
HS_CAPTURE_LAYERS = [10, 12]


# ---------------------------------------------------------------------------
# Hidden State Layer Capture -- L10 + L12 norm interception
# ---------------------------------------------------------------------------
class LayerCaptureWrapper:
    """Captures L2 norm of last-token hidden state from a transformer layer.

    Memory: ~1.6KB per layer (list of floats). No raw tensor storage.
    """

    def __init__(self, layer):
        self.layer = layer
        self.norms: list[float] = []
        self._lock = __import__("threading").Lock()

    def __getattr__(self, name):
        if name in ("layer", "norms"):
            return super().__getattribute__(name)
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        with self._lock:
            arr = np.array(mx.array(out, dtype=mx.float32).tolist(), dtype=np.float32)
            if arr.ndim >= 3:
                arr = arr[0, -1]
            self.norms.append(float(np.linalg.norm(arr)))
            del arr
        return out


def install_layer_captures(model):
    """Install capture wrappers on target layers. Returns (wrappers, originals)."""
    layers_list = model.language_model.model.layers
    wrappers: dict[int, LayerCaptureWrapper] = {}
    originals: dict[int, object] = {}
    for idx in HS_CAPTURE_LAYERS:
        originals[idx] = layers_list[idx]
        wrappers[idx] = LayerCaptureWrapper(layers_list[idx])
        layers_list[idx] = wrappers[idx]
    return wrappers, originals


def restore_layers(model, originals):
    """Restore original layers after generation."""
    layers_list = model.language_model.model.layers
    for idx in originals:
        layers_list[idx] = originals[idx]
    mx.clear_cache()


def compute_spike_ratio(norms, zone="late"):
    """Compute spike ratio for a temporal zone from captured L2 norms.

    A spike = norm value exceeding mean + 1.5*std.
    Zones: early (<15%), mid (15-60%), late (>60%) of generation length.
    """
    vec = np.array(norms, dtype=np.float32)
    if len(vec) == 0:
        return 0.5
    mean_v, std_v = float(vec.mean()), float(vec.std())
    if std_v < 1e-10:
        return 0.5
    threshold = mean_v + ENTROPY_SPIKE_SIGMA * std_v
    n = len(vec)
    spike_pos = [i for i, v in enumerate(vec) if v > threshold]
    if not spike_pos:
        return 0.5
    count = 0
    for s in spike_pos:
        if zone == "early" and s < n * 0.15:
            count += 1
        elif zone == "mid" and n * 0.15 <= s < n * 0.60:
            count += 1
        elif zone == "late" and s >= n * 0.60:
            count += 1
    return count / len(spike_pos)


# ---------------------------------------------------------------------------
# MLX Generation with entropy capture
# ---------------------------------------------------------------------------
def generate_with_entropy(
    model,
    tokenizer,
    prompt: str,
    hidden_size: int,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    capture_layers: bool = True,
    repetition_penalty: float | None = None,
):
    """Generate text with per-token entropy capture. Returns (text, features, entropy_vec).

    Args:
        repetition_penalty: Multiplicative penalty for repeating tokens.
            Values > 1.0 discourage repetition (e.g., 1.1 to 1.3).
            See Holtzman et al. (2019) "The Curious Case of Neural Text Degeneration".
    """
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    # Install layer capture wrappers before generation
    wrappers, originals = {}, {}
    if capture_layers:
        wrappers, originals = install_layer_captures(model)

    try:
        return _generate_inner(
            model,
            tokenizer,
            prompt,
            hidden_size,
            max_tokens,
            temperature,
            wrappers,
            originals,
            mlx_cache,
            generate_step,
            make_sampler,
            make_logits_processors,
            repetition_penalty,
        )
    finally:
        if originals:
            restore_layers(model, originals)


def _generate_inner(
    model,
    tokenizer,
    prompt,
    hidden_size,
    max_tokens,
    temperature,
    wrappers,
    originals,
    mlx_cache,
    generate_step,
    make_sampler,
    make_logits_processors,
    repetition_penalty,
):
    prompt_cache = mlx_cache.make_prompt_cache(model, max_kv_size=MAX_KV_SIZE)
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    sampler = make_sampler(temp=temperature)

    # Build logits processors (repetition penalty etc.)
    logits_processors = None
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=20,
        )

    gen = generate_step(
        prompt_tokens,
        model,
        prompt_cache=prompt_cache,
        sampler=sampler,
        max_tokens=max_tokens,
        max_kv_size=MAX_KV_SIZE,
        logits_processors=logits_processors,
    )

    tokens = []
    entropies = []
    for token, logprobs in gen:
        tokens.append(token)
        # Incremental entropy: compute per-token, avoid storing full logprob arrays
        probs = mx.softmax(logprobs.astype(mx.float32))
        ent = -mx.sum(probs * mx.log(probs + 1e-10))
        entropies.append(float(ent))
    mx.synchronize()

    text = tokenizer.decode(tokens)

    ent_arr = np.array(entropies, dtype=np.float32) if entropies else np.empty(0, dtype=np.float32)

    if len(ent_arr) > 0:
        mean_ent = float(ent_arr.mean())
        std_ent = float(ent_arr.std())
        spike_threshold = mean_ent + ENTROPY_SPIKE_SIGMA * std_ent
        spike_positions = [int(i) for i, e in enumerate(ent_arr) if e > spike_threshold]

        # -- Entropy delta features (zero-cost, from existing ent_arr) --
        # Compute token-to-token entropy deltas
        if len(ent_arr) >= 2:
            deltas = np.diff(ent_arr)
            max_pos_delta = float(np.max(deltas))
            max_neg_delta = float(np.min(deltas))
            delta_var = float(np.var(deltas))
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas))
        else:
            max_pos_delta = max_neg_delta = delta_var = delta_mean = delta_std = 0.0

        features = {
            "mean_entropy": mean_ent,
            "max_entropy": float(ent_arr.max()),
            "first_token_entropy": float(ent_arr[0]),
            "entropy_std": std_ent,
            "n_spike_tokens": len(spike_positions),
            "spike_positions": spike_positions[:10],
            "n_tokens": len(ent_arr),
            # Delta features -- computable from existing entropy_vec
            "max_positive_delta": max_pos_delta,
            "max_negative_delta": max_neg_delta,
            "delta_variance": delta_var,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
    else:
        features = {
            "mean_entropy": 5.0,
            "max_entropy": 5.0,
            "first_token_entropy": 5.0,
            "entropy_std": 0.0,
            "n_spike_tokens": 0,
            "spike_positions": [],
            "n_tokens": 0,
            "max_positive_delta": 0.0,
            "max_negative_delta": 0.0,
            "delta_variance": 0.0,
            "delta_mean": 0.0,
            "delta_std": 0.0,
        }

    # Extract hidden state spike features from captured layer norms
    if wrappers:
        if 12 in wrappers:
            features["L12_late_spike_ratio"] = compute_spike_ratio(
                wrappers[12].norms, zone="late"
            )
        if 10 in wrappers:
            features["L10_mid_spike_ratio"] = compute_spike_ratio(
                wrappers[10].norms, zone="mid"
            )
        for w in wrappers.values():
            w.norms.clear()

    return text, features, ent_arr
