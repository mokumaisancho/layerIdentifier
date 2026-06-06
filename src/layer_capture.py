"""Layer capture primitives — model-agnostic.

Adapted from reference/parallel_l4_tot/entropy_capture.py (LayerCaptureWrapper).
Difference from Qwen reference:
  * Captures EVERY layer (not just L10/L12), so ranking can be computed.
  * model.layers resolved generically (no hardcoded .language_model.model.layers path).
  * Stores L2 norm + (optional) raw last-token hidden state for downstream probing.

Same L2-norm-of-last-token-hidden-state convention as Qwen, so features are
directly comparable.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("layerIdentifier requires MLX. Install with: pip install mlx_lm") from e


ENTROPY_SPIKE_SIGMA = 1.5  # same as Qwen reference


class LayerCaptureWrapper:
    """Wraps a transformer layer module; captures L2 norm of last-token hidden state per call.

    Identical convention to Qwen's LayerCaptureWrapper:
      - Calls the underlying layer normally (no behavior change).
      - Reads the layer's output tensor, takes the last-token of batch 0.
      - Stores L2 norm (float). Optionally stores the raw vector for probing.
      - Thread-safe via a lock.

    Memory: ~8 bytes/layer/token (float64 norm). For 42 layers × 1000 tokens ≈ 0.3 MB.
    With raw-vector capture: hidden_size × 4 bytes × tokens × layers — can be large.
    """

    def __init__(self, layer, layer_idx: int, keep_raw: bool = False):
        self.layer = layer
        self.layer_idx = layer_idx
        self.norms: list[float] = []
        self.raw_last: list[np.ndarray] = []  # only populated if keep_raw=True
        self.keep_raw = keep_raw
        self._lock = threading.Lock()

    # Forward all attribute reads to the wrapped layer (so existing MLX graph works)
    def __getattr__(self, name):
        if name in ("layer", "layer_idx", "norms", "raw_last", "keep_raw", "_lock"):
            return super().__getattribute__(name)
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        with self._lock:
            # MLX returns either an array or a list/tuple of arrays.
            # We want the hidden-state output (first element if list).
            arr = out[0] if isinstance(out, (list, tuple)) else out
            arr_mx = mx.array(arr, dtype=mx.float32)
            # Last token of batch 0
            if arr_mx.ndim >= 3:
                last = arr_mx[0, -1]
            elif arr_mx.ndim == 2:
                last = arr_mx[-1]
            else:
                last = arr_mx
            # MLX-side norm (skip numpy roundtrip; ~10-20% faster per token)
            norm_val = float(mx.linalg.norm(last))
            self.norms.append(norm_val)
            if self.keep_raw:
                np_last = np.array(last.tolist(), dtype=np.float32)
                self.raw_last.append(np_last.copy())
        return out

    def clear(self):
        with self._lock:
            self.norms.clear()
            self.raw_last.clear()


def resolve_layers(model) -> list:
    """Resolve the transformer layer list for any MLX-loaded model.

    Tries (in order):
      1. model.language_model.model.layers        (Qwen3 HF layout, also Gemma3)
      2. model.model.layers                       (common HF causal-LM layout)
      3. model.layers                             (bare transformer)
      4. model.language_model.layers              (other multimodal layouts)
    """
    paths = [
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
        ("language_model", "layers"),
    ]
    for path in paths:
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, list) and len(obj) > 0:
            return obj
    raise AttributeError(
        f"Could not find transformer layer list on model. "
        f"Available top-level attrs: {dir(model)[:30]}"
    )


def install_layer_captures(model, layer_indices: Optional[list[int]] = None,
                           keep_raw: bool = False):
    """Install LayerCaptureWrapper on requested layers.

    Args:
        model: MLX-loaded model.
        layer_indices: Indices to wrap. None = wrap ALL layers.
        keep_raw: If True, also store raw last-token hidden state (memory-heavy).

    Returns:
        wrappers: dict[int, LayerCaptureWrapper]
        originals: dict[int, original layer object]
        layers: the resolved layer list (so caller can iterate)
    """
    layers = resolve_layers(model)
    n = len(layers)
    if layer_indices is None:
        target_idxs = list(range(n))
    else:
        target_idxs = [i for i in layer_indices if 0 <= i < n]

    wrappers: dict[int, LayerCaptureWrapper] = {}
    originals: dict[int, object] = {}
    for idx in target_idxs:
        originals[idx] = layers[idx]
        wrappers[idx] = LayerCaptureWrapper(layers[idx], idx, keep_raw=keep_raw)
        layers[idx] = wrappers[idx]
    return wrappers, originals, layers


def restore_layers(layers, originals):
    """Undo installation. Pass the `layers` list returned by install_layer_captures."""
    for idx, original in originals.items():
        layers[idx] = original
    mx.clear_cache()
