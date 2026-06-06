# Plan: Attention Pattern Capture for Fusion Error Detection

## Context

The selector picks wrong paths on 3/50 problems (Q110, Q112, Q136) because the model makes **fusion errors** — it incorrectly combines information from prior reasoning steps. These errors happen inside the model's attention computation and are invisible in output features (entropy, delta_std, composite scores). We confirmed:

- Wrong paths have HIGHER delta_std (0.860 vs 0.615) — entropy oscillates AFTER the error
- The fusion error itself happens confidently (low entropy) — it's an addressing error, not a struggle
- Tang & Xie 2026 (arXiv:2603.19272): Transformers are stateless DNCs — attention IS content-based addressing, fusion errors ARE addressing errors

**Goal**: Capture per-token attention patterns during generation to detect where the model reads wrong "memory cells" (addresses incorrect prior tokens) at fusion points.

---

## Architecture: Qwen3.5-4B-MLX

- **32 decoder layers**, hybrid attention:
  - 8 full-attention (softmax) layers at indices **3, 7, 11, 15, 19, 23, 27, 31**
  - 24 GatedDeltaNet (linear attention) layers — no attention weight matrix
- **GQA**: 16 Q heads, 4 KV heads, head_dim=256
- **Critical constraint**: `mx.fast.scaled_dot_product_attention` is a fused Metal kernel — returns only output, NOT attention weights. Must manually re-compute Q@K^T+softmax.
- **KV cache**: Existing code uses `RotatingKVCache` with `max_kv_size=256` — evicts old entries. Need unbounded `KVCache` for attention analysis.

---

## Step 1: Create `_attention_fusion_probe.py` (single prototype file)

### 1a. AttentionProbeWrapper class (~80 lines)

Wraps `Qwen3NextAttention` module on the 8 full-attention layers. Pattern follows existing `LayerCaptureWrapper` in `entropy_capture.py` (lines 34-59):

```python
class AttentionProbeWrapper:
    def __init__(self, attn_module, layer_idx, top_k=10):
        self.attn = attn_module
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.snapshots = []  # per-token attention data

    def __call__(self, x, mask=None, cache=None):
        output = self.attn(x, mask=mask, cache=cache)  # original computation
        self._capture_attention(x, cache)               # probe
        return output
```

**`_capture_attention` logic**:
1. Re-project Q from input `x` (last token only): `q_proj → split → q_norm → RoPE`
2. Extract all cached K from `cache.state` (already updated by original call)
3. Compute `scores = (Q_last @ K_all^T) * scale` — single dot product per KV head
4. Apply softmax → get attention weight vector `(1, n_kv_heads, seq_len)`
5. Compress: store top-K positions, weights, entropy, argmax per head
6. Free MLX intermediates immediately

**Memory**: 768 tokens × 8 layers × 4 KV heads × 10 top-K × 2 (pos+weight) = ~2.3MB per path. Three paths = ~7MB.

**Overhead**: One extra Q projection + one dot product per token per layer. ~5-10% generation slowdown.

### 1b. install_attention_probes() / restore_probes()

Same monkey-patching pattern as `install_layer_captures` in `entropy_capture.py`:
- Target `model.language_model.model.layers[i].self_attn` for i in [3,7,11,15,19,23,27,31]
- Only wrap if `not layer.is_linear` (defensive check)

### 1c. generate_with_attention_capture() (~60 lines)

Adapted from `entropy_capture.generate_with_entropy`:
- Use `model.make_cache()` (no max_kv_size) for unbounded KVCache
- Install `AttentionProbeWrapper` on 8 attention layers
- Install `LayerCaptureWrapper` on layers 10, 12 (existing, for hidden state norms)
- Run `generate_step`, collect tokens + logprobs + attention snapshots
- Returns `(text, features, entropy_array, attention_data)`

### 1d. Reasoning step segmentation (~30 lines)

Simple heuristic to map token positions to reasoning steps:
- Split on step markers: "Step N:", numbered lines, `$$` LaTeX blocks, `=`
- Use tokenizer to map character offsets → token positions
- Returns `[(step_idx, token_start, token_end), ...]`

### 1e. Fusion point analysis (~60 lines)

For each entropy spike position (entropy > mean + 1.5*std):
- Collect attention snapshots across all 8 layers
- Compute **cross-step attention fraction**: what fraction of attention weight goes to positions outside the current reasoning step
- Compute **step-of-argmax**: which reasoning step receives the most attention
- Compute **attention dispersion**: entropy of the attention distribution

### 1f. compare_correct_vs_wrong() (~50 lines)

For the same problem, compare correct path vs wrong path:
- At matching spike positions, compare cross-step attention fractions
- Check if wrong path's argmax attended position comes from a different step
- Compare attention entropy at fusion points

### 1g. main() (~60 lines)

- Load model
- For Q112, Q110, Q136 (the 3 remaining hard problems):
  - Generate 3 paths (temps 0.1, 0.7, 1.2)
  - Capture attention on each
  - Analyze fusion points
  - Print comparison: correct vs wrong path attention traces

---

## Step 2: Validate Attention Capture

Run the prototype on a simple problem first:
1. Verify attention weights sum to 1.0 per head
2. Verify output text is identical with and without probe wrapper
3. Verify argmax attention position is plausible (recent tokens for last layer)
4. Verify memory usage stays under budget (~7MB per path)

---

## Step 3: Fusion Error Analysis on Hard Problems

Run on Q112, Q110, Q136:
- For each problem, compare attention patterns of correct path vs wrong path
- Expected finding: at fusion points, wrong path attends to positions from an incorrect prior step (wrong "memory cell" in DNC terms)
- Report: per-spike comparison showing attended step vs expected step

---

## Step 4: Statistical Validation (if Step 3 shows signal)

Run on all 50 problems (150 paths):
- Compute AUROC of "cross-step attention fraction at fusion points" as predictor of path correctness
- Compare against existing features (entropy, delta_std)
- If AUROC > existing features → attention-based selector feature is viable

---

## Critical Files

| File | Role |
|------|------|
| `/Users/apple/Downloads/Py/parallel_l4_tot/entropy_capture.py` | Existing LayerCaptureWrapper pattern to follow |
| `/Users/apple/Downloads/Py/parallel_l4_tot/_attention_fusion_probe.py` | **NEW**: prototype script |
| `/opt/homebrew/lib/python3.14/site-packages/mlx_lm/models/qwen3_next.py` | Qwen3NextAttention source (lines 81-158) — the exact module being wrapped |
| `/opt/homebrew/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py` | Model definition, layer indexing, make_cache |
| `/Users/apple/Downloads/Py/parallel_l4_tot/run_benchmark.py` | Model loading pattern (line 435) |

## Key Technical Details

- RoPE offset during generation: `cache.offset - 1` (cache already updated by original call)
- Skip attention capture during prompt prefill (L > 1), only capture during generation (L = 1)
- GQA: compute attention per KV head (4 heads), not per Q head (16) — sufficient for fusion detection
- Gate signal (`mx.sigmoid(gate)`) in Qwen3NextAttention does not affect attention weights, only output scaling

## ETA

- AttentionProbeWrapper + generation: ~30 min
- Fusion analysis: ~20 min
- Validation on Q112: ~10 min
- **Total: ~60 min** (includes model loading time)
