"""layerIdentifier — model-agnostic entropy-layer identification.

Methodology ported (copied, not modified) from parallel_l4_tot's Qwen pipeline.
Applies the same capture primitives, feature formulas, and calibration pattern
to any MLX-loadable model so that Gemma (or any other model) can be compared
to Qwen on the same axis: which transformer layer carries the entropy signal?
"""
