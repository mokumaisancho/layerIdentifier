"""Model loader — wraps mlx_lm.load with sensible defaults for layerIdentifier.

For multimodal Gemma-4 (text+vision+audio), we want the text language_model.
mlx_lm.load should give us that, but we verify and warn if the loaded object
has a language_model submodule we should be using directly.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _patch_tokenizer_config(model_path: Path) -> None:
    """Convert extra_special_tokens from list to dict if needed (transformers 4.x compat).

    Idempotent: backs up the original once, no-ops if already dict.
    Only touches the model_path we own — never touches other repos.
    """
    cfg_path = model_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    est = cfg.get("extra_special_tokens")
    if isinstance(est, list):
        if not (model_path / "tokenizer_config.json.bak").exists():
            shutil.copy2(cfg_path, model_path / "tokenizer_config.json.bak")
        cfg["extra_special_tokens"] = {t: t for t in est}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    path: str
    n_layers: int
    arch: str


def load_model(model_path: str, dtype: Optional[str] = None) -> LoadedModel:
    """Load an MLX-format model from a local path.

    Args:
        model_path: directory containing config.json + model.safetensors.
        dtype: optional override ('float16', 'bfloat16', 'float32').

    Returns:
        LoadedModel dataclass.
    """
    # Use lower-level API so we can pass strict=False (needed for Gemma-4 KV-shared layers,
    # whose weights file contains per-layer copies of the shared KV that the model dedups).
    from mlx_lm.utils import load_model, load_tokenizer
    from pathlib import Path

    # Workaround: transformers 4.x expects extra_special_tokens as dict; Gemma-4 ships list.
    # Patch our local copy in-place (does NOT touch any other repo).
    _patch_tokenizer_config(Path(model_path))

    config_overrides = {}
    if dtype is not None:
        config_overrides["dtype"] = dtype
    model, config = load_model(
        Path(model_path), lazy=False, strict=False,
        model_config=config_overrides if config_overrides else None,
    )
    tokenizer = load_tokenizer(
        Path(model_path),
        eos_token_ids=config.get("eos_token_id", None),
    )

    # Resolve architecture
    arch = type(model).__name__

    # Count layers via the same resolver used at capture time
    from .layer_capture import resolve_layers
    layers = resolve_layers(model)
    n = len(layers)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        path=model_path,
        n_layers=n,
        arch=arch,
    )


def quick_summary(lm: LoadedModel) -> str:
    return (
        f"model: {lm.arch}  layers: {lm.n_layers}  path: {os.path.basename(lm.path)}"
    )
