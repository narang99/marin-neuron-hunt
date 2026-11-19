"""The whole model, for the spliced-in loss. It needs about 65 GB in bf16, so free the prefix and training first."""

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, Qwen3ForCausalLM

from marin_sae.config import ModelConfig
from marin_sae.model.prefix import DTYPES, default_dtype, weight_map


def load_full_model(model: ModelConfig, device: str) -> Qwen3ForCausalLM:
    config = AutoConfig.from_pretrained(model.shard_dir)
    config._attn_implementation = "sdpa"
    with torch.device(device), default_dtype(DTYPES[model.dtype]):
        full = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    keys = weight_map(model.shard_dir)
    loaded: set[str] = set()
    for shard in sorted(set(keys.values())):
        path: Path = model.shard_dir / shard
        with safe_open(path, framework="pt", device=device) as handle:
            part = {
                key: handle.get_tensor(key)
                for key, file in keys.items()
                if file == shard
            }
        full.load_state_dict(part, strict=False)
        loaded |= set(part)
        del part
    missing = set(full.state_dict()) - loaded
    if missing:
        raise RuntimeError(f"full model load is missing {sorted(missing)[:5]}")
    return full
