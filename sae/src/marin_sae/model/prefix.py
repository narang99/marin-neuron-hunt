"""The first layers of Marin, cut off at the layer whose residual stream the SAE reads.

- Only layers `0..layer` and the embedding are built, so layer 8 of a 32B model costs about 10 GB.
- Weights are read tensor by tensor straight from the safetensors shards.
"""

import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from jaxtyping import Float, Int
from safetensors import safe_open
from torch import Tensor
from transformers import AutoConfig, Qwen3Model

from marin_sae.config import ModelConfig

ResidualFn = Callable[[Int[Tensor, "batch seq"]], Float[Tensor, "batch seq d_model"]]
"""Maps token ids to the residual stream after the hooked layer."""

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
MODEL_PREFIX = "model."
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Build the model directly in `dtype`, without a float32 copy first."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def weight_map(shard_dir: Path) -> dict[str, str]:
    """Checkpoint key -> shard file name, for sharded and single-file checkpoints."""
    index = shard_dir / INDEX_FILE
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(shard_dir / SINGLE_FILE, framework="pt") as single:
        return {key: SINGLE_FILE for key in single.keys()}


def is_in_prefix(key: str, layer: int) -> bool:
    if key.startswith(f"{MODEL_PREFIX}embed_tokens."):
        return True
    return any(key.startswith(f"{MODEL_PREFIX}layers.{i}.") for i in range(layer + 1))


def build_prefix(model: ModelConfig, device: str) -> Qwen3Model:
    config = AutoConfig.from_pretrained(model.shard_dir)
    config.num_hidden_layers = model.layer + 1
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[: model.layer + 1]
    config._attn_implementation = "sdpa"
    with torch.device(device), default_dtype(DTYPES[model.dtype]):
        return Qwen3Model(config).eval().requires_grad_(False)


def load_prefix(model: ModelConfig, device: str) -> Qwen3Model:
    prefix = build_prefix(model, device)
    wanted = {
        k: f
        for k, f in weight_map(model.shard_dir).items()
        if is_in_prefix(k, model.layer)
    }
    tensors: dict[str, Tensor] = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(
            model.shard_dir / shard, framework="pt", device=device
        ) as handle:
            for key in (k for k, f in wanted.items() if f == shard):
                tensors[key.removeprefix(MODEL_PREFIX)] = handle.get_tensor(key)
    missing, unexpected = prefix.load_state_dict(tensors, strict=False)
    # The final norm is not in the prefix on purpose: the hook reads the layer output before it.
    missing = [key for key in missing if not key.startswith("norm.")]
    if missing or unexpected:
        raise RuntimeError(
            f"prefix load mismatch, missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    return prefix


class ResidualTap:
    """Callable that returns the residual stream after decoder layer `layer`.

    - A forward hook captures the layer output, so the model's own final norm is ignored.
    """

    def __init__(self, prefix: Qwen3Model, layer: int) -> None:
        self.prefix = prefix
        self.d_model: int = prefix.config.hidden_size
        self._captured: Tensor | None = None
        prefix.layers[layer].register_forward_hook(self._store)

    def _store(self, _module, _inputs, output) -> None:
        self._captured = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def __call__(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq d_model"]:
        self._captured = None
        self.prefix(input_ids=tokens.to(self.prefix.device), use_cache=False)
        if self._captured is None:
            raise RuntimeError("the layer hook did not fire")
        return self._captured
