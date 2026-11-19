"""Spliced-in loss: put the SAE's reconstruction back into the model and see how much next-token loss it costs.

- FVU only says the reconstruction is close in the residual stream. Spliced-in loss says whether
  the model still works, which is the number that matters for using the SAE to explain the model.
- Three runs per batch share the same tokens: clean, mean-ablated (the floor) and spliced.
- `loss recovered = (ablated - spliced) / (ablated - clean)`, so 1 is a perfect splice and 0 is no better than the mean.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

Replacement = Callable[
    [Float[Tensor, "batch seq d_model"]], Float[Tensor, "batch seq d_model"]
]


def next_token_loss(
    logits: Float[Tensor, "batch seq vocab"],
    tokens: Int[Tensor, "batch seq"],
    bos_id: int,
) -> Float[Tensor, ""]:
    """Mean cross-entropy over targets that are not BOS (the first token of a packed document)."""
    predictions = rearrange(
        logits[:, :-1], "batch seq vocab -> (batch seq) vocab"
    ).float()
    targets = rearrange(tokens[:, 1:], "batch seq -> (batch seq)")
    keep = targets != bos_id
    return F.cross_entropy(predictions[keep], targets[keep])


@torch.no_grad()
def loss_with_replacement(
    model: nn.Module,
    layer_module: nn.Module,
    tokens: Int[Tensor, "batch seq"],
    bos_id: int,
    replace: Replacement | None,
) -> float:
    """Next-token loss, with the layer output swapped through `replace` at every non-BOS position."""
    handle = None
    if replace is not None:
        is_bos = (tokens == bos_id).unsqueeze(-1)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            swapped = torch.where(is_bos, hidden, replace(hidden).to(hidden.dtype))
            return (swapped, *output[1:]) if isinstance(output, tuple) else swapped

        handle = layer_module.register_forward_hook(hook)
    try:
        logits = model(input_ids=tokens, use_cache=False).logits
    finally:
        if handle:
            handle.remove()
    return next_token_loss(logits, tokens, bos_id).item()


def loss_recovered(clean: float, ablated: float, spliced: float) -> float:
    gap = ablated - clean
    return (ablated - spliced) / gap if abs(gap) > 1e-9 else float("nan")
