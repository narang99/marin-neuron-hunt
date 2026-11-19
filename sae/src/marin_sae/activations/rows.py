"""Turns a batch of token ids into flat activation rows, one per non-BOS token."""

from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from marin_sae.model.prefix import ResidualFn


def residual_rows(
    residual_fn: ResidualFn, tokens: Int[Tensor, "batch seq"], bos_id: int
) -> Float[Tensor, "row d_model"]:
    """Residual rows for every token except BOS.

    - BOS is an attention sink with a huge norm, so keeping it would distort the scale and the SAE.
    """
    residuals = residual_fn(tokens)
    flat = rearrange(residuals, "batch seq d_model -> (batch seq) d_model")
    keep = rearrange(tokens.to(flat.device) != bos_id, "batch seq -> (batch seq)")
    return flat[keep]
