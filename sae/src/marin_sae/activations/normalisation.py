"""One scalar scale for the whole activation vector, as in Gemma Scope.

- Dividing each neuron by its own scale would distort the geometry the SAE has to learn.
- A single constant keeps every direction's relative size and only fixes the overall magnitude.
"""

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from marin_sae.activations.rows import residual_rows
from marin_sae.model.prefix import ResidualFn


class Normalisation(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    scale: float
    """Multiply raw activations by this so the mean norm is sqrt(d_model)."""
    mean: Float[Tensor, "d_model"]
    """Mean of the scaled activations, used to initialise the decoder bias."""


def fit_normalisation(
    residual_fn: ResidualFn, token_batches, bos_id: int, n_rows: int, d_model: int
) -> Normalisation:
    rows: list[Float[Tensor, "row d_model"]] = []
    collected = 0
    for tokens in token_batches:
        batch = residual_rows(residual_fn, tokens, bos_id).float()
        rows.append(batch)
        collected += len(batch)
        if collected >= n_rows:
            break
    data = torch.cat(rows)[:n_rows]
    scale = d_model**0.5 / data.norm(dim=-1).mean().item()
    return Normalisation(scale=scale, mean=(data * scale).mean(dim=0))
