"""Losses for the two SAE variants.

- Vanilla trains the full dictionary only.
- Matryoshka also asks each nested prefix of latents to reconstruct on its own.
  - The first latents are pushed toward broad, frequent features, and later ones toward specific ones.
- Both add the AuxK term, which gives dead latents a gradient so they can come back.
"""

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from marin_sae.dictionary.batchtopk import BatchTopKSae

EPS = 1e-8


def fvu(
    x: Float[Tensor, "batch d_in"], x_hat: Float[Tensor, "batch d_in"]
) -> Float[Tensor, ""]:
    """Fraction of variance unexplained, summed over the batch."""
    centred = x - reduce(x, "batch d_in -> d_in", "mean")
    return (x - x_hat).float().pow(2).sum() / (centred.float().pow(2).sum() + EPS)


def reconstruction_loss(
    x: Float[Tensor, "batch d_in"], reconstructions: list[Float[Tensor, "batch d_in"]]
) -> Float[Tensor, ""]:
    """Mean FVU over the nested reconstructions. One entry means a plain full-dictionary loss."""
    return torch.stack([fvu(x, x_hat) for x_hat in reconstructions]).mean()


def auxk_loss(
    sae: BatchTopKSae,
    x: Float[Tensor, "batch d_in"],
    x_hat: Float[Tensor, "batch d_in"],
    pre: Float[Tensor, "batch latent"],
    dead: Tensor,
    pool: int,
) -> Float[Tensor, ""]:
    """Dead latents reconstruct the residual the live ones missed.

    - Without it a TopK latent that stops firing gets zero gradient and never recovers.
    - The residual is detached, so this term only trains the dead latents and the encoder rows.
    """
    n_dead = int(dead.sum())
    if n_dead == 0:
        return x.new_zeros(())
    residual = (x - x_hat).detach().float()
    values, indices = (pre * dead).topk(min(pool, n_dead), dim=-1)
    guess = sae.decode_extra(indices, values)
    centred = residual - reduce(residual, "batch d_in -> d_in", "mean")
    return (residual - guess).pow(2).sum() / (centred.pow(2).sum() + EPS)
