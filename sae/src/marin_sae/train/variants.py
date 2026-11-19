"""The SAEs trained side by side, and what makes each one a variant."""

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from marin_sae.config import RunConfig, SaeVariant
from marin_sae.dictionary.batchtopk import BatchTopKSae


class Variant(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: SaeVariant
    sae: BatchTopKSae
    optimiser: torch.optim.Optimizer
    edges: list[int]
    """Latent-range edges for decoding. Vanilla has one range, Matryoshka one per prefix."""


def build_variants(
    config: RunConfig, b_dec_init: Float[Tensor, "d_in"], device: str
) -> list[Variant]:
    variants: list[Variant] = []
    for name in config.variants:
        # The same seed for every variant: both start from identical weights.
        generator = torch.Generator().manual_seed(config.train.seed)
        sae = BatchTopKSae(
            config.sae.d_in,
            config.sae.n_latents,
            config.sae.k,
            config.sae.candidates_per_token,
            b_dec_init,
            generator,
        ).to(device)
        optimiser = torch.optim.Adam(
            sae.parameters(), lr=config.train.lr, betas=(0.9, 0.999)
        )
        edges = (
            [config.sae.n_latents]
            if name == "vanilla"
            else config.sae.matryoshka_prefixes
        )
        variants.append(Variant(name=name, sae=sae, optimiser=optimiser, edges=edges))
    return variants
