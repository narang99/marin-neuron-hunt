"""A BatchTopK sparse autoencoder with a sparse decoder.

How a forward pass goes:
1. `pre_activations`: ReLU of the encoder output, for every latent.
2. `select`: keep the `batch * k` largest activations across the whole batch.
   - Tokens may use more or fewer than `k` latents, which is what BatchTopK means.
   - Each token first keeps its `candidates` best latents, so the batch-wide cut runs on a small pool.
3. `decode`: sum the decoder rows of the kept latents, one latent range at a time.
   - Matryoshka prefixes are cumulative sums over those ranges, so nesting costs no extra gathers.
"""

import torch
from einops import einsum, repeat
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional as F

THRESHOLD_EMA = 0.99


class SparseCode(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    indices: Int[Tensor, "batch candidates"]
    values: Float[Tensor, "batch candidates"]
    """Activation per candidate. Candidates that lost the batch-wide cut are exactly zero."""


class BatchTopKSae(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_latents: int,
        k: int,
        candidates: int,
        b_dec_init: Float[Tensor, "d_in"],
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.k = k
        self.candidates = candidates
        decoder = torch.randn(n_latents, d_in, generator=generator)
        decoder = decoder / decoder.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(decoder)
        self.W_enc = nn.Parameter(decoder.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(b_dec_init.clone().float())
        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("last_fired", torch.zeros(n_latents, dtype=torch.long))

    @property
    def n_latents(self) -> int:
        return self.W_dec.shape[0]

    def pre_activations(
        self, x: Float[Tensor, "batch d_in"]
    ) -> Float[Tensor, "batch latent"]:
        centred = x - self.b_dec
        return F.relu(
            einsum(centred, self.W_enc, "batch d_in, d_in latent -> batch latent")
            + self.b_enc
        )

    def select(self, pre: Float[Tensor, "batch latent"]) -> SparseCode:
        values, indices = pre.topk(self.candidates, dim=-1)
        budget = pre.shape[0] * self.k
        cut = values.flatten().topk(budget).values[-1]
        if self.training:
            self.threshold.mul_(THRESHOLD_EMA).add_(
                (1 - THRESHOLD_EMA) * cut.detach().float()
            )
        return SparseCode(indices=indices, values=values * (values >= cut))

    def decode(
        self, code: SparseCode, edges: list[int]
    ) -> list[Float[Tensor, "batch d_in"]]:
        """Reconstructions that use only latents below each edge, from the smallest edge up.

        - `edges=[n_latents]` is the ordinary full reconstruction.
        - The decoder bias is added to every reconstruction.
        """
        batch = code.indices.shape[0]
        token = repeat(
            torch.arange(batch, device=code.indices.device),
            "b -> b c",
            c=code.indices.shape[1],
        )
        live = code.values > 0
        token, latent, value = token[live], code.indices[live], code.values[live]
        running = self.b_dec.expand(batch, -1)
        out: list[Float[Tensor, "batch d_in"]] = []
        lower = 0
        for upper in edges:
            in_range = (latent >= lower) & (latent < upper)
            running = running + self._gather(
                token[in_range], latent[in_range], value[in_range], batch
            )
            out.append(running)
            lower = upper
        return out

    def _gather(self, token, latent, value, batch: int) -> Float[Tensor, "batch d_in"]:
        counts = torch.bincount(token, minlength=batch)
        offsets = counts.cumsum(0) - counts
        return F.embedding_bag(
            latent,
            self.W_dec,
            offsets,
            mode="sum",
            per_sample_weights=value.to(self.W_dec.dtype),
        )

    def decode_extra(
        self, indices: Int[Tensor, "batch pool"], values: Float[Tensor, "batch pool"]
    ) -> Float[Tensor, "batch d_in"]:
        """Decoder sum for an arbitrary sparse code, without the bias. Used by the AuxK term."""
        return F.embedding_bag(
            indices,
            self.W_dec,
            mode="sum",
            per_sample_weights=values.to(self.W_dec.dtype),
        )

    @torch.no_grad()
    def mark_fired(self, code: SparseCode, rows_seen: int) -> None:
        self.last_fired[code.indices[code.values > 0]] = rows_seen

    @torch.no_grad()
    def reconstruct_at_threshold(
        self, x: Float[Tensor, "batch d_in"]
    ) -> tuple[Float[Tensor, "batch d_in"], Float[Tensor, ""]]:
        """Inference path: a fixed threshold in place of the batch-wide cut. Returns (x_hat, mean L0)."""
        pre = self.pre_activations(x)
        active = pre * (pre >= self.threshold)
        x_hat = (
            einsum(active, self.W_dec, "batch latent, latent d_in -> batch d_in")
            + self.b_dec
        )
        return x_hat, (active > 0).float().sum(dim=-1).mean()

    @torch.no_grad()
    def project_decoder_gradient(self) -> None:
        """Drop the gradient component along each decoder row, so a step only rotates it."""
        grad = self.W_dec.grad
        along = (grad * self.W_dec).sum(dim=-1, keepdim=True)
        grad.sub_(along * self.W_dec)

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True))
