"""Top-activating contexts for a few latents, to read by eye.

- A number like FVU cannot tell you whether a latent means anything. Reading its best examples can.
- Latents are drawn at random from those that fired, plus some from the first Matryoshka prefix,
  which should hold the broadest features.
"""

import heapq

import torch
from jaxtyping import Int
from pydantic import BaseModel
from torch import Tensor


class Example(BaseModel):
    activation: float
    text: str


class TopExamples:
    """Keeps the best `per_latent` examples for each watched latent."""

    def __init__(self, latents: list[int], per_latent: int) -> None:
        self.per_latent = per_latent
        self.heaps: dict[int, list[tuple[float, int, str]]] = {
            latent: [] for latent in latents
        }
        self._count = 0

    def offer(self, latent: int, activation: float, text: str) -> None:
        heap = self.heaps[latent]
        self._count += 1
        entry = (activation, self._count, text)
        if len(heap) < self.per_latent:
            heapq.heappush(heap, entry)
        elif activation > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def best(self, latent: int) -> list[Example]:
        return [
            Example(activation=a, text=t)
            for a, _, t in sorted(self.heaps[latent], reverse=True)
        ]


def choose_latents(
    fired: Int[Tensor, "latent"],
    n_total: int,
    first_prefix: int,
    generator: torch.Generator,
) -> list[int]:
    """Half from the first prefix, half from the rest, only among latents that fired."""
    alive = fired.nonzero().squeeze(-1)
    early = alive[alive < first_prefix]
    late = alive[alive >= first_prefix]
    half = n_total // 2
    pick = lambda pool, n: pool[
        torch.randperm(len(pool), generator=generator)[:n]
    ].tolist()
    return pick(early, half) + pick(late, n_total - half)


def window(tokens: Int[Tensor, "seq"], position: int, context: int, decode) -> str:
    start = max(0, position - context)
    before = decode(tokens[start:position].tolist())
    return f"{before}[[{decode([int(tokens[position])])}]]{decode(tokens[position + 1 : position + 6].tolist())}"
