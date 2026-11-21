"""A shuffled pool of activation rows, so each SAE batch mixes tokens from many documents.

How it works:
- Fill the pool to capacity from the LLM, then shuffle every row.
- Batches are taken off the end of the pool until it is half empty.
- The freed half is refilled with fresh rows and the whole pool is shuffled again.
"""

from collections.abc import Iterator

import torch
from jaxtyping import Float, Int
from torch import Tensor

from marin_sae.activations.normalisation import Normalisation
from marin_sae.activations.rows import residual_rows
from marin_sae.model.prefix import ResidualFn


class ActivationBuffer:
    def __init__(
        self,
        residual_fn: ResidualFn,
        token_batches: Iterator[Int[Tensor, "batch seq"]],
        bos_id: int,
        normalisation: Normalisation,
        capacity: int,
        batch_rows: int,
        d_model: int,
        device: str,
        generator: torch.Generator,
    ) -> None:
        if capacity < 2 * batch_rows:
            raise ValueError(
                f"capacity {capacity} must hold at least two batches of {batch_rows}"
            )
        self.residual_fn = residual_fn
        self.token_batches = token_batches
        self.bos_id = bos_id
        self.scale = normalisation.scale
        self.capacity = capacity
        self.batch_rows = batch_rows
        self.generator = generator
        self.rows_loaded = 0
        self._pool = torch.empty(capacity, d_model, dtype=torch.bfloat16, device=device)
        self._count = 0
        self._refill()

    def _refill(self) -> None:
        while self._count < self.capacity:
            tokens = next(self.token_batches).to(self._pool.device)
            rows = residual_rows(self.residual_fn, tokens, self.bos_id) * self.scale
            take = min(len(rows), self.capacity - self._count)
            self._pool[self._count : self._count + take] = rows[:take].to(
                self._pool.dtype
            )
            self._count += take
            self.rows_loaded += take
        order = torch.randperm(self.capacity, generator=self.generator, device="cpu")
        self._pool = self._pool[order.to(self._pool.device)]

    def next_batch(self) -> Float[Tensor, "batch d_model"]:
        if self._count <= self.capacity // 2:
            self._refill()
        self._count -= self.batch_rows
        return self._pool[self._count : self._count + self.batch_rows]
