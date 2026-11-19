"""Packs token-id documents into fixed-length sequences, with a BOS at every document start."""

from collections.abc import Iterable, Iterator

import torch
from jaxtyping import Int
from torch import Tensor


def pack_documents(
    documents: Iterable[list[int]], seq_len: int, bos_id: int
) -> Iterator[Int[Tensor, "seq"]]:
    """Concatenate documents and cut them into `seq_len` chunks.

    - A document can span two sequences, and a sequence can hold several documents.
    - Attention is not masked across documents, which matches how packed pretraining sees text.
    - The tail shorter than `seq_len` is dropped when the input ends.
    """
    pending: list[int] = []
    for document in documents:
        pending.append(bos_id)
        pending.extend(document)
        while len(pending) >= seq_len:
            yield torch.tensor(pending[:seq_len], dtype=torch.long)
            del pending[:seq_len]


def batch_sequences(
    sequences: Iterable[Int[Tensor, "seq"]], batch_size: int
) -> Iterator[Int[Tensor, "batch seq"]]:
    group: list[Int[Tensor, "seq"]] = []
    for sequence in sequences:
        group.append(sequence)
        if len(group) == batch_size:
            yield torch.stack(group)
            group = []
