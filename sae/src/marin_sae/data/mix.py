"""The frozen training mix as a stream of packed token batches.

- `mixed_documents` is pure and takes its inputs as arguments, so tests need no network.
- `open_source` and `token_batches` are the thin layer over Hugging Face `datasets` and the tokenizer.
"""

from collections.abc import Callable, Iterator
from typing import Literal

from datasets import load_dataset
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from marin_sae.config import DataConfig, MixSource, ModelConfig
from marin_sae.data.packing import batch_sequences, pack_documents
from marin_sae.data.scheduler import ShareScheduler

Split = Literal["train", "eval"]
CHARS_PER_TOKEN_CAP = 8
"""Text is cut at this many characters per allowed token before tokenizing, so a 500k-character paper costs little."""


def open_source(source: MixSource, seed: int, buffer_size: int) -> Iterator[str]:
    """Stream one source's documents in a seeded shuffled order."""
    dataset = load_dataset(
        source.repo,
        name=source.config_name,
        data_dir=source.data_dir,
        data_files=source.data_files,
        split="train",
        streaming=True,
    )
    for row in dataset.shuffle(seed=seed, buffer_size=buffer_size):
        yield row[source.text_field]


def mixed_documents(
    scheduler: ShareScheduler,
    openers: dict[str, Callable[[int], Iterator[str]]],
    encode: Callable[[str], list[int]],
    seed: int,
) -> Iterator[list[int]]:
    """Yield token-id documents so each source's token share follows the scheduler.

    - A source that runs out is reopened with the next seed, so a small source still reaches its share.
    """
    reopened = dict.fromkeys(openers, 0)
    streams = {name: opener(seed) for name, opener in openers.items()}
    while True:
        name = scheduler.next_source()
        try:
            text = next(streams[name])
        except StopIteration:
            reopened[name] += 1
            streams[name] = openers[name](seed + reopened[name])
            text = next(streams[name])
        ids = encode(text)
        if not ids:
            continue
        scheduler.record(name, len(ids) + 1)
        yield ids


def load_tokenizer(model: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model.shard_dir)
    if tokenizer.bos_token_id is None:
        raise ValueError(
            f"{model.hf_id} tokenizer has no BOS token, which packing relies on"
        )
    return tokenizer


def token_batches(
    data: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: Split,
    batch_size: int,
) -> tuple[Iterator[Int[Tensor, "batch seq"]], ShareScheduler]:
    """Packed `batch seq` token batches for `split`, and the scheduler that reports realised shares.

    - The eval split uses a different seed, so its order differs from training.
      The sources are far larger than what is read, so overlap is negligible but not zero.
    """
    seed = data.seed if split == "train" else data.seed + 1

    def encode(text: str) -> list[int]:
        clipped = text[: data.max_doc_tokens * CHARS_PER_TOKEN_CAP]
        ids = tokenizer(clipped, add_special_tokens=False)["input_ids"]
        return ids[: data.max_doc_tokens - 1]

    scheduler = ShareScheduler(data.shares)
    openers = {
        source.name: (
            lambda s, source=source: open_source(source, s, data.shuffle_buffer)
        )
        for source in data.sources
    }
    documents = mixed_documents(scheduler, openers, encode, seed)
    sequences = pack_documents(documents, data.seq_len, tokenizer.bos_token_id)
    return batch_sequences(sequences, batch_size), scheduler
