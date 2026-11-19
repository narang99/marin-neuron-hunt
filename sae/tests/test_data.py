import itertools

import pytest
import torch

from marin_sae.data.mix import mixed_documents
from marin_sae.data.packing import batch_sequences, pack_documents
from marin_sae.data.scheduler import ShareScheduler

BOS = 99


def test_every_document_starts_with_bos_and_sequences_have_fixed_length():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    sequences = list(pack_documents(docs, seq_len=4, bos_id=BOS))
    assert [len(s) for s in sequences] == [4, 4, 4]
    flat = torch.cat(sequences).tolist()
    assert flat == [BOS, 1, 2, 3, BOS, 4, 5, BOS, 6, 7, 8, 9]


def test_a_document_longer_than_a_sequence_spans_several():
    sequences = list(pack_documents([list(range(10))], seq_len=4, bos_id=BOS))
    assert len(sequences) == 2


def test_the_incomplete_tail_is_dropped():
    assert list(pack_documents([[1]], seq_len=4, bos_id=BOS)) == []


def test_batches_stack_sequences_and_drop_the_partial_group():
    sequences = [torch.arange(4) + i for i in range(5)]
    batches = list(batch_sequences(sequences, batch_size=2))
    assert [b.shape for b in batches] == [torch.Size([2, 4])] * 2


def test_scheduler_picks_the_source_furthest_below_its_share():
    scheduler = ShareScheduler({"a": 0.75, "b": 0.25})
    assert scheduler.next_source() == "a"
    scheduler.record("a", 100)
    assert scheduler.next_source() == "b"


def test_scheduler_rejects_no_sources():
    with pytest.raises(ValueError):
        ShareScheduler({})


def fake_source(length: int):
    def open_(seed: int):
        return iter(["x" * length] * 1_000_000)

    return open_


def test_realised_token_shares_follow_the_weights_even_with_unequal_lengths():
    scheduler = ShareScheduler({"short": 0.8, "long": 0.2})
    openers = {"short": fake_source(3), "long": fake_source(50)}
    stream = mixed_documents(
        scheduler, openers, lambda text: list(range(len(text))), seed=0
    )
    list(itertools.islice(stream, 2000))
    realised = scheduler.realised_shares
    assert realised["short"] == pytest.approx(0.8, abs=0.02)
    assert realised["long"] == pytest.approx(0.2, abs=0.02)


def test_an_exhausted_source_is_reopened_with_the_next_seed():
    seeds: list[int] = []

    def opener(seed: int):
        seeds.append(seed)
        return iter(["ab"])

    scheduler = ShareScheduler({"only": 1.0})
    stream = mixed_documents(scheduler, {"only": opener}, lambda t: [1, 2], seed=7)
    list(itertools.islice(stream, 3))
    assert seeds == [7, 8, 9]
