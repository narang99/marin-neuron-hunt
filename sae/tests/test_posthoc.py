import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from marin_sae.posthoc.examples import TopExamples, choose_latents, window
from marin_sae.posthoc.spliced import loss_recovered, loss_with_replacement

BOS = 0


def tiny_model():
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).eval()


def tokens():
    out = torch.randint(1, 64, (4, 12))
    out[:, 0] = BOS
    return out


def test_an_identity_splice_leaves_the_loss_unchanged():
    model, batch = tiny_model(), tokens()
    layer = model.model.layers[1]
    clean = loss_with_replacement(model, layer, batch, BOS, None)
    spliced = loss_with_replacement(model, layer, batch, BOS, lambda hidden: hidden)
    assert spliced == pytest.approx(clean, abs=1e-5)


def test_zeroing_the_residual_changes_the_loss():
    model, batch = tiny_model(), tokens()
    layer = model.model.layers[1]
    clean = loss_with_replacement(model, layer, batch, BOS, None)
    ablated = loss_with_replacement(model, layer, batch, BOS, torch.zeros_like)
    assert abs(ablated - clean) > 1e-3


def test_loss_recovered_is_one_for_a_perfect_splice_and_zero_for_the_floor():
    assert loss_recovered(clean=2.0, ablated=5.0, spliced=2.0) == 1.0
    assert loss_recovered(clean=2.0, ablated=5.0, spliced=5.0) == 0.0
    assert loss_recovered(clean=2.0, ablated=5.0, spliced=3.5) == pytest.approx(0.5)


def test_top_examples_keeps_only_the_best_per_latent():
    store = TopExamples([3], per_latent=2)
    for activation in (1.0, 5.0, 3.0, 4.0):
        store.offer(3, activation, f"a{activation}")
    assert [e.activation for e in store.best(3)] == [5.0, 4.0]


def test_chosen_latents_come_only_from_those_that_fired_and_split_by_prefix():
    fired = torch.zeros(100, dtype=torch.long)
    fired[[2, 5, 60, 70, 80, 90]] = 1
    chosen = choose_latents(
        fired, n_total=4, first_prefix=10, generator=torch.Generator().manual_seed(0)
    )
    assert set(chosen) <= {2, 5, 60, 70, 80, 90}
    assert sum(latent < 10 for latent in chosen) == 2


def test_the_window_marks_the_firing_token():
    decode = lambda ids: " ".join(map(str, ids))
    text = window(torch.arange(10), position=4, context=2, decode=decode)
    assert text == "2 3[[4]]5 6 7 8 9"


def test_the_posthoc_runner_imports():
    """`run.py` needs the full model, so no other test imports it."""
    import importlib

    importlib.import_module("marin_sae.posthoc.run")
