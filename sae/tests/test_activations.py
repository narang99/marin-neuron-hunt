import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from marin_sae.activations.buffer import ActivationBuffer
from marin_sae.activations.normalisation import fit_normalisation
from marin_sae.activations.rows import residual_rows
from marin_sae.config import ModelConfig
from marin_sae.model.prefix import ResidualTap, is_in_prefix, load_prefix

BOS = 0
D_MODEL = 16


def tiny_checkpoint(tmp_path):
    config = Qwen3Config(
        hidden_size=D_MODEL,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    return model


def test_prefix_keeps_the_embedding_and_layers_up_to_the_hooked_one():
    assert is_in_prefix("model.embed_tokens.weight", layer=1)
    assert is_in_prefix("model.layers.1.mlp.up_proj.weight", layer=1)
    assert not is_in_prefix("model.layers.2.mlp.up_proj.weight", layer=1)
    assert not is_in_prefix("lm_head.weight", layer=1)
    assert not is_in_prefix("model.layers.10.mlp.up_proj.weight", layer=1)


def test_the_tap_matches_the_full_model_hidden_state_after_that_layer(tmp_path):
    full = tiny_checkpoint(tmp_path)
    model_config = ModelConfig(shard_dir=tmp_path, layer=1, dtype="float32")
    tap = ResidualTap(load_prefix(model_config, device="cpu"), layer=1)
    tokens = torch.randint(1, 64, (2, 10))
    with torch.no_grad():
        expected = full.model(
            input_ids=tokens, output_hidden_states=True
        ).hidden_states[2]
    torch.testing.assert_close(tap(tokens), expected, atol=1e-5, rtol=1e-5)


def test_rows_drop_bos_positions():
    tokens = torch.tensor([[BOS, 5, 6], [7, BOS, 8]])
    residual_fn = lambda t: t.unsqueeze(-1).float().expand(-1, -1, D_MODEL)
    rows = residual_rows(residual_fn, tokens, BOS)
    assert rows[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0]


def token_stream(batch=4, seq=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(1, 50, (batch, seq), generator=generator)


def fake_residual(tokens):
    return tokens.unsqueeze(-1).float().expand(-1, -1, D_MODEL) * 0.5


def test_normalisation_makes_the_mean_norm_sqrt_d_model():
    norm = fit_normalisation(
        fake_residual, token_stream(), BOS, n_rows=200, d_model=D_MODEL
    )
    rows = residual_rows(fake_residual, next(token_stream()), BOS) * norm.scale
    assert rows.norm(dim=-1).mean().item() == pytest.approx(D_MODEL**0.5, rel=0.25)


def make_buffer(capacity=64, batch_rows=8):
    norm = fit_normalisation(
        fake_residual, token_stream(), BOS, n_rows=100, d_model=D_MODEL
    )
    return ActivationBuffer(
        fake_residual,
        token_stream(seed=1),
        BOS,
        norm,
        capacity,
        batch_rows,
        D_MODEL,
        device="cpu",
        generator=torch.Generator().manual_seed(0),
    )


def test_batches_have_the_requested_shape_and_the_buffer_keeps_refilling():
    buffer = make_buffer()
    for _ in range(40):
        assert buffer.next_batch().shape == (8, D_MODEL)
    assert buffer.rows_loaded > buffer.capacity


def test_the_buffer_rejects_a_capacity_below_two_batches():
    norm = fit_normalisation(
        fake_residual, token_stream(), BOS, n_rows=50, d_model=D_MODEL
    )
    with pytest.raises(ValueError, match="at least two batches"):
        ActivationBuffer(
            fake_residual,
            token_stream(),
            BOS,
            norm,
            8,
            8,
            D_MODEL,
            "cpu",
            torch.Generator(),
        )


def test_a_batch_mixes_rows_from_many_sequences():
    buffer = make_buffer(capacity=128, batch_rows=16)
    first_columns = buffer.next_batch()[:, 0]
    assert first_columns.unique().numel() > 4
