import pytest
import torch

from marin_sae.activations.buffer import ActivationBuffer
from marin_sae.activations.normalisation import Normalisation
from marin_sae.config import DataConfig, MixSource, RunConfig, SaeConfig, TrainConfig
from marin_sae.train.checkpoint import (
    TrainState,
    load_checkpoint,
    save_checkpoint,
    tar_directory,
)
from marin_sae.train.evaluate import evaluate
from marin_sae.train.loop import train
from marin_sae.train.runlog import RunLog
from marin_sae.train.schedule import learning_rate
from marin_sae.train.variants import build_variants

D_IN = 16


def tiny_config(tmp_path, tokens=64 * 400) -> RunConfig:
    return RunConfig(
        data=DataConfig(
            sources=[MixSource(name="a", weight=100, repo="x/y")], train_tokens=tokens
        ),
        sae=SaeConfig(
            d_in=D_IN,
            n_latents=64,
            k=4,
            candidates_per_token=12,
            matryoshka_prefixes=[8, 24, 64],
        ),
        train=TrainConfig(
            out_dir=tmp_path / "run",
            batch_tokens=64,
            lr=3e-3,
            warmup_steps=5,
            log_every=50,
            checkpoint_every_tokens=64 * 200,
            eval_every_tokens=64 * 200,
            keep_every_tokens=64 * 200,
            dead_after_tokens=64 * 50,
            device="cpu",
            eval_rows=512,
            norm_tokens=256,
        ),
        _cli_parse_args=False,
    )


def features_stream(generator, features, batch=8, seq=16):
    while True:
        pick = torch.randint(0, len(features), (batch, seq, 2), generator=generator)
        yield features[pick].sum(dim=2)


def build_buffer(config, features):
    stream = features_stream(torch.Generator().manual_seed(2), features)
    tokens = (torch.arange(8 * 16).reshape(8, 16) % 50) + 1

    def residual_fn(_tokens):
        return next(stream)

    normalisation = Normalisation(scale=1.0, mean=torch.zeros(D_IN))
    buffer = ActivationBuffer(
        residual_fn,
        iter(lambda: tokens, None),
        0,
        normalisation,
        1024,
        64,
        D_IN,
        "cpu",
        torch.Generator().manual_seed(0),
    )
    return buffer, normalisation


def test_learning_rate_warms_up_holds_then_decays_to_zero():
    lrs = [learning_rate(s, 100, 1.0, 10, 0.2) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[50] == 1.0
    assert lrs[79] == 1.0
    assert lrs[99] == pytest.approx(0.05)
    assert lrs[90] < lrs[85]


def test_both_variants_train_and_matryoshka_wins_at_the_small_prefix(tmp_path):
    config = tiny_config(tmp_path)
    features = torch.nn.functional.normalize(
        torch.randn(24, D_IN, generator=torch.Generator().manual_seed(1)), dim=-1
    )
    buffer, normalisation = build_buffer(config, features)
    variants = build_variants(config, torch.zeros(D_IN), "cpu")
    eval_rows = torch.cat(
        [
            next(features_stream(torch.Generator().manual_seed(9), features)).reshape(
                -1, D_IN
            )
        ]
        * 4
    )
    state = train(
        config,
        buffer,
        variants,
        normalisation,
        eval_rows,
        TrainState(step=0, rows_seen=0),
        RunLog(config.train.out_dir),
    )
    assert state.step == config.total_steps
    result = evaluate(
        variants,
        eval_rows,
        [8, 24, 64],
        state.rows_seen,
        config.train.dead_after_tokens,
    )
    for name in ("vanilla", "matryoshka"):
        assert result[name]["fvu_64"] < 0.8
    assert result["matryoshka"]["fvu_8"] <= result["vanilla"]["fvu_8"] + 0.05


def test_a_resume_tar_restores_weights_optimiser_and_state(tmp_path):
    config = tiny_config(tmp_path)
    variants = build_variants(config, torch.zeros(D_IN), "cpu")
    x = torch.randn(64, D_IN)
    from marin_sae.train.loop import train_step

    train_step(variants[0], x, 1e-3, config, rows_seen=64)
    normalisation = Normalisation(scale=2.5, mean=torch.ones(D_IN))
    archive = tar_directory(
        save_checkpoint(
            tmp_path / "resume",
            variants,
            normalisation,
            TrainState(step=7, rows_seen=448),
            True,
        )
    )
    fresh = build_variants(config, torch.zeros(D_IN), "cpu")
    restored_norm, state = load_checkpoint(archive, tmp_path / "unpacked", fresh)
    assert state.step == 7 and restored_norm.scale == 2.5
    torch.testing.assert_close(fresh[0].sae.W_dec, variants[0].sae.W_dec)
    assert len(fresh[0].optimiser.state) > 0
