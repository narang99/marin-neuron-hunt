from pathlib import Path

import pytest
from pydantic import ValidationError

from marin_sae.config import (
    DataConfig,
    MixSource,
    RunConfig,
    SaeConfig,
    load_run_config,
)

CONFIGS = Path(__file__).parent.parent / "configs"


def source(name: str, weight: float) -> MixSource:
    return MixSource(name=name, weight=weight, repo="org/repo")


def test_source_weights_must_sum_to_one_hundred():
    with pytest.raises(ValidationError, match="sum to 90"):
        DataConfig(sources=[source("a", 60), source("b", 30)])


def test_source_names_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        DataConfig(sources=[source("a", 50), source("a", 50)])


def test_shares_are_fractions_of_one():
    data = DataConfig(sources=[source("a", 75), source("b", 25)])
    assert data.shares == {"a": 0.75, "b": 0.25}


def test_matryoshka_prefixes_must_end_at_the_dictionary_size():
    with pytest.raises(ValidationError, match="must equal n_latents"):
        SaeConfig(n_latents=1024, matryoshka_prefixes=[128, 512])


def test_matryoshka_prefixes_must_ascend():
    with pytest.raises(ValidationError, match="ascending"):
        SaeConfig(n_latents=1024, matryoshka_prefixes=[512, 128, 1024])


def test_a_typo_in_a_section_fails():
    with pytest.raises(ValidationError):
        SaeConfig(n_latent=1024)


def test_the_committed_pilot_config_loads_and_sums_to_one_hundred():
    config = load_run_config(["--config", str(CONFIGS / "layer8_mix_300m.yaml")])
    assert sum(config.data.shares.values()) == pytest.approx(1.0)
    assert config.total_steps == 300_000_000 // 4096


def test_a_flag_overrides_one_key_and_keeps_the_rest():
    argv = ["--config", str(CONFIGS / "layer8_mix_300m.yaml"), "--train.lr", "0.001"]
    config = load_run_config(argv)
    assert config.train.lr == 0.001
    assert config.train.batch_tokens == 4096


def test_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("MODEL", "nonsense")
    config = RunConfig(
        data=DataConfig(sources=[source("a", 100)]), _cli_parse_args=False
    )
    assert config.model.layer == 8
