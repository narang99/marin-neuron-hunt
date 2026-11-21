"""Typed settings for one SAE run.

- Every knob lives here and in a YAML file under `configs/`. There are no ad hoc flags or env vars.
- Precedence is CLI flags over YAML over defaults.
- Usage: `python -m marin_sae.train --config configs/layer8_mix_300m.yaml --train.lr 1e-4`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from marin_sae.device import default_device

SaeVariant = Literal["vanilla", "matryoshka"]
"""Both variants share the architecture. Only the training loss differs."""

WEIGHT_TOTAL = 100.0
WEIGHT_TOLERANCE = 1e-6


class Section(BaseModel):
    """A typo in a config file should fail, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class MixSource(Section):
    name: str = Field(description="short label used in logs and per-source stats")
    weight: float = Field(gt=0, description="share of training tokens, in percent")
    repo: str = Field(description="Hugging Face dataset repo")
    config_name: str | None = Field(
        None, description="dataset config, e.g. a FineWeb sample"
    )
    data_dir: str | None = None
    data_files: str | None = Field(
        None, description="glob inside the repo, for sharded data"
    )
    text_field: str = "text"


class DataConfig(Section):
    sources: list[MixSource]
    seq_len: int = Field(1024, gt=1, description="tokens per sequence, BOS included")
    max_doc_tokens: int = Field(
        4096,
        gt=1,
        description="longer documents are cut, so one paper cannot fill many sequences",
    )
    shuffle_buffer: int = Field(
        1000,
        gt=0,
        description="documents buffered per source for shuffling. Larger starts slower",
    )
    seed: int = Field(20261120, description="fixes the mix draw and document order")
    train_tokens: int = Field(300_000_000, gt=0)
    eval_tokens: int = Field(2_000_000, gt=0)

    @model_validator(mode="after")
    def weights_sum_to_one_hundred(self) -> "DataConfig":
        total = sum(source.weight for source in self.sources)
        if abs(total - WEIGHT_TOTAL) > WEIGHT_TOLERANCE:
            raise ValueError(f"source weights sum to {total}, expected {WEIGHT_TOTAL}")
        return self

    @model_validator(mode="after")
    def source_names_are_unique(self) -> "DataConfig":
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")
        return self

    @property
    def shares(self) -> dict[str, float]:
        return {source.name: source.weight / WEIGHT_TOTAL for source in self.sources}


class ModelConfig(Section):
    hf_id: str = "marin-community/marin-32b-base"
    layer: int = Field(
        8, ge=0, description="0-indexed decoder layer whose output is hooked"
    )
    shard_dir: Path = Field(
        Path("/content/marin-32b-shards"), description="local safetensors shards"
    )
    dtype: Literal["bfloat16", "float32"] = Field(
        "bfloat16", description="dtype of the frozen prefix"
    )


class SaeConfig(Section):
    d_in: int = Field(5120, gt=0)
    n_latents: int = Field(65_536, gt=0)
    k: int = Field(64, gt=0, description="average active latents per token")
    candidates_per_token: int = Field(
        128, gt=0, description="per-token pool the batch-wide top-k is taken from"
    )
    matryoshka_prefixes: list[int] = Field(
        [2_048, 8_192, 32_768, 65_536], description="nested dictionary sizes, ascending"
    )

    @model_validator(mode="after")
    def prefixes_nest_inside_the_dictionary(self) -> "SaeConfig":
        prefixes = self.matryoshka_prefixes
        if prefixes != sorted(set(prefixes)):
            raise ValueError(f"prefixes must be strictly ascending, got {prefixes}")
        if prefixes[-1] != self.n_latents:
            raise ValueError(
                f"last prefix {prefixes[-1]} must equal n_latents {self.n_latents}"
            )
        return self

    @model_validator(mode="after")
    def pool_is_at_least_k(self) -> "SaeConfig":
        if self.candidates_per_token < self.k:
            raise ValueError("candidates_per_token must be >= k")
        return self


class TrainConfig(Section):
    out_dir: Path = Path("/content/runs")
    batch_tokens: int = Field(4096, gt=0)
    sequences_per_forward: int = Field(
        32, gt=0, description="documents per LLM forward pass"
    )
    buffer_tokens: int = Field(
        1_000_000, gt=0, description="shuffle buffer, in activation rows"
    )
    norm_tokens: int = Field(
        100_000, gt=0, description="rows used to fix the scalar scale and b_dec"
    )
    lr: float = Field(2e-4, gt=0)
    warmup_steps: int = Field(500, ge=0)
    decay_fraction: float = Field(
        0.2, ge=0, lt=1, description="final share of steps with linear decay"
    )
    auxk_k: int = Field(
        512, gt=0, description="dead latents that reconstruct the residual"
    )
    auxk_coef: float = Field(1 / 32, ge=0)
    dead_after_tokens: int = Field(
        5_000_000, gt=0, description="no firing for this long means dead"
    )
    seed: int = 20261120
    log_every: int = Field(50, gt=0)
    checkpoint_every_tokens: int = Field(25_000_000, gt=0)
    eval_every_tokens: int = Field(50_000_000, gt=0)
    eval_rows: int = Field(
        500_000, gt=0, description="activation rows cached on the GPU for in-run eval"
    )
    keep_every_tokens: int = Field(
        100_000_000, gt=0, description="weights-only milestones kept on the archive"
    )
    archive_dir: Path | None = Field(
        None,
        description="e.g. a Drive folder. Checkpoints are tarred, then copied here",
    )
    resume_from: Path | None = Field(
        None, description="a resume tar written by an earlier run"
    )
    device: str = Field(default_factory=default_device)


class PosthocConfig(Section):
    weights_tar: Path | None = Field(
        None, description="a weights or resume tar from a finished run"
    )
    n_sequences: int = Field(
        256, gt=0, description="held-out sequences for the spliced-in loss"
    )
    example_tokens: int = Field(
        2_000_000, gt=0, description="held-out tokens scanned for top examples"
    )
    n_example_latents: int = Field(24, gt=0)
    examples_per_latent: int = Field(8, gt=0)
    context_tokens: int = Field(
        24, gt=0, description="tokens of context shown around an example"
    )


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True)

    data: DataConfig
    model: ModelConfig = ModelConfig()
    sae: SaeConfig = SaeConfig()
    train: TrainConfig = TrainConfig()
    posthoc: PosthocConfig = PosthocConfig()
    variants: list[SaeVariant] = Field(
        ["vanilla", "matryoshka"],
        description="SAEs trained side by side on the same activations",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        **_: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Ignore env vars: a stray `DEVICE` would silently override a field."""
        return (init_settings,)

    @property
    def total_steps(self) -> int:
        return self.data.train_tokens // self.train.batch_tokens


def config_path_from(argv: Sequence[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    known, _ = parser.parse_known_args(list(argv))
    return known.config


def load_run_config(argv: Sequence[str]) -> RunConfig:
    path = config_path_from(argv)
    from_yaml = yaml.safe_load(path.read_text()) if path else {}
    rest = _without_config_flag(argv)
    return RunConfig(**from_yaml, _cli_parse_args=rest)


def _without_config_flag(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
        elif token == "--config":
            skip = True
        elif not token.startswith("--config="):
            out.append(token)
    return out
