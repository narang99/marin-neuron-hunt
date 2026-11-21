"""One optimisation step per variant on a shared batch, and the loop around it."""

import time
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor

from marin_sae.activations.buffer import ActivationBuffer
from marin_sae.activations.normalisation import Normalisation
from marin_sae.config import RunConfig
from marin_sae.dictionary.loss import auxk_loss, reconstruction_loss
from marin_sae.train.checkpoint import (
    TrainState,
    copy_in_background,
    save_checkpoint,
    tar_directory,
)
from marin_sae.train.evaluate import evaluate
from marin_sae.train.runlog import RunLog
from marin_sae.train.schedule import learning_rate
from marin_sae.train.variants import Variant


class StepStats(NamedTuple):
    loss: Float[Tensor, ""]
    fvu_full: Float[Tensor, ""]
    auxk: Float[Tensor, ""]
    dead_fraction: Float[Tensor, ""]


def train_step(
    variant: Variant,
    x: Float[Tensor, "batch d_in"],
    lr: float,
    config: RunConfig,
    rows_seen: int,
) -> StepStats:
    sae, train = variant.sae, config.train
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        pre = sae.pre_activations(x)
    pre = pre.float()
    code = sae.select(pre)
    reconstructions = sae.decode(code, variant.edges)
    target = x.float()
    reconstruction = reconstruction_loss(target, reconstructions)
    dead = (rows_seen - sae.last_fired) > train.dead_after_tokens
    auxk = auxk_loss(sae, target, reconstructions[-1], pre, dead, train.auxk_k)
    loss = reconstruction + train.auxk_coef * auxk
    for group in variant.optimiser.param_groups:
        group["lr"] = lr
    variant.optimiser.zero_grad(set_to_none=True)
    loss.backward()
    sae.project_decoder_gradient()
    variant.optimiser.step()
    sae.normalise_decoder()
    sae.mark_fired(code, rows_seen)
    with torch.no_grad():
        full = reconstruction_loss(target, reconstructions[-1:]).detach()
    return StepStats(loss.detach(), full, auxk.detach(), dead.float().mean())


def train(
    config: RunConfig,
    buffer: ActivationBuffer,
    variants: list[Variant],
    normalisation: Normalisation,
    eval_rows: Float[Tensor, "row d_in"],
    state: TrainState,
    log: RunLog,
) -> TrainState:
    train_config, total = config.train, config.total_steps
    next_checkpoint = (
        state.rows_seen // train_config.checkpoint_every_tokens + 1
    ) * train_config.checkpoint_every_tokens
    next_eval = (
        state.rows_seen // train_config.eval_every_tokens + 1
    ) * train_config.eval_every_tokens
    next_milestone = (
        state.rows_seen // train_config.keep_every_tokens + 1
    ) * train_config.keep_every_tokens
    started, rows_at_start = time.time(), state.rows_seen
    copies: list = []
    for step in range(state.step + 1, total + 1):
        x = buffer.next_batch()
        state = state.model_copy(
            update={"step": step, "rows_seen": state.rows_seen + len(x)}
        )
        lr = learning_rate(
            step,
            total,
            train_config.lr,
            train_config.warmup_steps,
            train_config.decay_fraction,
        )
        stats = {
            v.name: train_step(v, x, lr, config, state.rows_seen) for v in variants
        }
        if step == 1 or step % train_config.log_every == 0:
            elapsed = time.time() - started
            log.write(
                "train",
                {
                    "step": step,
                    "rows": state.rows_seen,
                    "lr": lr,
                    "rows_per_s": round((state.rows_seen - rows_at_start) / elapsed, 1),
                    "elapsed_min": round(elapsed / 60, 1),
                    "gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)
                    if torch.cuda.is_available()
                    else 0,
                    **{
                        f"{name}/{key}": round(float(value), 5)
                        for name, s in stats.items()
                        for key, value in s._asdict().items()
                    },
                },
            )
        if state.rows_seen >= next_eval:
            next_eval += train_config.eval_every_tokens
            log.write(
                "eval",
                {
                    "rows": state.rows_seen,
                    "step": step,
                    **{
                        name: result
                        for name, result in evaluate(
                            variants,
                            eval_rows,
                            config.sae.matryoshka_prefixes,
                            state.rows_seen,
                            train_config.dead_after_tokens,
                        ).items()
                    },
                },
            )
        if state.rows_seen >= next_checkpoint:
            next_checkpoint += train_config.checkpoint_every_tokens
            milestone = state.rows_seen >= next_milestone
            if milestone:
                next_milestone += train_config.keep_every_tokens
            copies.extend(checkpoint(config, variants, normalisation, state, milestone))
    for thread in copies:
        thread.join()
    return state


def checkpoint(
    config: RunConfig,
    variants: list[Variant],
    normalisation: Normalisation,
    state: TrainState,
    milestone: bool,
) -> list:
    """Write the resume tar (overwritten), and a weights-only tar on milestones. Returns copy threads."""
    out = config.train.out_dir
    threads = []
    resume = tar_directory(
        save_checkpoint(
            out / "resume", variants, normalisation, state, with_optimiser=True
        )
    )
    if config.train.archive_dir:
        threads.append(
            copy_in_background(
                resume, config.train.archive_dir / "resume.tar", keep_local=False
            )
        )
    if milestone:
        name = f"weights_{state.rows_seen // 1_000_000}M"
        weights = tar_directory(
            save_checkpoint(
                out / name, variants, normalisation, state, with_optimiser=False
            )
        )
        if config.train.archive_dir:
            threads.append(
                copy_in_background(
                    weights, config.train.archive_dir / f"{name}.tar", keep_local=True
                )
            )
    return threads
