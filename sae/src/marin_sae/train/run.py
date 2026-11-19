"""Wires the pieces together for one run. Needs a GPU and the real model, so it is run on Colab.

Order matters here:
- A fresh run fits the normalisation from the first tokens, then builds the SAEs around it.
- A resumed run loads the normalisation and weights from the tar and skips the fit, so the scale never changes mid-run.
- The training stream is seeded with `seed + resume_count`, so a resume reads new text, not the first text again.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from marin_sae.activations.buffer import ActivationBuffer
from marin_sae.activations.normalisation import Normalisation, fit_normalisation
from marin_sae.activations.rows import residual_rows
from marin_sae.config import RunConfig
from marin_sae.data.mix import load_tokenizer, token_batches
from marin_sae.model.prefix import ResidualFn, ResidualTap, load_prefix
from marin_sae.train.checkpoint import TrainState, load_checkpoint
from marin_sae.train.evaluate import evaluate
from marin_sae.train.loop import checkpoint, train
from marin_sae.train.runlog import RunLog
from marin_sae.train.variants import Variant, build_variants


def cache_eval_rows(
    config: RunConfig, residual_fn: ResidualFn, tokenizer, scale: float
) -> Float[Tensor, "row d_in"]:
    stream, _ = token_batches(
        config.data, tokenizer, "eval", config.train.sequences_per_forward
    )
    chunks: list[Tensor] = []
    collected = 0
    while collected < config.train.eval_rows:
        tokens = next(stream).to(config.train.device)
        rows = residual_rows(residual_fn, tokens, tokenizer.bos_token_id) * scale
        chunks.append(rows.to(torch.bfloat16))
        collected += len(rows)
    return torch.cat(chunks)[: config.train.eval_rows]


def start_fresh(
    config: RunConfig, tap: ResidualTap, tokenizer, log: RunLog
) -> tuple[Normalisation, list[Variant], TrainState]:
    stream, _ = token_batches(
        config.data, tokenizer, "train", config.train.sequences_per_forward
    )
    normalisation = fit_normalisation(
        tap, stream, tokenizer.bos_token_id, config.train.norm_tokens, config.sae.d_in
    )
    log.write(
        "setup",
        {"scale": normalisation.scale, "mean_norm": float(normalisation.mean.norm())},
    )
    variants = build_variants(
        config, normalisation.mean.to(config.train.device), config.train.device
    )
    return normalisation, variants, TrainState(step=0, rows_seen=0)


def resume(
    config: RunConfig, log: RunLog
) -> tuple[Normalisation, list[Variant], TrainState]:
    variants = build_variants(config, torch.zeros(config.sae.d_in), config.train.device)
    normalisation, state = load_checkpoint(
        config.train.resume_from, config.train.out_dir, variants
    )
    state = state.model_copy(update={"resume_count": state.resume_count + 1})
    log.write("resume", state.model_dump())
    return normalisation, variants, state


def run(config: RunConfig) -> TrainState:
    torch.manual_seed(config.train.seed)
    out = config.train.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(config.model_dump_json(indent=2))
    log = RunLog(out)

    tokenizer = load_tokenizer(config.model)
    tap = ResidualTap(
        load_prefix(config.model, config.train.device), config.model.layer
    )
    if config.train.resume_from:
        normalisation, variants, state = resume(config, log)
    else:
        normalisation, variants, state = start_fresh(config, tap, tokenizer, log)

    eval_rows = cache_eval_rows(config, tap, tokenizer, normalisation.scale)
    data = config.data.model_copy(
        update={"seed": config.data.seed + state.resume_count}
    )
    stream, scheduler = token_batches(
        data, tokenizer, "train", config.train.sequences_per_forward
    )
    buffer = ActivationBuffer(
        tap,
        stream,
        tokenizer.bos_token_id,
        normalisation,
        config.train.buffer_tokens,
        config.train.batch_tokens,
        config.sae.d_in,
        config.train.device,
        torch.Generator().manual_seed(config.train.seed + state.resume_count),
    )
    state = train(config, buffer, variants, normalisation, eval_rows, state, log)

    final = evaluate(
        variants,
        eval_rows,
        config.sae.matryoshka_prefixes,
        state.rows_seen,
        config.train.dead_after_tokens,
    )
    log.write(
        "final",
        {
            "rows": state.rows_seen,
            "realised_shares": scheduler.realised_shares,
            **final,
        },
    )
    for thread in checkpoint(config, variants, normalisation, state, milestone=True):
        thread.join()
    return state
