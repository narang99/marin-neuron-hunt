"""Post-hoc evaluation of a finished run: top examples first (needs only the prefix), then spliced-in loss.

- Run on Colab after training. The full model takes about 65 GB, so the prefix is freed before it loads.
- Results go to `posthoc_results.json` and `examples.md` in the run's out_dir.
"""

import gc
import json

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from marin_sae.activations.normalisation import Normalisation
from marin_sae.config import RunConfig
from marin_sae.data.mix import load_tokenizer, token_batches
from marin_sae.dictionary.batchtopk import BatchTopKSae
from marin_sae.model.full import load_full_model
from marin_sae.model.prefix import ResidualTap, load_prefix
from marin_sae.posthoc.examples import TopExamples, choose_latents, window
from marin_sae.posthoc.spliced import Replacement, loss_recovered, loss_with_replacement
from marin_sae.train.checkpoint import load_checkpoint
from marin_sae.train.variants import build_variants

CHUNK_ROWS = 4096


def active_at_threshold(
    sae: BatchTopKSae, rows: Float[Tensor, "row d_in"], scale: float
) -> Float[Tensor, "row latent"]:
    with torch.autocast(rows.device.type, dtype=torch.bfloat16):
        pre = sae.pre_activations(rows.float() * scale)
    pre = pre.float()
    return pre * (pre >= sae.threshold)


def sae_replacement(sae: BatchTopKSae, scale: float, prefix: int) -> Replacement:
    """Residual -> SAE reconstruction using only the first `prefix` latents, in the model's own units."""

    def replace(
        hidden: Float[Tensor, "batch seq d_model"],
    ) -> Float[Tensor, "batch seq d_model"]:
        flat = rearrange(hidden, "batch seq d_model -> (batch seq) d_model")
        pieces = []
        for chunk in flat.split(CHUNK_ROWS):
            active = active_at_threshold(sae, chunk, scale)
            pieces.append((active[:, :prefix] @ sae.W_dec[:prefix] + sae.b_dec) / scale)
        return rearrange(
            torch.cat(pieces),
            "(batch seq) d_model -> batch seq d_model",
            batch=hidden.shape[0],
        )

    return replace


def load_trained(config: RunConfig):
    variants = build_variants(config, torch.zeros(config.sae.d_in), config.train.device)
    normalisation, _ = load_checkpoint(
        config.posthoc.weights_tar, config.train.out_dir / "unpacked", variants
    )
    return normalisation, {v.name: v.sae.eval() for v in variants}


@torch.no_grad()
def scan_examples(
    config: RunConfig,
    tap: ResidualTap,
    tokenizer,
    saes: dict[str, BatchTopKSae],
    normalisation: Normalisation,
) -> str:
    post, device = config.posthoc, config.train.device
    stream, _ = token_batches(
        config.data, tokenizer, "eval", config.train.sequences_per_forward
    )
    fires = {
        name: torch.zeros(sae.n_latents, dtype=torch.long, device=device)
        for name, sae in saes.items()
    }
    scanned, batches = 0, []
    while scanned < post.example_tokens:
        tokens = next(stream)
        batches.append(tokens)
        scanned += tokens.numel()
        rows = rearrange(
            tap(tokens.to(device)), "batch seq d_model -> (batch seq) d_model"
        )
        for name, sae in saes.items():
            for chunk in rows.split(CHUNK_ROWS):
                fires[name] += (
                    active_at_threshold(sae, chunk, normalisation.scale) > 0
                ).sum(0)
    generator = torch.Generator().manual_seed(config.train.seed)
    watch = {
        name: choose_latents(
            f.cpu(),
            post.n_example_latents,
            config.sae.matryoshka_prefixes[0],
            generator,
        )
        for name, f in fires.items()
    }
    stores = {name: TopExamples(watch[name], post.examples_per_latent) for name in saes}
    for tokens in batches:
        flat_tokens = tokens.flatten()
        rows = rearrange(
            tap(tokens.to(device)), "batch seq d_model -> (batch seq) d_model"
        )
        for name, sae in saes.items():
            for start in range(0, len(rows), CHUNK_ROWS):
                chunk = rows[start : start + CHUNK_ROWS]
                active = active_at_threshold(sae, chunk, normalisation.scale)[
                    :, watch[name]
                ]
                best, where = active.topk(
                    min(post.examples_per_latent, len(chunk)), dim=0
                )
                for column, latent in enumerate(watch[name]):
                    for value, row in zip(
                        best[:, column].tolist(), where[:, column].tolist()
                    ):
                        if value <= 0:
                            continue
                        index = start + row
                        seq_start = (index // tokens.shape[1]) * tokens.shape[1]
                        text = window(
                            flat_tokens[seq_start : seq_start + tokens.shape[1]],
                            index - seq_start,
                            post.context_tokens,
                            tokenizer.decode,
                        )
                        stores[name].offer(latent, value, text.replace("\n", " "))
    return render_examples(
        saes, fires, watch, stores, scanned, config.sae.matryoshka_prefixes[0]
    )


def render_examples(saes, fires, watch, stores, scanned: int, first_prefix: int) -> str:
    lines = [f"# Top examples, {scanned:,} held-out tokens scanned", ""]
    for name in saes:
        counts = fires[name].cpu()
        alive = counts > 0
        lines += [
            f"## {name}",
            f"- alive on held-out tokens: {alive.float().mean():.1%}",
            f"- median firing rate among alive: {(counts[alive].float() / scanned).median():.2e}",
            "",
        ]
        for latent in watch[name]:
            where = "first prefix" if latent < first_prefix else "later"
            lines += [
                f"### latent {latent} ({where}, fires on {counts[latent] / scanned:.2e} of tokens)"
            ]
            lines += [
                f"- {e.activation:.2f}: {e.text}" for e in stores[name].best(latent)
            ] + [""]
    return "\n".join(lines)


def spliced_losses(
    config: RunConfig,
    saes: dict[str, BatchTopKSae],
    normalisation: Normalisation,
    tokenizer,
) -> dict:
    device, bos_id = config.train.device, tokenizer.bos_token_id
    full = load_full_model(config.model, device)
    layer_module = full.model.layers[config.model.layer]
    stream, _ = token_batches(
        config.data, tokenizer, "eval", config.train.sequences_per_forward
    )
    n_batches = max(1, config.posthoc.n_sequences // config.train.sequences_per_forward)
    raw_mean = (normalisation.mean / normalisation.scale).to(device)
    totals: dict[str, float] = {}

    def add(key: str, value: float) -> None:
        totals[key] = totals.get(key, 0.0) + value / n_batches

    for _ in range(n_batches):
        tokens = next(stream).to(device)
        add("clean", loss_with_replacement(full, layer_module, tokens, bos_id, None))
        add(
            "mean_ablated",
            loss_with_replacement(
                full, layer_module, tokens, bos_id, lambda h: raw_mean.expand_as(h)
            ),
        )
        for name, sae in saes.items():
            for prefix in config.sae.matryoshka_prefixes:
                replace = sae_replacement(sae, normalisation.scale, prefix)
                add(
                    f"{name}/{prefix}",
                    loss_with_replacement(full, layer_module, tokens, bos_id, replace),
                )
    results = {
        "clean": totals["clean"],
        "mean_ablated": totals["mean_ablated"],
        "sequences": n_batches * config.train.sequences_per_forward,
    }
    for key, value in totals.items():
        if "/" in key:
            results[key] = {
                "loss": value,
                "recovered": loss_recovered(
                    totals["clean"], totals["mean_ablated"], value
                ),
            }
    del full, layer_module
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run(config: RunConfig) -> None:
    out = config.train.out_dir
    tokenizer = load_tokenizer(config.model)
    normalisation, saes = load_trained(config)
    tap = ResidualTap(
        load_prefix(config.model, config.train.device), config.model.layer
    )
    (out / "examples.md").write_text(
        scan_examples(config, tap, tokenizer, saes, normalisation)
    )
    del tap
    gc.collect()
    torch.cuda.empty_cache()
    results = spliced_losses(config, saes, normalisation, tokenizer)
    (out / "posthoc_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
