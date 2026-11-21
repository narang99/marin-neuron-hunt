"""In-run evaluation on a cached block of held-out activation rows.

- Every variant is scored at every Matryoshka prefix, so the comparison shows what nesting buys.
  A vanilla SAE has no reason to reconstruct well from its first 2k latents. A Matryoshka one does.
- The fixed-threshold path is used, which is how the SAE runs at inference.
"""

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from marin_sae.train.variants import Variant

EVAL_CHUNK = 4096


@torch.no_grad()
def evaluate(
    variants: list[Variant],
    rows: Float[Tensor, "row d_in"],
    prefixes: list[int],
    rows_seen: int,
    dead_after: int,
) -> dict[str, dict[str, float]]:
    mean = reduce(rows.float(), "row d_in -> d_in", "mean")
    total_variance = sum(
        (chunk.float() - mean).pow(2).sum() for chunk in rows.split(EVAL_CHUNK)
    )
    results: dict[str, dict[str, float]] = {}
    for variant in variants:
        sae = variant.sae.eval()
        errors = torch.zeros(len(prefixes), device=rows.device)
        l0 = 0.0
        for chunk in rows.split(EVAL_CHUNK):
            x = chunk.float()
            with torch.autocast(rows.device.type, dtype=torch.bfloat16):
                pre = sae.pre_activations(x)
            active = pre.float() * (pre.float() >= sae.threshold)
            l0 += (active > 0).float().sum().item()
            for i, prefix in enumerate(prefixes):
                x_hat = active[:, :prefix] @ sae.W_dec[:prefix] + sae.b_dec
                errors[i] += (x - x_hat).pow(2).sum()
        fvus = (errors / total_variance).tolist()
        dead = ((rows_seen - sae.last_fired) > dead_after).float().mean().item()
        results[variant.name] = {
            **{f"fvu_{prefix}": fvu for prefix, fvu in zip(prefixes, fvus)},
            "l0": l0 / len(rows),
            "dead_fraction": dead,
        }
        sae.train()
    return results
