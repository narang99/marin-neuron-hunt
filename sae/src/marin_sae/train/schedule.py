"""Learning-rate schedule: linear warmup, flat, then a linear decay to zero over the last steps."""


def learning_rate(
    step: int, total_steps: int, base: float, warmup_steps: int, decay_fraction: float
) -> float:
    if step < warmup_steps:
        return base * (step + 1) / warmup_steps
    decay_start = int(total_steps * (1 - decay_fraction))
    if step >= decay_start:
        return base * max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
    return base
