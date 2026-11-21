"""Checkpoints are one tar per save, because thousands of small files are slow on Drive.

- `resume` tars hold weights and optimiser state, and are overwritten on the archive each time.
- `weights` tars hold weights only, and milestones are kept.
- The copy to the archive runs in a background thread, so training does not wait for Drive.
"""

import json
import shutil
import tarfile
import threading
from pathlib import Path

import torch
from pydantic import BaseModel

from marin_sae.activations.normalisation import Normalisation
from marin_sae.train.variants import Variant

STATE_FILE = "state.json"
NORMALISATION_FILE = "normalisation.pt"


class TrainState(BaseModel):
    step: int
    rows_seen: int
    resume_count: int = 0
    """How many times the run was resumed. It offsets the data seed so a resume reads fresh text."""


def save_checkpoint(
    directory: Path,
    variants: list[Variant],
    normalisation: Normalisation,
    state: TrainState,
    with_optimiser: bool,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        payload = {"sae": variant.sae.state_dict()}
        if with_optimiser:
            payload["optimiser"] = variant.optimiser.state_dict()
        torch.save(payload, directory / f"{variant.name}.pt")
    torch.save(
        {"scale": normalisation.scale, "mean": normalisation.mean.cpu()},
        directory / NORMALISATION_FILE,
    )
    (directory / STATE_FILE).write_text(state.model_dump_json())
    return directory


def tar_directory(directory: Path) -> Path:
    archive = directory.with_suffix(".tar")
    with tarfile.open(archive, "w") as handle:
        handle.add(directory, arcname=directory.name)
    shutil.rmtree(directory)
    return archive


def copy_in_background(
    archive: Path, destination: Path, keep_local: bool
) -> threading.Thread:
    """Copy via a temporary name, then rename, so the archive never holds a half-written tar."""

    def work() -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".partial")
        shutil.copyfile(archive, partial)
        partial.replace(destination)
        print(
            f"[archive] {destination.name} {destination.stat().st_size / 2**30:.2f} GiB",
            flush=True,
        )
        if not keep_local:
            archive.unlink()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def load_checkpoint(
    tar_path: Path, workdir: Path, variants: list[Variant]
) -> tuple[Normalisation, TrainState]:
    with tarfile.open(tar_path) as handle:
        handle.extractall(workdir, filter="data")
    directory = workdir / tar_path.stem
    for variant in variants:
        payload = torch.load(directory / f"{variant.name}.pt", map_location="cpu")
        variant.sae.load_state_dict(payload["sae"])
        variant.optimiser.load_state_dict(payload["optimiser"])
    saved = torch.load(directory / NORMALISATION_FILE, map_location="cpu")
    state = TrainState(**json.loads((directory / STATE_FILE).read_text()))
    return Normalisation(scale=saved["scale"], mean=saved["mean"]), state
