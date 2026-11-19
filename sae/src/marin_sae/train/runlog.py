"""Progress goes to stdout as `[tag] json` lines and to a jsonl file that survives the notebook."""

import json
from pathlib import Path


class RunLog:
    def __init__(self, out_dir: Path, name: str = "train_log.jsonl") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / name

    def write(self, tag: str, record: dict) -> None:
        line = json.dumps(record)
        print(f"[{tag}] {line}", flush=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"tag": tag, **record}) + "\n")
