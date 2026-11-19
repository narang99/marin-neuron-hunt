"""Entrypoint: python -m marin_sae.train --config configs/layer8_mix_300m.yaml.

The process is ended with os._exit on purpose.
- Hugging Face `datasets` keeps non-daemon download threads alive.
- After a crash they keep the interpreter running, so a failed run looks alive forever.
- After a clean finish they delay the exit for no reason.
"""

import os
import sys
import traceback

from marin_sae.config import load_run_config
from marin_sae.train.run import run


def main() -> None:
    code = 0
    try:
        run(load_run_config(sys.argv[1:]))
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
