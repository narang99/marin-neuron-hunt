import sys

from marin_sae.config import load_run_config
from marin_sae.posthoc.run import run

if __name__ == "__main__":
    run(load_run_config(sys.argv[1:]))
