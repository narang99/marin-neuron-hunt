import importlib

import pytest


@pytest.mark.parametrize("module", ["marin_sae.train.run", "marin_sae.train.__main__"])
def test_the_entrypoint_modules_import(module):
    """`run.py` needs the real model, so no other test imports it. A bad import there only shows on Colab."""
    importlib.import_module(module)
