from pathlib import Path

import pytest

from ragadapt.data import load_squad_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="session")
def dev_corpus():
    return load_squad_file(FIXTURES / "squad_dev_mini.json")


@pytest.fixture(scope="session")
def train_corpus():
    return load_squad_file(FIXTURES / "squad_train_mini.json")
