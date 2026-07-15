import os
import sys
from pathlib import Path

os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_config import resolve_model_id


def test_resolve_llama_model_aliases():
    assert resolve_model_id("llama_3_2_1b") == "meta-llama/Llama-3.2-1B"
    assert resolve_model_id("llama_3_2_3b") == "meta-llama/Llama-3.2-3B"


def test_resolve_phi_model_aliases():
    assert resolve_model_id("phi_4_mini_instruct") == "microsoft/Phi-4-mini-instruct"
    assert resolve_model_id("phi_4_mini_reasoning") == "microsoft/Phi-4-mini-reasoning"
