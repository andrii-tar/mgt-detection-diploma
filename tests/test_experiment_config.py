import os
import sys
from pathlib import Path

os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_config import resolve_dataset_splits, resolve_model_id, resolve_model_name


def test_resolve_llama_model_aliases():
    assert resolve_model_id("llama_3_2_1b") == "meta-llama/Llama-3.2-1B"
    assert resolve_model_id("llama_3_2_3b") == "meta-llama/Llama-3.2-3B"


def test_resolve_phi_model_aliases():
    assert resolve_model_id("phi_4_mini_instruct") == "microsoft/Phi-4-mini-instruct"
    assert resolve_model_id("phi_4_mini_reasoning") == "microsoft/Phi-4-mini-reasoning"


def test_resolve_ministral_model_aliases():
    assert resolve_model_id("ministral_3_3b") == "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
    assert resolve_model_id("ministral_3_3b_reasoning") == "unsloth/Ministral-3-3B-Reasoning-2512-GGUF"


def test_resolve_model_name_uses_alias_for_known_huggingface_id():
    assert resolve_model_name("google/gemma-2-2b-it") == "gemma_2_2b_it"
    assert resolve_model_name("custom/model") == "custom/model"


def test_resolve_custom_2025_dataset_splits(tmp_path: Path):
    assert resolve_dataset_splits("custom_2025", tmp_path) == {
        "HG": tmp_path / "raw/custom/samples_2025/custom_2025_HG.csv",
        "MG": tmp_path / "raw/custom/samples_2025/custom_2025_MG.csv",
    }


def test_resolve_semeval_subtask_c_train_dataset_split(tmp_path: Path):
    assert resolve_dataset_splits("semeval2024_task8_subtask_c_train", tmp_path) == {
        "MG": tmp_path
        / "raw/semevals_2024-task8/SemEval2024-task8/subtaskC/SemEval2024_task8_C_train_MG.csv",
    }


def test_resolve_semeval_subtask_c_dev_dataset_split(tmp_path: Path):
    assert resolve_dataset_splits("semeval2024_task8_subtask_c_dev", tmp_path) == {
        "MG": tmp_path
        / "raw/semevals_2024-task8/SemEval2024-task8/subtaskC/SemEval2024_task8_C_dev_MG.csv",
    }
