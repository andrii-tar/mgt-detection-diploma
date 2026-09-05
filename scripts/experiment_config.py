from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(".")


MODEL_IDS = {
    "gemma_2_2b_it": "google/gemma-2-2b-it",
    "gemma_3_1b_it": "google/gemma-3-1b-it",
    "gpt2": "openai-community/gpt2",
    "llama_3_2_1b": "meta-llama/Llama-3.2-1B",
    "llama_3_2_3b": "meta-llama/Llama-3.2-3B",
    "opt_350m": "facebook/opt-350m",
    "phi_4_mini_instruct": "microsoft/Phi-4-mini-instruct",
    "phi_4_mini_reasoning": "microsoft/Phi-4-mini-reasoning",
    "ministral_3_3b": "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
    "ministral_3_3b_reasoning": "unsloth/Ministral-3-3B-Reasoning-2512-GGUF",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

def build_datasets(data_root: Path) -> dict[str, dict[str, object]]:
    return {
        "kaggle_train": {
        "dataset_name": "kaggle_train",
        "splits": {
            "HG": data_root / "raw/kaggle_train/kaggle_train_hg.csv",
            "MG": data_root / "raw/kaggle_train/kaggle_train_mg.csv",
        },
    },
    "mage_train": {
        "dataset_name": "mage_train",
        "splits": {
            "HG": data_root / "raw/MAGE/train/MAGE_train_HG.csv",
            "MG": data_root / "raw/MAGE/train/MAGE_train_MG.csv",
        },
    },
    "outfox_chatgpt_train": {
        "dataset_name": "outfox_chatgpt_train",
        "splits": {
            "HG": data_root / "raw/outfox/outfox_chatgpt_train_HG.csv",
            "MG": data_root / "raw/outfox/outfox_chatgpt_train_MG.csv",
        },
    },
    "outfox_chatgpt_valid": {
        "dataset_name": "outfox_chatgpt_valid",
        "splits": {
            "HG": data_root / "raw/outfox/outfox_chatgpt_valid_HG.csv",
            "MG": data_root / "raw/outfox/outfox_chatgpt_valid_MG.csv",
        },
    },
    "outfox_chatgpt_test": {
        "dataset_name": "outfox_chatgpt_test",
        "splits": {
            "HG": data_root / "raw/outfox/outfox_chatgpt_test_HG.csv",
            "MG": data_root / "raw/outfox/outfox_chatgpt_test_MG.csv",
        },
    },
    "outfox_chatgpt_all": {
        "dataset_name": "outfox_chatgpt_all",
        "splits": {
            "HG": data_root / "raw/outfox/outfox_chatgpt_all_HG.csv",
            "MG": data_root / "raw/outfox/outfox_chatgpt_all_MG.csv",
        },
    },
    "semeval2024_task8_subtask_c_train": {
        "dataset_name": "semeval2024_task8_subtask_c_train",
        "splits": {
            "MG": data_root
            / "raw/semevals_2024-task8/SemEval2024-task8/subtaskC/SemEval2024_task8_C_train_MG.csv",
        },
    },
    "semeval2024_task8_subtask_c_dev": {
        "dataset_name": "semeval2024_task8_subtask_c_dev",
        "splits": {
            "MG": data_root
            / "raw/semevals_2024-task8/SemEval2024-task8/subtaskC/SemEval2024_task8_C_dev_MG.csv",
        },
    },
    "custom_2025": {
        "dataset_name": "custom_2025",
        "splits": {
            "HG": data_root / "raw/custom/samples_2025/custom_2025_HG.csv",
            "MG": data_root / "raw/custom/samples_2025/custom_2025_MG.csv",
        },
    },
    "bbc_news_content": {
        "dataset_name": "bbc_news_content",
        "splits": {
            "HG": data_root / "raw/bbc_news/bbc_news_content_HG.csv",
        },
    },
    "bbc_news_titles": {
        "dataset_name": "bbc_news_titles",
        "splits": {
            "HG": data_root / "raw/bbc_news/bbc_news_titles_HG.csv",
        },
    },
    }


DATASETS = build_datasets(DATA_ROOT)


def resolve_model_id(model: str) -> str:
    return MODEL_IDS.get(model, model)


def resolve_model_name(model: str) -> str:
    if model in MODEL_IDS:
        return model
    for alias, model_id in MODEL_IDS.items():
        if model_id == model:
            return alias
    return model


def resolve_dataset_name(dataset: str) -> str:
    return DATASETS.get(dataset, {}).get("dataset_name", dataset)


def resolve_dataset_splits(dataset: str, data_root: Path = DATA_ROOT) -> dict[str, Path]:
    return build_datasets(data_root)[dataset]["splits"]
