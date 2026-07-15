from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value is None:
        raise RuntimeError("TEXT_DETECTION_DATA is not set")
    return Path(value)


DATA_ROOT = _data_root()


MODEL_IDS = {
    "gemma_2_2b_it": "google/gemma-2-2b-it",
    "gemma_3_1b_it": "google/gemma-3-1b-it",
    "gpt2": "openai-community/gpt2",
    "llama_3_2_1b": "meta-llama/Llama-3.2-1B",
    "llama_3_2_3b": "meta-llama/Llama-3.2-3B",
    "opt_350m": "facebook/opt-350m",
    "phi_4_mini_instruct": "microsoft/Phi-4-mini-instruct",
    "phi_4_mini_reasoning": "microsoft/Phi-4-mini-reasoning",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

DATASETS = {
    "kaggle_train": {
        "dataset_name": "kaggle_train",
        "splits": {
            "HG": DATA_ROOT / "raw/kaggle_train/kaggle_train_hg.csv",
            "MG": DATA_ROOT / "raw/kaggle_train/kaggle_train_mg.csv",
        },
    },
    "outfox_chatgpt_train": {
        "dataset_name": "outfox_chatgpt_train",
        "splits": {
            "HG": DATA_ROOT / "raw/outfox/outfox_chatgpt_train_HG.csv",
            "MG": DATA_ROOT / "raw/outfox/outfox_chatgpt_train_MG.csv",
        },
    },
    "outfox_chatgpt_valid": {
        "dataset_name": "outfox_chatgpt_valid",
        "splits": {
            "HG": DATA_ROOT / "raw/outfox/outfox_chatgpt_valid_HG.csv",
            "MG": DATA_ROOT / "raw/outfox/outfox_chatgpt_valid_MG.csv",
        },
    },
    "outfox_chatgpt_test": {
        "dataset_name": "outfox_chatgpt_test",
        "splits": {
            "HG": DATA_ROOT / "raw/outfox/outfox_chatgpt_test_HG.csv",
            "MG": DATA_ROOT / "raw/outfox/outfox_chatgpt_test_MG.csv",
        },
    },
    "outfox_chatgpt_all": {
        "dataset_name": "outfox_chatgpt_all",
        "splits": {
            "HG": DATA_ROOT / "raw/outfox/outfox_chatgpt_all_HG.csv",
            "MG": DATA_ROOT / "raw/outfox/outfox_chatgpt_all_MG.csv",
        },
    },
}


def resolve_model_id(model: str) -> str:
    return MODEL_IDS.get(model, model)


def resolve_dataset_name(dataset: str) -> str:
    return DATASETS.get(dataset, {}).get("dataset_name", dataset)


def resolve_dataset_splits(dataset: str) -> dict[str, str]:
    return DATASETS[dataset]["splits"]
