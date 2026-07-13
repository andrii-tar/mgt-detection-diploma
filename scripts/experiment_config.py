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
    "opt_350m": "facebook/opt-350m",
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
}


def resolve_model_id(model: str) -> str:
    return MODEL_IDS.get(model, model)


def resolve_dataset_name(dataset: str) -> str:
    return DATASETS.get(dataset, {}).get("dataset_name", dataset)


def resolve_dataset_splits(dataset: str) -> dict[str, str]:
    return DATASETS[dataset]["splits"]
