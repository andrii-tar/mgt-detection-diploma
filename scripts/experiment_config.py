from __future__ import annotations


MODEL_IDS = {
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

DATASETS = {
    "kaggle_train": {
        "dataset_name": "kaggle_train",
        "splits": {
            "HG": "data/raw/kaggle_train_hg.csv",
            "MG": "data/raw/kaggle_train_mg.csv",
        },
    },
}


def resolve_model_id(model: str) -> str:
    return MODEL_IDS.get(model, model)


def resolve_dataset_name(dataset: str) -> str:
    return DATASETS.get(dataset, {}).get("dataset_name", dataset)


def resolve_dataset_splits(dataset: str) -> dict[str, str]:
    return DATASETS[dataset]["splits"]
