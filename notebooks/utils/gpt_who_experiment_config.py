from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

DATASET_GROUPS = {
    "kaggle_train": "kaggle_train",
    "mage_train": "mage_train",
    "outfox_chatgpt_train": "outfox_chatgpt_train",
    "bbc_news_content": "bbc_news_content",
}

SOURCE_SPLITS = {
    "kaggle_train": ("HG", "MG"),
    "mage_train": ("HG", "MG"),
    "outfox_chatgpt_train": ("HG", "MG"),
    "bbc_news_content": ("HG",),
}

SCORER_LLMS = (
    "gemma_3_1b_it",
    "gemma_3_270m_it",
    "littlelamb",
    "smollm2_360m",
)

DEFAULT_EXPERIMENTS = (
    {
        "name": "smollm_mixed",
        "scorer_llms": SCORER_LLMS,
        "training_datasets": (
            "kaggle_train",
            "mage_train",
            "outfox_chatgpt_train",
        ),
        "evaluation_datasets": (
            "kaggle_train",
            "mage_train",
            "outfox_chatgpt_train",
        ),
    },
    {
        "name": "smollm_mixed_bbc",
        "scorer_llms": SCORER_LLMS,
        "training_datasets": (
            "kaggle_train",
            "mage_train",
            "outfox_chatgpt_train",
            "bbc_news_content",
        ),
        "evaluation_datasets": (
            "kaggle_train",
            "mage_train",
            "outfox_chatgpt_train",
        ),
    },
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DATASET_FIELDS = ("training_datasets", "evaluation_datasets")


def _validate_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a non-empty path-safe name")
    return value


def validate_experiments(experiments: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    validated = []
    for raw_experiment in experiments:
        if not isinstance(raw_experiment, Mapping):
            raise ValueError("each experiment must be a mapping")
        experiment = dict(raw_experiment)
        _validate_name(experiment.get("name"), field="experiment name")

        scorer_llms = tuple(experiment.get("scorer_llms", ()))
        if not scorer_llms or any(llm not in SCORER_LLMS for llm in scorer_llms):
            raise ValueError("scorer_llms must contain only configured scorer LLMs")

        for field in _DATASET_FIELDS:
            datasets = tuple(experiment.get(field, ()))
            if not datasets or any(dataset not in DATASET_GROUPS for dataset in datasets):
                raise ValueError(f"{field} must contain only configured dataset names")
            experiment[field] = datasets
        experiment["scorer_llms"] = scorer_llms
        validated.append(experiment)
    return tuple(validated)


def experiment_dataset_names(experiment: Mapping[str, object], field: str) -> tuple[str, ...]:
    if field not in _DATASET_FIELDS:
        raise ValueError(f"field must be one of {_DATASET_FIELDS}")
    return tuple(DATASET_GROUPS[name] for name in experiment[field])


def experiment_run_id(base_run_id: str, experiment_name: str) -> str:
    _validate_name(base_run_id, field="base run ID")
    _validate_name(experiment_name, field="experiment name")
    return f"{base_run_id}__{experiment_name}"
