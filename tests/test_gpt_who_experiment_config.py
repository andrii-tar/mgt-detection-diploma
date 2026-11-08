import pytest

from notebooks.utils.gpt_who_experiment_config import (
    DATASET_GROUPS,
    DEFAULT_EXPERIMENTS,
    SCORER_LLMS,
    experiment_dataset_names,
    experiment_run_id,
    validate_experiments,
)


def test_valid_experiment_expands_dataset_names_in_order():
    experiment = validate_experiments(
        (
            {
                "name": "kaggle_to_outfox",
                "scorer_llms": ("smollm2_360m",),
                "training_datasets": ("kaggle_train",),
                "evaluation_datasets": ("outfox_chatgpt_train",),
            },
        )
    )[0]

    assert experiment_dataset_names(experiment, "training_datasets") == ("kaggle_train",)
    assert experiment_dataset_names(experiment, "evaluation_datasets") == ("outfox_chatgpt_train",)


def test_default_experiments_are_valid():
    assert validate_experiments(DEFAULT_EXPERIMENTS)
    assert set(DATASET_GROUPS) >= {"kaggle_train", "mage_train", "outfox_chatgpt_train"}


def test_default_experiments_run_all_configured_scorer_llms():
    for experiment in validate_experiments(DEFAULT_EXPERIMENTS):
        assert experiment["scorer_llms"] == SCORER_LLMS


def test_default_experiments_include_mixed_bbc_training():
    experiments = {experiment["name"]: experiment for experiment in validate_experiments(DEFAULT_EXPERIMENTS)}

    mixed_bbc = experiments["smollm_mixed_bbc"]
    assert experiment_dataset_names(mixed_bbc, "training_datasets") == (
        "kaggle_train",
        "mage_train",
        "outfox_chatgpt_train",
        "bbc_news_content",
    )
    assert experiment_dataset_names(mixed_bbc, "evaluation_datasets") == (
        "kaggle_train",
        "mage_train",
        "outfox_chatgpt_train",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "unsafe/name"),
        ("scorer_llms", ("unknown_model",)),
        ("training_datasets", ("unknown_dataset",)),
        ("evaluation_datasets", ()),
    ],
)
def test_invalid_experiment_configuration_is_rejected(field, value):
    experiment = {
        "name": "valid",
        "scorer_llms": ("smollm2_360m",),
        "training_datasets": ("kaggle_train",),
        "evaluation_datasets": ("mage_train",),
    }
    experiment[field] = value

    with pytest.raises(ValueError):
        validate_experiments((experiment,))


def test_experiment_run_id_is_deterministic_and_path_safe():
    assert experiment_run_id("gpt-who-baseline", "kaggle_to_outfox") == (
        "gpt-who-baseline__kaggle_to_outfox"
    )
