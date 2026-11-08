import numpy as np
import pandas as pd
import pytest

from notebooks.utils.gpt_who_baseline_utils import (
    GPT_WHO_FEATURE_COLUMNS,
    build_gpt_who_feature_frame,
    compute_gpt_who_features,
    fit_and_score_logistic,
    make_experiment_cohorts,
    make_mixed_parent_cohorts,
    make_parent_cohorts,
    summarize_scores,
)


def test_exact_global_features_and_column_order():
    row = {"tokens": [{"logprob": -1.0}, {"logprob": -2.0}] * 10}

    result = compute_gpt_who_features(row, window_size=20)

    assert tuple(result) == GPT_WHO_FEATURE_COLUMNS
    assert result["uid_mean_surprisal"] == pytest.approx(1.5)
    assert result["uid_surprisal_variance"] == pytest.approx(0.25)
    assert result["uid_mean_abs_surprisal_delta"] == pytest.approx(1.0)
    assert result["uid_mean_squared_surprisal_delta"] == pytest.approx(1.0)
    assert result["uid_min_window_surprisal_variance"] == pytest.approx(0.25)
    assert result["uid_max_window_surprisal_variance"] == pytest.approx(0.25)
    assert result["uid_min_window_abs_delta"] == pytest.approx(1.0)
    assert result["uid_max_window_abs_delta"] == pytest.approx(1.0)
    assert result["uid_min_window_squared_delta"] == pytest.approx(1.0)
    assert result["uid_max_window_squared_delta"] == pytest.approx(1.0)
    assert result["uid_valid_token_count"] == 20


def test_exact_window_size_is_accepted_and_longer_text_has_multiple_windows():
    tokens = [{"logprob": -float(index % 3 + 1)} for index in range(21)]

    result = compute_gpt_who_features({"tokens": tokens}, window_size=20)

    assert result["uid_valid_token_count"] == 21
    assert np.isfinite(result["uid_min_window_surprisal_variance"])
    assert result["uid_min_window_surprisal_variance"] <= result["uid_max_window_surprisal_variance"]


def test_numpy_token_array_from_parquet_is_supported():
    tokens = np.asarray([{"logprob": -1.0}] * 20, dtype=object)

    result = compute_gpt_who_features({"tokens": tokens}, window_size=20)

    assert result["uid_valid_token_count"] == 20


@pytest.mark.parametrize(
    ("tokens", "error_match"),
    [
        ([{"logprob": -1.0}] * 19, "fewer than 20"),
        ([{"logprob": None}] * 20, "no valid"),
        ([{"logprob": float("nan")}] * 20, "no valid"),
    ],
)
def test_short_or_invalid_rows_raise(tokens, error_match):
    with pytest.raises(ValueError, match=error_match):
        compute_gpt_who_features({"tokens": tokens}, window_size=20)


def test_frame_builder_preserves_identity_and_reports_exclusions():
    rows = [
        {
            "id": 7,
            "text": "accepted",
            "dataset": "kaggle_train",
            "model": "smollm2_360m",
            "split": "HG",
            "tokens": [{"logprob": -1.0}] * 20,
        },
        {
            "id": 8,
            "text": "too short",
            "dataset": "kaggle_train",
            "model": "smollm2_360m",
            "split": "MG",
            "tokens": [{"logprob": -1.0}] * 3,
        },
    ]

    features, exclusions = build_gpt_who_feature_frame(rows, window_size=20)

    assert features["id"].tolist() == [7]
    assert features.loc[0, "split"] == "HG"
    assert set(GPT_WHO_FEATURE_COLUMNS).issubset(features.columns)
    assert exclusions[["id", "reason"]].to_dict("records") == [{"id": 8, "reason": "short_text"}]
    assert isinstance(features, pd.DataFrame)


def _feature_row(value: float) -> dict[str, float]:
    return {column: value for column in GPT_WHO_FEATURE_COLUMNS}


def test_parent_groups_never_cross_cohorts():
    frame = pd.DataFrame(
        [
            {"id": 1, "label": 0, **_feature_row(0.0)},
            {"id": 1, "label": 0, **_feature_row(0.1)},
            {"id": 2, "label": 1, **_feature_row(1.0)},
            {"id": 3, "label": 1, **_feature_row(1.1)},
        ]
    )

    cohorts = make_parent_cohorts(frame, seed=42, validation_fraction=0.5)

    assert set(cohorts["train"]["id"]).isdisjoint(cohorts["validation"]["id"])
    assert set(cohorts) == {"train", "validation"}


def test_parent_cohorts_are_deterministic_and_retain_labels_when_possible():
    frame = pd.DataFrame(
        [
            {"id": index, "label": label, **_feature_row(float(label))}
            for label in (0, 1)
            for index in range(label * 3, label * 3 + 3)
        ]
    )

    first = make_parent_cohorts(frame, seed=7, validation_fraction=0.34)
    second = make_parent_cohorts(frame, seed=7, validation_fraction=0.34)

    pd.testing.assert_frame_equal(first["train"], second["train"])
    pd.testing.assert_frame_equal(first["validation"], second["validation"])
    assert set(first["train"]["label"]) == {0, 1}
    assert set(first["validation"]["label"]) == {0, 1}


def test_logistic_scoring_returns_scores_coefficients_and_metrics_without_mutating_evaluation():
    rows = []
    for index in range(12):
        label = index % 2
        rows.append(
            {
                "id": index,
                "label": label,
                **{column: float(label) for column in GPT_WHO_FEATURE_COLUMNS},
            }
        )
    frame = pd.DataFrame(rows)
    train = frame.iloc[:8].copy()
    evaluation = frame.iloc[8:].copy()
    evaluation_before = evaluation.copy(deep=True)

    scores, coefficients = fit_and_score_logistic(train, evaluation)
    metrics = summarize_scores(scores, partition="validation")

    assert set(("id", "label", "predicted_label", "mg_probability")).issubset(scores.columns)
    assert coefficients["feature"].tolist() == list(GPT_WHO_FEATURE_COLUMNS)
    assert metrics.loc[0, "partition"] == "validation"
    assert metrics.loc[0, "sample_count"] == 4
    assert metrics.loc[0, "roc_auc"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(evaluation, evaluation_before)


def test_missing_group_column_is_rejected():
    with pytest.raises(ValueError, match="group_column"):
        make_parent_cohorts(pd.DataFrame({"label": [0, 1]}))


def test_mixed_parent_cohorts_match_balanced_method_counts_and_do_not_overlap():
    rows = []
    for dataset_index, dataset in enumerate(("kaggle_train", "mage_train", "outfox_chatgpt_train")):
        for label, split in ((0, "HG"), (1, "MG")):
            for text_id in range(600):
                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "id": dataset_index * 10_000 + label * 1_000 + text_id,
                    "label": label,
                    **_feature_row(float(label)),
                })

    cohorts = make_mixed_parent_cohorts(
        pd.DataFrame(rows),
        seed=42,
        max_per_split=500,
        train_count_per_class=900,
        eval_count_per_class=300,
    )

    assert cohorts["train"]["label"].value_counts().to_dict() == {0: 900, 1: 900}
    assert cohorts["validation"]["label"].value_counts().to_dict() == {0: 300, 1: 300}
    train_ids = set(zip(cohorts["train"]["dataset"], cohorts["train"]["split"], cohorts["train"]["id"]))
    validation_ids = set(zip(cohorts["validation"]["dataset"], cohorts["validation"]["split"], cohorts["validation"]["id"]))
    assert train_ids.isdisjoint(validation_ids)
    assert cohorts["train"].groupby(["dataset", "split"]).size().eq(300).all()
    assert cohorts["validation"].groupby(["dataset", "split"]).size().eq(100).all()
    assert cohorts["train"]["id"].mod(1_000).max() < 500
    assert cohorts["validation"]["id"].mod(1_000).max() < 500


def test_experiment_cohorts_use_configured_datasets_without_overlap():
    rows = []
    for dataset in ("train_dataset", "eval_dataset"):
        for label, split in ((0, "HG"), (1, "MG")):
            for text_id in range(20):
                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "id": text_id,
                    "label": label,
                    **_feature_row(float(label)),
                })

    cohorts = make_experiment_cohorts(
        pd.DataFrame(rows),
        training_datasets=("train_dataset", "eval_dataset"),
        evaluation_datasets=("eval_dataset",),
        seed=42,
        max_per_split=20,
        train_count_per_class=4,
        eval_count_per_class=2,
    )

    assert cohorts["train"]["label"].value_counts().to_dict() == {0: 4, 1: 4}
    assert cohorts["evaluation"]["label"].value_counts().to_dict() == {0: 2, 1: 2}
    assert set(cohorts["evaluation"]["dataset"]) == {"eval_dataset"}
    train_ids = set(zip(cohorts["train"]["dataset"], cohorts["train"]["split"], cohorts["train"]["id"]))
    evaluation_ids = set(zip(cohorts["evaluation"]["dataset"], cohorts["evaluation"]["split"], cohorts["evaluation"]["id"]))
    assert train_ids.isdisjoint(evaluation_ids)


def test_experiment_cohorts_pool_hg_only_bbc_with_binary_training_sources():
    rows = []
    for dataset in ("kaggle_train", "mage_train", "outfox_chatgpt_train"):
        for label, split in ((0, "HG"), (1, "MG")):
            for text_id in range(12):
                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "id": text_id,
                    "label": label,
                    **_feature_row(float(label)),
                })
    for text_id in range(12):
        rows.append({
            "dataset": "bbc_news_content",
            "split": "HG",
            "id": text_id,
            "label": 0,
            **_feature_row(0.0),
        })

    cohorts = make_experiment_cohorts(
        pd.DataFrame(rows),
        training_datasets=("kaggle_train", "mage_train", "outfox_chatgpt_train", "bbc_news_content"),
        evaluation_datasets=("kaggle_train", "mage_train", "outfox_chatgpt_train"),
        seed=42,
        max_per_split=5,
        train_count_per_class=8,
        eval_count_per_class=3,
    )

    assert cohorts["train"]["label"].value_counts().to_dict() == {0: 8, 1: 8}
    assert "bbc_news_content" in set(cohorts["train"]["dataset"])
