import pandas as pd

from notebooks.utils.similarity_ablation_utils import (
    FEATURE_SET_COLUMNS,
    PROBABILISTIC_FEATURE_COLUMNS,
    SIMILARITY_FEATURE_COLUMNS,
    intersect_eligible_texts_by_stratum,
    paired_macro_f1_bootstrap,
    sample_paired_text_cohorts,
)


def test_feature_sets_differ_only_by_the_two_similarity_aggregates():
    assert len(PROBABILISTIC_FEATURE_COLUMNS) == 11
    assert SIMILARITY_FEATURE_COLUMNS == (
        "mean_candidate_similarity__mean",
        "mean_candidate_similarity__std",
    )
    assert FEATURE_SET_COLUMNS["P+S"] == (
        *FEATURE_SET_COLUMNS["P"],
        *SIMILARITY_FEATURE_COLUMNS,
    )


def test_sampling_selects_disjoint_equal_sized_cohorts_per_source_and_class():
    parents = pd.DataFrame(
        [
            {"dataset": dataset, "split": split, "text_id": f"{dataset}-{split}-{index}", "label": label}
            for dataset in ("kaggle_train", "mage_train")
            for split, label in (("HG", 0), ("MG", 1))
            for index in range(5)
        ]
    )

    selected = sample_paired_text_cohorts(
        parents, train_texts_per_stratum=2, evaluation_texts_per_stratum=2, seed=19
    )

    assert selected.groupby(["partition", "dataset", "split"]).size().eq(2).all()
    train = selected.loc[selected["partition"].eq("train"), "text_id"]
    evaluation = selected.loc[selected["partition"].eq("evaluation"), "text_id"]
    assert set(train).isdisjoint(evaluation)
    assert selected.equals(
        sample_paired_text_cohorts(
            parents, train_texts_per_stratum=2, evaluation_texts_per_stratum=2, seed=19
        )
    )


def test_sampling_fails_when_a_stratum_cannot_supply_both_cohorts():
    parents = pd.DataFrame(
        [{"dataset": "kaggle_train", "split": "HG", "text_id": "only-one", "label": 0}]
    )

    try:
        sample_paired_text_cohorts(
            parents, train_texts_per_stratum=1, evaluation_texts_per_stratum=1, seed=1
        )
    except ValueError as error:
        assert "kaggle_train/HG" in str(error)
    else:
        raise AssertionError("expected insufficient cohort candidates to be rejected")


def test_eligible_text_intersection_is_per_dataset_and_split_then_combined():
    partitions = {
        ("kaggle_train", "HG"): [
            pd.DataFrame(
                [
                    {"dataset": "kaggle_train", "split": "HG", "text_id": text_id, "label": 0}
                    for text_id in (1, 2)
                ]
            ),
            pd.DataFrame(
                [
                    {"dataset": "kaggle_train", "split": "HG", "text_id": text_id, "label": 0}
                    for text_id in (2, 3)
                ]
            ),
        ],
        ("mage_train", "MG"): [
            pd.DataFrame(
                [
                    {"dataset": "mage_train", "split": "MG", "text_id": text_id, "label": 1}
                    for text_id in (4, 5)
                ]
            ),
            pd.DataFrame(
                [
                    {"dataset": "mage_train", "split": "MG", "text_id": text_id, "label": 1}
                    for text_id in (5, 6)
                ]
            ),
        ],
    }

    eligible = intersect_eligible_texts_by_stratum(partitions)

    assert set(map(tuple, eligible[["dataset", "split", "text_id"]].itertuples(index=False, name=None))) == {
        ("kaggle_train", "HG", 2),
        ("mage_train", "MG", 5),
    }


def test_paired_bootstrap_reports_document_clustered_delta_and_is_reproducible():
    rows = []
    for dataset in ("kaggle_train", "mage_train"):
        for split, label in (("HG", 0), ("MG", 1)):
            for text_index in range(2):
                text_id = f"{dataset}-{split}-{text_index}"
                for window_index in range(2):
                    for feature_set, prediction in (("P", 1), ("P+S", label)):
                        rows.append(
                            {
                                "scorer_llm": "gemma_3_270m_it",
                                "model_family": "logistic_regression",
                                "window": 32,
                                "feature_set": feature_set,
                                "dataset": dataset,
                                "split": split,
                                "text_id": text_id,
                                "label": label,
                                "prediction": prediction,
                                "window_id": window_index,
                            }
                        )
    predictions = pd.DataFrame(rows)

    first = paired_macro_f1_bootstrap(predictions, n_bootstrap=50, seed=7)
    second = paired_macro_f1_bootstrap(predictions, n_bootstrap=50, seed=7)

    assert len(first) == 1
    result = first.iloc[0]
    assert result["macro_f1_P_plus_S"] == 1.0
    assert result["delta_macro_f1"] > 0
    assert result["ci_lower"] == second.iloc[0]["ci_lower"]
    assert result["ci_upper"] == second.iloc[0]["ci_upper"]
