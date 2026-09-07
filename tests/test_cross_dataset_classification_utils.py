from __future__ import annotations

import pandas as pd

from notebooks.cross_dataset_classification_utils import (
    DATASETS,
    SCORER_LLMS,
    TRAINING_REGIMES,
    build_deterministic_parent_cohorts,
    build_shared_parent_manifest,
    combine_whole_text_dataset_results,
    label_result_frame,
    prepare_matrix_frame,
    pivot_matrix,
    source_coverage_row,
    select_scalar_partition,
)
from notebooks.whole_text_classification_utils import WholeTextEvaluationResult


def test_matrix_configuration_has_required_models_datasets_and_regimes() -> None:
    assert SCORER_LLMS == (
        "gemma_3_1b_it",
        "gemma_3_270m_it",
        "littlelamb",
        "smollm2_360m",
    )
    assert DATASETS == {
        "Kaggle": "kaggle_train",
        "MAGE": "mage_train",
        "Outfox": "outfox_chatgpt_train",
        "BBC": "bbc_news_content",
    }
    assert TRAINING_REGIMES == {
        "Kaggle": ("Kaggle",),
        "MAGE": ("MAGE",),
        "Outfox": ("Outfox",),
        "Mixed": ("Kaggle", "MAGE", "Outfox"),
    }


def test_parent_cohorts_are_numeric_capped_and_reproducible() -> None:
    rows = [
        {"split": split, "text_id": text_id, "label": int(split == "MG")}
        for split in ("HG", "MG")
        for text_id in (9, 1, 4, 2, 8, 3, 5)
    ]
    frame = pd.DataFrame(rows)

    first = build_deterministic_parent_cohorts(frame, id_column="text_id", max_per_split=5, seed=42)
    second = build_deterministic_parent_cohorts(frame, id_column="text_id", max_per_split=5, seed=42)

    assert {name: value.to_dict("records") for name, value in first.items()} == {
        name: value.to_dict("records") for name, value in second.items()
    }
    assert {name: len(partition) for name, partition in first.items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert set(first["train"]["text_id"]).union(first["validation"]["text_id"], first["test"]["text_id"]) <= {1, 2, 3, 4, 5}
    assert set(first["train"]["split"]) == {"HG", "MG"}


def test_parent_cohorts_support_bbc_style_human_only_sources() -> None:
    frame = pd.DataFrame(
        {"split": ["HG"] * 5, "text_id": [4, 1, 3, 0, 2], "label": [0] * 5}
    )

    cohorts = build_deterministic_parent_cohorts(frame, id_column="text_id", seed=42)

    assert {name: len(partition) for name, partition in cohorts.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert all(partition["label"].eq(0).all() for partition in cohorts.values())


def test_shared_manifest_intersects_scalar_and_local_ids_before_partitioning() -> None:
    scalar = pd.DataFrame({
        "split": ["HG"] * 6 + ["MG"] * 6,
        "id": [1, 2, 3, 4, 5, 99] * 2,
        "label": [0] * 6 + [1] * 6,
    })
    local = pd.DataFrame({
        "split": ["HG"] * 6 + ["MG"] * 6,
        "text_id": [1, 2, 3, 4, 5, 98] * 2,
        "label": [0] * 6 + [1] * 6,
    })

    manifest = build_shared_parent_manifest(scalar, local, seed=42)

    selected = set().union(*(set(zip(part["split"], part["text_id"])) for part in manifest.values()))
    assert selected == {("HG", 1), ("HG", 2), ("HG", 3), ("HG", 4), ("HG", 5), ("MG", 1), ("MG", 2), ("MG", 3), ("MG", 4), ("MG", 5)}
    assert all(
        set(zip(part["split"], part["text_id"])).isdisjoint(set(zip(other["split"], other["text_id"])))
        for name, part in manifest.items()
        for other_name, other in manifest.items()
        if name != other_name
    )


def test_matrix_preparation_normalizes_classifier_and_pivots_real_block_keys() -> None:
    frame = pd.DataFrame([
        {"llm": "llm", "training_regime": "Kaggle", "evaluation_regime": "BBC", "evaluation_dataset": "bbc_news_content", "model": "logistic_regression", "partition": "test", "window_length": 32, "aggregation": "maximum_window", "macro_f1": 0.5},
    ])

    prepared = prepare_matrix_frame(frame, block="local_evidence")
    matrix = pivot_matrix(
        prepared,
        value="macro_f1",
        index=["llm", "training_regime", "classifier", "partition", "aggregation", "window_length"],
        columns=["evaluation_dataset"],
    )

    assert prepared["classifier"].tolist() == ["logistic_regression"]
    assert matrix.iloc[0, 0] == 0.5


def test_labelled_block_frames_keep_partition_and_matrix_dimensions() -> None:
    frame = pd.DataFrame([{"model": "logistic_regression", "macro_f1": 0.75}])

    labelled = label_result_frame(
        frame,
        llm="gemma_3_1b_it",
        training_regime="Kaggle",
        evaluation_regime="BBC",
        evaluation_dataset="bbc_news_content",
        block="whole_text",
    ).assign(partition="validation", feature_set="perplexity")

    assert {
        "classifier", "llm", "training_regime", "evaluation_regime",
        "evaluation_dataset", "partition", "feature_set",
    } <= set(labelled.columns)
    assert labelled.loc[0, "classifier"] == "logistic_regression"


def test_scalar_partition_selection_matches_string_and_integer_parent_ids() -> None:
    scalar = pd.DataFrame({"split": ["HG", "MG"], "id": ["001", "2"], "value": [1.0, 2.0]})
    manifest_partition = pd.DataFrame({"split": ["HG", "MG"], "text_id": [1, 2], "label": [0, 1]})

    selected = select_scalar_partition(scalar, manifest_partition)

    assert selected[["split", "id"]].to_dict("records") == [
        {"split": "HG", "id": "001"},
        {"split": "MG", "id": "2"},
    ]


def test_source_cache_failure_becomes_a_retained_coverage_row() -> None:
    row = source_coverage_row("llm", "Kaggle", FileNotFoundError("cache missing"))

    assert row == {
        "llm": "llm",
        "source_dataset": "Kaggle",
        "status": "unavailable",
        "classifier": "source_coverage",
        "error": "cache missing",
        "rows": 0,
    }


def test_matrix_preparation_supports_region_and_boundary_outputs() -> None:
    region = prepare_matrix_frame(
        pd.DataFrame([{"model": "logistic_regression", "window_length": 32, "mg_f1": 0.8}]),
        block="semeval_region",
    ).assign(llm="llm", training_regime="Kaggle", evaluation_dataset="semeval_train")
    boundary = prepare_matrix_frame(
        pd.DataFrame([{"model": "logistic_regression", "window_length": 32, "mean_absolute_token_error": 2.0}]),
        block="semeval_boundary",
    ).assign(llm="llm", training_regime="Kaggle", evaluation_dataset="semeval_train")

    region_matrix = pivot_matrix(region, value="mg_f1", index=["llm", "training_regime", "classifier", "window_length"], columns=["evaluation_dataset"])
    boundary_matrix = pivot_matrix(boundary, value="mean_absolute_token_error", index=["llm", "training_regime", "classifier", "window_length"], columns=["evaluation_dataset"])

    assert region_matrix.iloc[0, 0] == 0.8
    assert boundary_matrix.iloc[0, 0] == 2.0


def test_combine_whole_text_dataset_results_recomputes_pooled_metrics() -> None:
    def result(dataset: str, labels: list[int], scores: list[float]) -> WholeTextEvaluationResult:
        predictions = pd.DataFrame({
            "dataset": dataset,
            "label": labels,
            "mg_score": scores,
            "prediction": [int(score >= 0.5) for score in scores],
            "classifier": "logistic_regression",
            "feature_set": "all",
            "threshold": 0.5,
            "direction": "higher_is_mg",
        })
        return WholeTextEvaluationResult(
            metrics=pd.DataFrame([{"dataset": dataset, "macro_f1": 1.0}]),
            predictions=predictions,
            coverage=pd.DataFrame([{"dataset": dataset, "rows": len(labels)}]),
        )

    combined = combine_whole_text_dataset_results(
        [result("kaggle_train", [0, 1], [0.1, 0.9]), result("outfox_chatgpt_train", [0, 1], [0.9, 0.1])],
        evaluation_regime="Mixed",
    )

    assert len(combined.metrics) == 1
    assert combined.metrics.loc[0, "evaluation_dataset"] == "Mixed"
    assert combined.metrics.loc[0, "macro_f1"] == 0.5
    assert len(combined.predictions) == 4
    assert set(combined.predictions["dataset"]) == {"kaggle_train", "outfox_chatgpt_train"}
    assert set(combined.predictions["evaluation_dataset"]) == {"Mixed"}
    assert set(combined.coverage["dataset"]) == {"kaggle_train", "outfox_chatgpt_train"}
    assert set(combined.coverage["evaluation_dataset"]) == {"Mixed"}
