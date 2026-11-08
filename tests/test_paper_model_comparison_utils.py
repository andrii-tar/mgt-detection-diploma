import json

import numpy as np
import pandas as pd
import pytest

from notebooks.utils.paper_model_comparison_utils import (
    binary_window_metrics,
    build_cohort_manifest,
    build_configuration_coverage,
    filter_window_rows_to_common_documents,
    label_semeval_windows,
    load_evaluation_window_rows,
    load_processed_token_counts,
    load_semeval_token_labels,
    load_document_word_counts,
    normalize_t08_aggregation_labels,
    single_class_rates,
)


def test_semeval_boundary_windows_are_labeled_or_excluded_by_gold_boundary():
    windows = pd.DataFrame({
        "text_id": [7, 7, 7],
        "start_token": [0, 2, 3],
        "stop_token": [3, 4, 5],
        "mg_score": [0.1, 0.8, 0.9],
        "prediction": [0, 1, 1],
    })
    annotations = pd.DataFrame([{"id": 7, "text": "a b c d", "boundary_word_index": 2}])
    token_labels = {7: ["a", " ", "b", " ", "c", " ", "d"]}

    labeled, excluded = label_semeval_windows(windows, annotations, token_labels)

    assert labeled[["start_token", "gold_label"]].to_records(index=False).tolist() == [(0, 0), (3, 1)]
    assert excluded.loc[0, "excluded_boundary_window_count"] == 1
    assert excluded.loc[0, "gold_token_boundary"] == 3


def test_unmappable_semeval_annotations_are_reported_and_their_windows_excluded():
    windows = pd.DataFrame({
        "text_id": [8], "start_token": [0], "stop_token": [2],
        "mg_score": [0.4], "prediction": [0],
    })
    annotations = pd.DataFrame([{"id": 8, "text": "a b", "boundary_word_index": 7}])

    labeled, excluded = label_semeval_windows(windows, annotations, {8: ["a", " ", "b"]})

    assert labeled.empty
    assert excluded.loc[0, "excluded_boundary_window_count"] == 1
    assert excluded.loc[0, "exclusion_reason"] == "boundary_word_index is outside the text word range"


def test_binary_window_metrics_returns_macro_f1_and_counts_by_configuration():
    windows = pd.DataFrame({
        "scorer_llm": ["gemma"] * 4,
        "model_family": ["lr"] * 4,
        "window_length": [32] * 4,
        "label": [0, 0, 1, 1],
        "prediction": [0, 1, 1, 1],
    })

    result = binary_window_metrics(windows, group_columns=("scorer_llm", "model_family", "window_length"))

    assert result.loc[0, "window_count"] == 4
    assert result.loc[0, "hg_window_count"] == 2
    assert result.loc[0, "mg_window_count"] == 2
    assert result.loc[0, "macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.loc[0, "tn"] == 1
    assert result.loc[0, "fp"] == 1
    assert result.loc[0, "fn"] == 0
    assert result.loc[0, "tp"] == 2
    assert result.loc[0, "false_positive_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize("label,expected_rate_name,expected_rate", [
    (1, "mg_detection_rate", 0.5),
    (0, "false_positive_rate", 0.5),
])
def test_single_class_metrics_report_only_the_relevant_operating_rate(label, expected_rate_name, expected_rate):
    rows = pd.DataFrame({
        "dataset": ["target"] * 4,
        "label": [label] * 4,
        "prediction": [0, 1, 1, 0],
    })

    result = single_class_rates(rows, group_columns=("dataset",))

    assert result.loc[0, "sample_count"] == 4
    assert result.loc[0, expected_rate_name] == pytest.approx(expected_rate)
    assert "macro_f1" not in result


def test_single_class_metrics_reject_mixed_labels():
    rows = pd.DataFrame({"label": [0, 1], "prediction": [0, 1]})
    with pytest.raises(ValueError, match="single-class"):
        single_class_rates(rows, group_columns=())


def test_window_rows_are_filtered_to_common_documents_and_coverage_is_reported():
    rows = pd.DataFrame([
        {"source_dataset": "s", "split": "HG", "text_id": 1, "start_token": 0, "scorer_llm": "gemma",
         "evaluation_regime": "Mixed", "model_family": "lr", "training_window": 32, "evaluation_window": 32},
        {"source_dataset": "s", "split": "HG", "text_id": 2, "start_token": 0, "scorer_llm": "gemma",
         "evaluation_regime": "Mixed", "model_family": "lr", "training_window": 32, "evaluation_window": 32},
        {"source_dataset": "s", "split": "HG", "text_id": 2, "start_token": 0, "scorer_llm": "gemma",
         "evaluation_regime": "Mixed", "model_family": "rf", "training_window": 64, "evaluation_window": 64},
        {"source_dataset": "s", "split": "HG", "text_id": 3, "start_token": 0, "scorer_llm": "gemma",
         "evaluation_regime": "Mixed", "model_family": "rf", "training_window": 64, "evaluation_window": 64},
    ])

    filtered, coverage = filter_window_rows_to_common_documents(
        rows,
        comparison_group_columns=("scorer_llm", "evaluation_regime"),
        configuration_columns=("model_family", "training_window", "evaluation_window"),
    )

    assert filtered.text_id.tolist() == [2, 2]
    assert coverage.common_document_count.tolist() == [1, 1]
    assert coverage.excluded_document_count.tolist() == [1, 1]


def test_cohort_manifest_records_document_inclusion_reason_and_stable_hash():
    available = pd.DataFrame([
        {"scorer_llm": "gemma", "window_length": 32, "source_dataset": "source", "split": "HG", "text_id": 1, "label": 0},
        {"scorer_llm": "gemma", "window_length": 32, "source_dataset": "source", "split": "MG", "text_id": 2, "label": 1},
    ])
    included = available.loc[available.text_id.eq(1)]

    first = build_cohort_manifest(
        available, included, configuration_columns=("scorer_llm", "window_length"),
        exclusion_reason="not_in_common_test_cohort",
    )
    second = build_cohort_manifest(
        available, included, configuration_columns=("scorer_llm", "window_length"),
        exclusion_reason="not_in_common_test_cohort",
    )

    assert first.cohort_id.nunique() == 1
    assert first.cohort_id.iloc[0] == second.cohort_id.iloc[0]
    assert first.set_index("text_id").inclusion_status.to_dict() == {1: "included", 2: "excluded"}
    assert first.set_index("text_id").loc[2, "exclusion_reason"] == "not_in_common_test_cohort"


def test_configuration_coverage_lists_expected_missing_cells_with_reason():
    expected = pd.DataFrame({"scorer_llm": ["gemma", "little"], "window": [32, 32]})
    available = pd.DataFrame({"scorer_llm": ["gemma"], "window": [32]})

    result = build_configuration_coverage(
        expected, available, block="E_name_that_llm", key_columns=("scorer_llm", "window"),
        missing_reason="prepared MG-only feature windows unavailable",
    )

    assert result.status.tolist() == ["available", "unavailable"]
    assert result.loc[1, "reason"] == "prepared MG-only feature windows unavailable"


def test_evaluation_loader_selects_noncanonical_feature_and_training_regime(tmp_path):
    root = tmp_path / "predictions"
    artifact = root / "artifact"
    artifact.mkdir(parents=True)
    run = "inventory-run"
    artifact_id = "random_forest__littlelamb__extended__Kaggle__64__inventory-run"
    (artifact / "metadata.json").write_text(json.dumps({
        "artifact_id": artifact_id, "partition": "test",
        "evaluation_regime": "Mixed", "evaluation_window": "32",
    }), encoding="utf-8")
    pd.DataFrame([{
        "source_dataset": "source_a", "split": "MG", "text_id": 12, "label": 1,
        "start_token": 0, "mg_score": 0.7, "prediction": 1,
        "model_family": "random_forest", "scorer_llm": "littlelamb",
        "feature_set": "extended", "training_regime": "Kaggle", "training_window": "64",
        "evaluation_regime": "Mixed", "evaluation_window": "32", "partition": "test",
    }]).to_parquet(artifact / "data.parquet", index=False)

    result = load_evaluation_window_rows(
        root, run_id=run, scorer_llm="littlelamb", model_family="random_forest",
        feature_set="extended", training_regime="Kaggle", training_window=64,
        evaluation_regime="Mixed", evaluation_window=32,
    )

    assert result.text_id.tolist() == [12]
    assert result.training_window.tolist() == [64]
    assert result.evaluation_window.tolist() == [32]


def test_evaluation_loader_keeps_whole_text_representation_label(tmp_path):
    root = tmp_path / "predictions"
    artifact = root / "artifact"
    artifact.mkdir(parents=True)
    run = "whole-run"
    (artifact / "metadata.json").write_text(json.dumps({
        "artifact_id": "logistic_regression__gemma_3_270m_it__simple_similarity__Kaggle_MAGE_Outfox__whole_text__whole-run",
        "partition": "test", "evaluation_regime": "Mixed", "evaluation_window": "whole_text",
    }), encoding="utf-8")
    pd.DataFrame([{
        "source_dataset": "source_a", "split": "MG", "text_id": 3, "label": 1,
        "start_token": 0, "mg_score": 0.8, "prediction": 1,
        "model_family": "logistic_regression", "scorer_llm": "gemma_3_270m_it",
        "feature_set": "simple_similarity", "training_regime": "Kaggle+MAGE+Outfox",
        "training_window": "whole_text", "evaluation_regime": "Mixed",
        "evaluation_window": "whole_text", "partition": "test",
    }]).to_parquet(artifact / "data.parquet", index=False)

    result = load_evaluation_window_rows(
        root, run_id=run, scorer_llm="gemma_3_270m_it", model_family="logistic_regression",
        training_window="whole_text", evaluation_regime="Mixed", evaluation_window="whole_text",
    )

    assert result.training_window.tolist() == ["whole_text"]
    assert result.evaluation_window.tolist() == ["whole_text"]


def test_evaluation_loader_prefers_canonical_source_aware_artifact_over_dataset_children(tmp_path):
    root = tmp_path / "predictions"
    artifact = root / "artifact"
    canonical = artifact / "Mixed" / "32" / "Mixed" / "test"
    child = artifact / "Mixed" / "32" / "kaggle_train" / "test"
    canonical.mkdir(parents=True)
    child.mkdir(parents=True)
    run = "source-aware-run"
    artifact_id = "logistic_regression__gemma_3_270m_it__simple_similarity__Kaggle_MAGE_Outfox__32__source-aware-run"
    metadata = {
        "artifact_id": artifact_id, "partition": "test",
        "evaluation_regime": "Mixed", "evaluation_window": "32",
    }
    for path in (canonical, child):
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    common = {
        "split": "HG", "text_id": 1, "label": 0, "start_token": 0, "mg_score": 0.1,
        "prediction": 0, "model_family": "logistic_regression", "scorer_llm": "gemma_3_270m_it",
        "feature_set": "simple_similarity", "training_regime": "Kaggle+MAGE+Outfox",
        "training_window": "32", "evaluation_regime": "Mixed", "evaluation_window": "32", "partition": "test",
    }
    pd.DataFrame([{**common, "source_dataset": "kaggle_train"}]).to_parquet(canonical / "data.parquet", index=False)
    pd.DataFrame([{**common, "dataset": "kaggle_train"}]).to_parquet(child / "data.parquet", index=False)

    result = load_evaluation_window_rows(
        root, run_id=run, scorer_llm="gemma_3_270m_it", model_family="logistic_regression",
        training_window=32, evaluation_regime="Mixed", evaluation_window=32,
    )

    assert len(result) == 1
    assert result.source_dataset.tolist() == ["kaggle_train"]


def test_processed_token_helpers_return_exact_counts_and_decoded_tokens(tmp_path):
    partition = tmp_path / "processed" / "dataset__scorer" / "MG" / "data"
    partition.mkdir(parents=True)
    pd.DataFrame({
        "id": [1, 2],
        "tokens": [
            [{"token": "hello"}, {"token": " world"}],
            [{"token": "machine"}],
        ],
    }).to_parquet(partition / "batch.parquet", index=False)

    counts = load_processed_token_counts(tmp_path, scorer_llm="scorer", dataset="dataset", split="MG")
    labels = load_semeval_token_labels(tmp_path, scorer_llm="scorer", dataset="dataset", split="MG")

    assert counts.set_index("id").token_count.to_dict() == {"1": 2, "2": 1}
    assert labels == {1: ["hello", " world"], 2: ["machine"]}


def test_word_counts_join_raw_documents_by_dataset_split_and_id(tmp_path):
    raw = tmp_path / "raw" / "kaggle_train"
    raw.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "text": ["one two three", "a\nb"]}).to_csv(
        raw / "kaggle_train_hg.csv", index=False,
    )
    documents = pd.DataFrame({
        "source_dataset": ["kaggle_train", "kaggle_train"],
        "split": ["HG", "HG"], "text_id": [1, 2],
    })

    result = load_document_word_counts(tmp_path, documents)

    assert result.set_index("text_id").word_count.to_dict() == {"1": 3, "2": 2}


def test_t08_aggregation_labels_follow_evaluation_representation():
    predictions = pd.DataFrame({
        "evaluation_representation": ["whole_text", "mean_window32", "mean_window64"],
        "aggregation": ["mean_window_score", "single_whole_text_vector", "mean_window_score"],
    })

    normalized = normalize_t08_aggregation_labels(predictions)

    assert normalized.aggregation.tolist() == [
        "single_whole_text_vector", "mean_window_score", "mean_window_score",
    ]
