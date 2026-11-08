import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from notebooks.utils.document_window_evaluation_utils import (
    aggregate_window_chunks,
    aggregate_window_predictions,
    common_document_keys,
    evaluate_document_predictions,
    evaluate_configurations,
    load_document_configurations,
)


def _row(source="kaggle_train", split="HG", text_id=1, window_id=0, label=0, score=0.2):
    return {
        "source_dataset": source,
        "split": split,
        "text_id": text_id,
        "window_id": window_id,
        "label": label,
        "mg_score": score,
    }


def test_aggregate_window_predictions_averages_continuous_scores_and_thresholds_at_half():
    rows = pd.DataFrame([_row(window_id=0, score=0.2), _row(window_id=1, score=0.8)])

    result = aggregate_window_predictions(rows, document_threshold=0.5)

    assert result.loc[0, "document_score"] == pytest.approx(0.5)
    assert result.loc[0, "prediction"] == 1
    assert result.loc[0, "window_count"] == 2


def test_document_key_keeps_same_text_id_from_different_sources_separate():
    rows = pd.DataFrame([_row(source="kaggle_train", score=0.1), _row(source="mage_train", score=0.9)])

    result = aggregate_window_predictions(rows, document_threshold=0.5)

    assert len(result) == 2
    assert result["document_key"].nunique() == 2


def test_duplicate_windows_and_conflicting_labels_are_rejected():
    duplicate = pd.DataFrame([_row(score=0.1), _row(score=0.2)])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_window_predictions(duplicate, document_threshold=0.5)

    conflicting = pd.DataFrame([_row(label=0), _row(window_id=1, label=1)])
    with pytest.raises(ValueError, match="label"):
        aggregate_window_predictions(conflicting, document_threshold=0.5)


def test_invalid_scores_are_rejected():
    rows = pd.DataFrame([_row(score=np.nan)])
    with pytest.raises(ValueError, match="finite"):
        aggregate_window_predictions(rows, document_threshold=0.5)


def test_document_metrics_use_document_scores_not_window_rows():
    documents = pd.DataFrame([
        {"document_key": "hg", "label": 0, "document_score": 0.49},
        {"document_key": "mg", "label": 1, "document_score": 0.51},
    ])

    metrics = evaluate_document_predictions(documents, document_threshold=0.5)

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["document_count"] == 2
    assert metrics["evaluation_unit"] == "document"
    assert metrics["aggregation"] == "mean_window_score"
    assert metrics["tn"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 1
    assert metrics["false_positive_rate"] == 0.0


def test_common_document_keys_are_intersected():
    first = pd.DataFrame({"document_key": ["a", "b", "c"]})
    second = pd.DataFrame({"document_key": ["b", "c", "d"]})

    assert common_document_keys([first, second]) == {"b", "c"}


def test_window_rows_from_multiple_chunks_are_aggregated_once():
    first = pd.DataFrame([_row(text_id=4, window_id=0, score=0.2)])
    second = pd.DataFrame([_row(text_id=4, window_id=1, score=0.8)])

    result = aggregate_window_chunks([first, second], document_threshold=0.5)

    assert len(result) == 1
    assert result.loc[0, "document_score"] == pytest.approx(0.5)


@pytest.mark.parametrize("incomplete_directory,has_metadata", [
    (".validation.unfinished.tmp", False),
    (".validation.unfinished.tmp", True),
    ("unfinished-artifact", False),
])
def test_load_document_configurations_reads_artifacts_and_ignores_uncommitted_files(
    tmp_path, incomplete_directory, has_metadata,
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps({"partition": "test"}), encoding="utf-8")
    incomplete = tmp_path / incomplete_directory
    incomplete.mkdir()
    (incomplete / "data.parquet").write_bytes(b"PAR1unfinished parquet without footer")
    if has_metadata:
        (incomplete / "metadata.json").write_text(json.dumps({"partition": "test"}), encoding="utf-8")
    pd.DataFrame([
        {"source_dataset": "kaggle_train", "split": "HG", "text_id": 1, "start_token": 0, "label": 0, "mg_score": 0.2,
         "scorer_llm": "gemma_3_270m_it", "model_family": "logistic_regression", "evaluation_regime": "Mixed", "evaluation_window": 32,
         "training_window": 32, "feature_set": "simple_similarity", "partition": "test"},
        {"source_dataset": "kaggle_train", "split": "HG", "text_id": 1, "start_token": 16, "label": 0, "mg_score": 0.8,
         "scorer_llm": "gemma_3_270m_it", "model_family": "logistic_regression", "evaluation_regime": "Mixed", "evaluation_window": 32,
         "training_window": 32, "feature_set": "simple_similarity", "partition": "test"},
    ]).to_parquet(artifact / "data.parquet", index=False)

    result = load_document_configurations(tmp_path, scorer_llms=("gemma_3_270m_it",), model_families=("logistic_regression",), evaluation_regimes=("Mixed",), evaluation_windows=(32,), training_window=32, feature_set="simple_similarity", partition="test")

    assert list(result) == ["logistic_regression__gemma_3_270m_it__Mixed__32"]
    assert result[next(iter(result))].loc[0, "document_score"] == pytest.approx(0.5)


def test_common_evaluation_keeps_regime_samples_separate_and_reports_exclusions():
    def documents(ids, regime):
        frame = aggregate_window_predictions(pd.DataFrame([_row(text_id=value) for value in ids]))
        return frame.assign(evaluation_regime=regime)

    predictions, metrics, coverage = evaluate_configurations({
        "mixed-a": documents([1, 2], "Mixed"),
        "mixed-b": documents([2, 3], "Mixed"),
        "bbc-a": documents([4], "Mixed+BBC"),
        "bbc-b": documents([4], "Mixed+BBC"),
    })

    assert set(predictions["text_id"]) == {2, 4}
    assert metrics["document_count"].tolist() == [1, 1, 1, 1]
    assert coverage["excluded_document_count"].tolist() == [1, 1, 0, 0]


def test_empty_configuration_input_reports_missing_data_instead_of_empty_tables():
    with pytest.raises(ValueError, match="No document configurations"):
        evaluate_configurations({})


def test_common_scope_can_compare_classifiers_within_each_scorer():
    def documents(ids, scorer):
        return aggregate_window_predictions(pd.DataFrame([_row(text_id=value) for value in ids])).assign(
            evaluation_regime="Mixed", scorer_llm=scorer,
        )

    configurations = {
        "a-lr": documents([1, 2], "a"), "a-rf": documents([2, 3], "a"),
        "b-lr": documents([4, 5], "b"), "b-rf": documents([5, 6], "b"),
    }
    with pytest.raises(ValueError, match="No common documents"):
        evaluate_configurations(configurations)
    predictions, metrics, coverage = evaluate_configurations(
        configurations, common_group_columns=("evaluation_regime", "scorer_llm"),
    )
    assert set(predictions["text_id"]) == {2, 5}
    assert metrics["document_count"].tolist() == [1, 1, 1, 1]
    for table in (predictions, metrics, coverage):
        assert table["common_scope"].eq("evaluation_regime+scorer_llm").all()
    assert coverage["excluded_document_count"].tolist() == [1, 1, 1, 1]


def test_common_group_columns_must_exist_and_be_constant_per_configuration():
    documents = aggregate_window_predictions(pd.DataFrame([_row(text_id=1), _row(text_id=2)]))
    with pytest.raises(ValueError, match="scorer_llm"):
        evaluate_configurations({"a": documents}, common_group_columns=("scorer_llm",))
    documents["scorer_llm"] = ["a", "b"]
    with pytest.raises(ValueError, match="scorer_llm"):
        evaluate_configurations({"a": documents}, common_group_columns=("scorer_llm",))
