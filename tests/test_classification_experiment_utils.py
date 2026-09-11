from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from notebooks.feature_aggregation_utils import FeatureSet, save_prepared_partition
from notebooks.model_training_utils import fit_feature_vector_model
from notebooks.classification_experiment_utils import (
    DATASETS,
    EVALUATION_REGIMES,
    TRAINING_REGIMES,
    availability_row,
    build_feature_parent_cohorts,
    discover_model_artifacts,
    evaluate_feature_vector_model,
    export_result_frames,
    load_episode_partitions,
    load_training_episode,
    release_frames,
    validate_evaluation_metadata,
)


FEATURES = ("feature_a",)


def _prepared_frame(dataset: str, split: str, ids: list[int], window: str = "32") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": [split] * len(ids),
            "text_id": ids,
            "label": [int(split == "MG")] * len(ids),
            "start_token": [0] * len(ids),
            "stop_token": [32] * len(ids),
            "window_configuration": [window] * len(ids),
            "feature_a": [float(index + int(split == "MG")) for index in range(len(ids))],
        }
    )


def test_regimes_are_canonical_and_namespaced_cohorts_are_disjoint() -> None:
    assert DATASETS["Kaggle"] == "kaggle_train"
    assert TRAINING_REGIMES["Mixed"] == ("Kaggle", "MAGE", "Outfox")
    assert EVALUATION_REGIMES["BBC"] == ("BBC",)
    frame = pd.DataFrame(
        [
            {"dataset": dataset, "split": split, "text_id": index, "label": int(split == "MG")}
            for dataset in ("Kaggle", "MAGE")
            for split in ("HG", "MG")
            for index in range(5)
        ]
    )

    cohorts = build_feature_parent_cohorts(frame, seed=42, max_per_split=5)

    assert {name: len(value) for name, value in cohorts.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    assert all(len(identity) == 3 for values in cohorts.values() for identity in values)
    assert set().union(*cohorts.values()) <= {
        (dataset, split, index)
        for dataset in ("Kaggle", "MAGE")
        for split in ("HG", "MG")
        for index in range(5)
    }
    assert all(
        cohorts[first].isdisjoint(cohorts[second])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )


def test_feature_parent_cohorts_support_bbc_human_only_sources() -> None:
    frame = pd.DataFrame({"dataset": ["BBC"] * 5, "split": ["HG"] * 5, "text_id": range(5), "label": [0] * 5})

    cohorts = build_feature_parent_cohorts(frame, seed=42, max_per_split=5)

    assert {name: len(value) for name, value in cohorts.items()} == {"train": 3, "validation": 1, "test": 1}
    assert all(identity[1:] == ("HG", identity[2]) and identity[0] == "BBC" for values in cohorts.values() for identity in values)


def test_load_episode_partitions_filters_parent_cohort_before_returning_windows(tmp_path: Path) -> None:
    for split in ("HG", "MG"):
        save_prepared_partition(
            tmp_path,
            _prepared_frame("source", split, [1, 2]),
            scorer_llm="llm",
            dataset="source",
            split=split,
            feature_set=FeatureSet("d", FEATURES),
            window_configuration=32,
            stride=16,
        )

    frames = load_episode_partitions(
        tmp_path,
        scorer_llm="llm",
        datasets=("source",),
        partition="test",
        feature_set="d",
        window_configuration=32,
        feature_columns=FEATURES,
        cohort={("source", "HG", 2), ("source", "MG", 1)},
    )

    assert set(frames) == {"source"}
    assert set(zip(frames["source"]["split"], frames["source"]["text_id"])) == {
        ("HG", 2),
        ("MG", 1),
    }


def test_training_episode_concatenates_train_and_validation_source_frames(tmp_path: Path) -> None:
    feature_set = FeatureSet("d", FEATURES)
    for split in ("HG", "MG"):
        save_prepared_partition(
            tmp_path,
            _prepared_frame("source", split, [1, 2]),
            scorer_llm="llm", dataset="source", split=split,
            feature_set=feature_set, window_configuration=32, stride=16,
        )

    train, validation = load_training_episode(
        tmp_path, scorer_llm="llm", datasets=("source",), feature_set="d",
        window_configuration=32, feature_columns=FEATURES,
        train_cohort={("source", "HG", 1), ("source", "MG", 1)},
        validation_cohort={("source", "HG", 2), ("source", "MG", 2)},
    )

    assert set(train["text_id"]) == {1}
    assert set(validation["text_id"]) == {2}


def test_discover_model_artifacts_filters_metadata_without_unpickling(tmp_path: Path) -> None:
    matching = tmp_path / "matching"
    different = tmp_path / "different"
    for path, scorer in ((matching, "llm"), (different, "other")):
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({"scorer_llm": scorer, "feature_set": "d"}))
        (path / "model.pkl").write_bytes(b"not a pickle")

    assert discover_model_artifacts(tmp_path, filters={"scorer_llm": "llm", "feature_set": "d"}) == [matching]


def test_discover_model_artifacts_skips_missing_and_invalid_metadata(tmp_path: Path) -> None:
    (tmp_path / "missing").mkdir()
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "metadata.json").write_text("{")

    assert discover_model_artifacts(tmp_path, filters={}) == []


def test_validation_matches_llm_and_ordered_features_but_allows_window_transfer() -> None:
    artifact = {"scorer_llm": "llm", "feature_set": "d", "feature_columns": ["a", "b"], "training_window": "32"}
    prepared = {"scorer_llm": "llm", "feature_set": "d", "feature_columns": ["a", "b"], "window_configuration": "whole_text"}
    validate_evaluation_metadata(artifact, prepared)

    with pytest.raises(ValueError, match="feature columns"):
        validate_evaluation_metadata(artifact, {**prepared, "feature_columns": ["b", "a"]})
    with pytest.raises(ValueError, match="scorer_llm"):
        validate_evaluation_metadata(artifact, {**prepared, "scorer_llm": "other"})
    with pytest.raises(ValueError, match="feature set"):
        validate_evaluation_metadata(artifact, {**prepared, "feature_set": "other"})


def test_evaluate_feature_vector_model_returns_metrics_predictions_and_coverage() -> None:
    train = pd.DataFrame({"split": ["HG", "HG", "MG", "MG"], "text_id": [1, 2, 3, 4], "label": [0, 0, 1, 1], "feature_a": [0.0, 0.1, 1.0, 1.1]})
    fitted = fit_feature_vector_model("logistic_regression", train, feature_columns=FEATURES, seed=42)
    frames = {"source": train.assign(dataset="source", window_configuration="whole_text")}

    metrics, predictions, coverage = evaluate_feature_vector_model(
        fitted, frames, partition="test", metadata={"scorer_llm": "llm", "feature_set": "d", "feature_columns": list(FEATURES), "training_window": "32", "evaluation_window": "whole_text"}
    )

    assert len(metrics) == len(coverage) == 1
    assert len(predictions) == 4
    assert metrics.loc[0, "evaluation_window"] == "whole_text"
    assert coverage.loc[0, "rows"] == 4
    assert {
        "split", "class", "parent_count", "row_count", "availability_status",
        "dataset", "partition",
    }.issubset(coverage.columns)
    assert coverage.loc[0, "split"] == "HG+MG"
    assert coverage.loc[0, "class"] == "binary"
    assert coverage.loc[0, "parent_count"] == 4
    assert coverage.loc[0, "row_count"] == 4
    assert coverage.loc[0, "availability_status"] == "available"


def test_evaluate_feature_vector_model_allows_training_and_evaluation_window_transfer() -> None:
    train = pd.DataFrame({"split": ["HG", "HG", "MG", "MG"], "text_id": [1, 2, 3, 4], "label": [0, 0, 1, 1], "feature_a": [0.0, 0.1, 1.0, 1.1]})
    fitted = fit_feature_vector_model("logistic_regression", train, feature_columns=FEATURES, seed=42)
    evaluation = train.assign(dataset="source", scorer_llm="llm", window_configuration="whole_text")

    metrics, _, _ = evaluate_feature_vector_model(
        fitted,
        {"source": evaluation},
        partition="test",
        metadata={"scorer_llm": "llm", "feature_set": "d", "feature_columns": list(FEATURES), "training_window": "32", "evaluation_window": "whole_text"},
    )

    assert metrics.loc[0, "training_window"] != metrics.loc[0, "evaluation_window"]


def test_export_result_frames_writes_normalized_csvs_and_rejects_missing_run_dimensions(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"model_family": "logistic_regression", "scorer_llm": "llm", "feature_set": "d", "training_regime": "Mixed", "training_window": "32", "evaluation_regime": "Kaggle", "evaluation_window": "whole_text", "partition": "test", "macro_f1": 1.0}])
    written = export_result_frames(tmp_path, run_id="run-1", frames={"evaluation_metrics": frame})
    assert written["evaluation_metrics"].name == "evaluation_metrics.csv"
    assert written["evaluation_metrics"].parent == tmp_path / "experiment-results" / "classification" / "evaluations" / "run-1"
    assert written["evaluation_metrics"].is_file()
    with pytest.raises(ValueError, match="required dimension"):
        export_result_frames(tmp_path, run_id="run-2", frames={"evaluation_metrics": frame.drop(columns="evaluation_window")})

    coverage = pd.DataFrame([{"dataset": "source", "partition": "test"}])
    with pytest.raises(ValueError, match="required dimension"):
        export_result_frames(tmp_path, run_id="run-3", frames={"evaluation_coverage": coverage})


def test_validation_rejects_malformed_feature_columns_without_type_error() -> None:
    with pytest.raises(ValueError, match="feature columns"):
        validate_evaluation_metadata(
            {"scorer_llm": "llm", "feature_set": "d", "feature_columns": None},
            {"scorer_llm": "llm", "feature_set": "d", "feature_columns": ["a"]},
        )


def test_availability_row_is_structured_and_release_frames_collects() -> None:
    row = availability_row(stage="evaluation", status="unavailable", scorer_llm="llm", dataset="Kaggle", split="HG", feature_set="d", window_configuration=32, error="missing")
    assert row["status"] == "unavailable"
    assert row["error"] == "missing"
    assert release_frames(pd.DataFrame()) is None
