from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from notebooks.utils.feature_aggregation_utils import FeatureSet, save_prepared_partition
from notebooks.utils.model_training_utils import fit_feature_vector_model
from notebooks.utils.classification_utils import evaluate_scores
from notebooks.utils.streaming_classification_evaluation_utils import (
    aggregate_prediction_artifact,
    iter_prepared_batches,
    prediction_artifact_is_complete,
    prediction_artifact_path,
    write_prediction_artifact,
)
from notebooks.utils.classification_experiment_utils import (
    DATASETS,
    EVALUATION_REGIMES,
    TRAINING_REGIMES,
    availability_row,
    append_result_frame,
    build_feature_parent_cohorts,
    discover_model_artifacts,
    ensure_result_exports,
    evaluate_feature_vector_model,
    export_semeval_result_frames,
    export_result_frames,
    load_episode_partitions,
    load_training_episode,
    release_frames,
    read_existing_result_table,
    successful_result_keys,
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


def test_load_episode_partitions_accepts_prepared_dataset_identity(tmp_path: Path) -> None:
    frame = _prepared_frame("source", "HG", [1]).assign(dataset="source")
    save_prepared_partition(
        tmp_path,
        frame,
        scorer_llm="llm",
        dataset="source",
        split="HG",
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
        cohort={("source", "HG", 1)},
    )

    assert frames["source"]["dataset"].tolist() == ["source"]


def test_iter_prepared_batches_filters_selected_texts_without_pandas_parquet_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_set = FeatureSet("d", FEATURES)
    for split in ("HG", "MG"):
        save_prepared_partition(
            tmp_path,
            _prepared_frame("source", split, [1, 2, 3]),
            scorer_llm="llm", dataset="source", split=split,
            feature_set=feature_set, window_configuration=32, stride=16,
        )
    path = tmp_path / "experiment-results" / "classification" / "prepared" / "d" / "llm" / "source" / "HG" / "32"
    expected = {
        "schema_version": 1, "scorer_llm": "llm", "dataset": "source", "split": "HG",
        "feature_set": "d", "feature_columns": ["feature_a"], "window_configuration": "32", "stride": 16,
    }
    monkeypatch.setattr(pd, "read_parquet", pytest.fail)

    batches = list(iter_prepared_batches(path, expected=expected, selected_ids={2, 3}, batch_size=1))

    assert [text_id for batch in batches for text_id in batch["text_id"]] == [2, 3]


def test_prediction_artifact_is_atomically_published(tmp_path: Path) -> None:
    key = {
        "artifact_id": "linear-model", "evaluation_regime": "Kaggle", "dataset": "source", "partition": "test",
    }
    path = prediction_artifact_path(tmp_path, key)
    batches = iter((pd.DataFrame({"label": [0, 1], "mg_score": [0.1, 0.9]}),))

    assert write_prediction_artifact(path, expected=key, batches=batches) == 2
    assert prediction_artifact_is_complete(path, expected={**key, "row_count": 2})
    assert not list(path.parent.glob(".*.tmp"))


def test_aggregate_prediction_artifact_matches_evaluate_scores_with_ties(tmp_path: Path) -> None:
    labels = pd.Series([0, 1, 0, 1, 1, 0])
    scores = pd.Series([0.2, 0.2, 0.7, 0.7, 0.9, 0.1])
    path = prediction_artifact_path(tmp_path, {
        "artifact_id": "linear-model", "evaluation_regime": "Kaggle", "dataset": "source", "partition": "test",
    })
    write_prediction_artifact(path, expected={}, batches=iter((pd.DataFrame({"label": labels, "mg_score": scores}),)))

    assert aggregate_prediction_artifact(path, threshold=0.5) == pytest.approx(
        evaluate_scores(labels, scores, 0.5), nan_ok=True,
    )


def test_aggregate_prediction_artifact_matches_evaluate_scores_for_single_class(tmp_path: Path) -> None:
    labels = pd.Series([0, 0, 0])
    scores = pd.Series([0.2, 0.7, 0.1])
    path = prediction_artifact_path(tmp_path, {
        "artifact_id": "linear-model", "evaluation_regime": "Kaggle", "dataset": "source", "partition": "test",
    })
    write_prediction_artifact(path, expected={}, batches=iter((pd.DataFrame({"label": labels, "mg_score": scores}),)))

    assert aggregate_prediction_artifact(path, threshold=0.5) == pytest.approx(
        evaluate_scores(labels, scores, 0.5), nan_ok=True,
    )


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


def test_append_result_frame_writes_header_once_and_appends_rows(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_predictions.csv"
    first = pd.DataFrame([{"id": 1, "score": 0.1}])
    second = pd.DataFrame([{"id": 2, "score": 0.9}])

    append_result_frame(path, first)
    append_result_frame(path, second)

    assert pd.read_csv(path).to_dict(orient="records") == [
        {"id": 1, "score": 0.1},
        {"id": 2, "score": 0.9},
    ]


def test_append_result_frame_replaces_placeholder_header_before_first_batch(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_metrics.csv"
    path.write_text("model_family,partition\n", encoding="utf-8")
    frame = pd.DataFrame([{"model_family": "logistic_regression", "partition": "test", "f1": 0.8}])

    append_result_frame(path, frame)

    assert pd.read_csv(path).to_dict(orient="records") == [
        {"model_family": "logistic_regression", "partition": "test", "f1": 0.8},
    ]


def test_append_result_frame_initializes_newline_only_csv(tmp_path: Path) -> None:
    path = tmp_path / "semeval_boundary_exclusions.csv"
    path.write_text("\n", encoding="utf-8")
    frame = pd.DataFrame([{"dataset": "SemEval", "status": "available"}])

    append_result_frame(path, frame)

    assert pd.read_csv(path).to_dict(orient="records") == [
        {"dataset": "SemEval", "status": "available"},
    ]


def test_ensure_result_exports_preserves_existing_files_and_creates_missing_headers(tmp_path: Path) -> None:
    destination = tmp_path / "experiment-results" / "classification" / "evaluations" / "run-1"
    destination.mkdir(parents=True)
    existing = destination / "training_metrics.csv"
    existing.write_text("model_family,scorer_llm\nlogistic_regression,llm\n", encoding="utf-8")

    standard, semeval = ensure_result_exports(tmp_path, "run-1")

    assert existing.read_text(encoding="utf-8") == "model_family,scorer_llm\nlogistic_regression,llm\n"
    assert standard["training_metrics"] == existing
    assert standard["evaluation_metrics"].read_text(encoding="utf-8").startswith("model_family,")
    assert all(path.is_file() for path in standard.values())
    assert all(path.is_file() for path in semeval.values())


def test_read_existing_result_table_treats_missing_and_empty_tables_as_empty(tmp_path: Path) -> None:
    missing = read_existing_result_table(tmp_path / "missing.csv")
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    empty = read_existing_result_table(empty_path)

    assert missing.empty and empty.empty


def test_successful_result_keys_filters_status_and_normalizes_windows(tmp_path: Path) -> None:
    path = tmp_path / "run_summary.csv"
    pd.DataFrame(
        [
            {"status": "available", "scorer_llm": "llm", "training_window": 32, "model_family": "lr"},
            {"status": "unavailable", "scorer_llm": "other", "training_window": 64, "model_family": "lr"},
            {"status": "available", "scorer_llm": "missing", "training_window": None, "model_family": "lr"},
        ]
    ).to_csv(path, index=False)

    keys = successful_result_keys(path, ("scorer_llm", "training_window", "model_family"))

    assert keys == {("llm", "32", "lr")}


def test_export_result_frames_normalizes_explicit_empty_frames(tmp_path: Path) -> None:
    written = export_result_frames(
        tmp_path,
        run_id="run-empty",
        frames={
            "training_metrics": pd.DataFrame(),
            "training_coverage": pd.DataFrame(),
            "evaluation_metrics": pd.DataFrame(),
            "evaluation_predictions": pd.DataFrame(),
            "evaluation_coverage": pd.DataFrame(),
            "run_summary": pd.DataFrame(),
        },
    )

    assert written["training_metrics"].read_text(encoding="utf-8").splitlines()[0].split(",")[:6] == [
        "model_family", "scorer_llm", "feature_set", "training_regime", "training_window", "partition",
    ]


def test_export_semeval_result_frames_writes_separate_csvs(tmp_path: Path) -> None:
    written = export_semeval_result_frames(
        tmp_path,
        run_id="run-semeval",
        frames={"semeval_summary": pd.DataFrame([{"model": "classifier"}])},
    )

    assert written["semeval_summary"].name == "semeval_summary.csv"
    assert written["semeval_summary"].read_text(encoding="utf-8").splitlines() == ["model", "classifier"]
    assert (written["semeval_summary"].parent / "semeval_boundary_summary.csv").is_file()


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
