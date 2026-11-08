import json

import numpy as np
import pandas as pd
import pytest

from notebooks.utils.document_level_model_experiment_utils import (
    DESCRIPTIVE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_document_feature_table,
    fit_and_evaluate_document_models,
    fit_document_models,
    load_validation_test_features,
    score_document_models,
    summarize_mg_detection,
    validate_disjoint_document_splits,
    write_document_model_experiment,
)


def _window(source, text_id, start, stop, score, label=1, **metadata):
    return {
        "source_dataset": source,
        "split": "MG" if label else "HG",
        "text_id": text_id,
        "start_token": start,
        "stop_token": stop,
        "mg_score": score,
        "label": label,
        **metadata,
    }


def test_document_profile_features_are_aggregated_from_ordered_windows():
    windows = pd.DataFrame([
        _window("source_a", 3, 0, 32, 0.2),
        _window("source_a", 3, 16, 48, 0.8),
        _window("source_a", 3, 32, 64, 0.9),
    ])

    result = build_document_feature_table(windows)
    row = result.iloc[0]

    assert row["mean_window_score"] == pytest.approx(0.6333333333)
    assert row["std_window_score"] == pytest.approx(np.std([0.2, 0.8, 0.9]))
    assert row["p90_window_score"] == pytest.approx(np.quantile([0.2, 0.8, 0.9], 0.9))
    assert row["fraction_windows_ge_0_5"] == pytest.approx(2 / 3)
    assert row["longest_positive_run_fraction"] == pytest.approx(2 / 3)
    assert row["mean_adjacent_score_abs_diff"] == pytest.approx(0.35)
    assert row["document_token_length"] == 64
    assert row["window_count"] == 3
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint({"document_token_length", "window_count"})


def test_document_identity_includes_source_split_and_text_id():
    result = build_document_feature_table(pd.DataFrame([
        _window("source_a", 3, 0, 32, 0.1, label=0),
        _window("source_b", 3, 0, 32, 0.9, label=1),
    ]))

    assert len(result) == 2
    assert result["document_key"].nunique() == 2


@pytest.mark.parametrize("changes, message", [
    ({"start_token": 0}, "duplicate"),
    ({"label": 0}, "conflicting labels"),
    ({"mg_score": np.nan}, "finite"),
    ({"mg_score": 1.1}, "between 0 and 1"),
])
def test_invalid_window_rows_are_rejected(changes, message):
    base = _window("source_a", 3, 0, 32, 0.2)
    other = _window("source_a", 3, 16, 48, 0.8)
    other.update(changes)
    with pytest.raises(ValueError, match=message):
        build_document_feature_table(pd.DataFrame([base, other]))


def test_single_window_has_zero_adjacent_difference_and_run_fraction():
    row = build_document_feature_table(pd.DataFrame([_window("source_a", 1, 0, 20, 0.5)])).iloc[0]
    assert row["mean_adjacent_score_abs_diff"] == 0.0
    assert row["longest_positive_run_fraction"] == 1.0


def test_validation_and_test_documents_must_be_disjoint():
    train = build_document_feature_table(pd.DataFrame([_window("source_a", 1, 0, 32, 0.2)]))
    test = build_document_feature_table(pd.DataFrame([_window("source_a", 1, 0, 32, 0.9)]))

    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_document_splits(train, test)


def test_loader_ignores_temporary_parquet_and_selects_both_partitions(tmp_path):
    from pathlib import Path

    root = tmp_path / "predictions"
    run = "fixture-run"
    for partition, text_id in (("validation", 1), ("test", 2)):
        artifact = root / "artifact" / partition
        artifact.mkdir(parents=True)
        (artifact / "metadata.json").write_text(json.dumps({
            "artifact_id": "logistic_regression__gemma_3_270m_it__simple_similarity__Kaggle_MAGE_Outfox__32__fixture-run",
            "partition": partition, "evaluation_regime": "Kaggle+MAGE+Outfox", "evaluation_window": "32",
        }), encoding="utf-8")
        pd.DataFrame([_window(
            "source_a", text_id, 0, 32, 0.7, scorer_llm="gemma_3_270m_it",
            model_family="logistic_regression", feature_set="simple_similarity",
            training_regime="Kaggle+MAGE+Outfox", training_window=32,
            evaluation_regime="Kaggle+MAGE+Outfox", evaluation_window=32,
            run_id=run, partition=partition,
        )]).to_parquet(artifact / "data.parquet", index=False)
    temporary = root / ".validation.tmp" / "data.parquet"
    temporary.parent.mkdir()
    temporary.write_bytes(b"PAR1 invalid temporary data")

    train, test = load_validation_test_features(
        root, run_id=run, scorer_llm="gemma_3_270m_it", model_family="logistic_regression",
        feature_set="simple_similarity", training_regime="Kaggle+MAGE+Outfox", training_window=32,
        evaluation_regime="Kaggle+MAGE+Outfox", evaluation_window=32,
    )
    assert train["text_id"].tolist() == [1]
    assert test["text_id"].tolist() == [2]
    assert train["evaluation_regime"].tolist() == ["Kaggle+MAGE+Outfox"]
    assert train["scorer_llm"].tolist() == ["gemma_3_270m_it"]


def test_model_fit_uses_validation_and_returns_test_predictions_only():
    rows = []
    for label in (0, 1):
        for index in range(8):
            text_id = label * 100 + index
            score = (0.15 + index * 0.01) if label == 0 else (0.75 + index * 0.01)
            rows.append({
                "document_key": f"train::{label}::{text_id}", "source_dataset": "train", "split": str(label),
                "text_id": text_id, "label": label,
                "mean_window_score": score, "std_window_score": 0.02, "p90_window_score": score + 0.02,
                "fraction_windows_ge_0_5": float(label), "longest_positive_run_fraction": float(label),
                "mean_adjacent_score_abs_diff": 0.01, "document_token_length": 100, "window_count": 4,
            })
    train = pd.DataFrame(rows)
    test = train.copy()
    test["source_dataset"] = "test"
    test["document_key"] = "test::" + test["document_key"]
    test["text_id"] += 1000

    predictions, metrics, models = fit_and_evaluate_document_models(train, test)

    assert set(predictions["model"]) == {"mean_score_baseline", "logistic_regression", "ebm"}
    assert set(predictions["source_dataset"]) == {"test"}
    assert set(metrics["evaluation_split"]) == {"test"}
    assert set(models) == {"logistic_regression", "ebm"}


def _profile_table(source_dataset, labels, *, source_llms=None):
    rows = []
    for index, label in enumerate(labels):
        score = 0.15 + 0.01 * (index % 8) if label == 0 else 0.75 + 0.01 * (index % 8)
        rows.append({
            "document_key": f"{source_dataset}::{'HG' if label == 0 else 'MG'}::{index}",
            "source_dataset": source_dataset, "split": "HG" if label == 0 else "MG",
            "text_id": index, "label": label,
            "mean_window_score": score, "std_window_score": 0.02,
            "p90_window_score": score + 0.02, "fraction_windows_ge_0_5": float(label),
            "longest_positive_run_fraction": float(label), "mean_adjacent_score_abs_diff": 0.01,
            "document_token_length": 100, "window_count": 4,
        })
        if source_llms is not None:
            rows[-1]["source_llm"] = source_llms[index]
    return pd.DataFrame(rows)


def test_document_models_fit_on_source_validation_and_score_target_profiles():
    validation = _profile_table("main", [0, 1] * 8)
    target = _profile_table("name_that_llm", [1, 1, 1], source_llms=["model_a", "model_a", "model_b"])
    models = fit_document_models(validation)

    predictions = score_document_models(target, models)

    assert set(models) == {"logistic_regression", "ebm"}
    assert set(predictions["model"]) == {"mean_score_baseline", "logistic_regression", "ebm"}
    assert set(predictions["source_dataset"]) == {"name_that_llm"}
    assert set(predictions["source_llm"]) == {"model_a", "model_b"}
    assert len(predictions) == 3 * len(target)
    assert not set(DESCRIPTIVE_COLUMNS) & set(MODEL_FEATURE_COLUMNS)


def test_mg_detection_summary_reports_denominators_and_rejects_non_mg_targets():
    target = _profile_table("name_that_llm", [1, 1, 1], source_llms=["model_a", "model_a", "model_b"])
    target["model"] = "ebm"
    target["prediction"] = [1, 0, 1]

    summary = summarize_mg_detection(target, group_columns=("source_llm",))

    assert summary.set_index("source_llm")["document_count"].to_dict() == {"model_a": 2, "model_b": 1}
    assert summary.set_index("source_llm")["detected_mg_count"].to_dict() == {"model_a": 1, "model_b": 1}
    assert summary.set_index("source_llm")["mg_detection_rate"].to_dict() == {"model_a": 0.5, "model_b": 1.0}
    target.loc[0, "label"] = 0
    with pytest.raises(ValueError, match="MG-only"):
        summarize_mg_detection(target, group_columns=("source_llm",))


def test_output_writer_exports_predictions_metrics_manifest_and_ebm_importance(tmp_path):
    rows = []
    for label in (0, 1):
        for index in range(8):
            score = 0.1 + index * 0.01 if label == 0 else 0.8 + index * 0.01
            rows.append({
                "document_key": f"train::{label}::{index}", "source_dataset": "train", "split": str(label),
                "text_id": label * 100 + index, "label": label,
                "mean_window_score": score, "std_window_score": 0.02, "p90_window_score": score + 0.02,
                "fraction_windows_ge_0_5": float(label), "longest_positive_run_fraction": float(label),
                "mean_adjacent_score_abs_diff": 0.01, "document_token_length": 100, "window_count": 4,
            })
    train = pd.DataFrame(rows)
    test = train.assign(source_dataset="test", document_key=lambda d: "test::" + d.document_key, text_id=lambda d: d.text_id + 1000)
    predictions, metrics, models = fit_and_evaluate_document_models(train, test)
    output = tmp_path / "results"

    importance, _ = write_document_model_experiment(
        output, predictions=predictions, metrics=metrics,
        manifest={"validation_model_selection_caveat": True}, ebm_model=models["ebm"],
    )

    assert {path.name for path in output.iterdir()} == {
        "document_model_predictions.parquet", "document_model_metrics.csv", "run_manifest.json",
        "ebm_global_feature_importance.csv", "ebm_global_feature_importance.png",
    }
    assert len(pd.read_parquet(output / "document_model_predictions.parquet")) == len(predictions)
    assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["validation_model_selection_caveat"]
    assert set(importance["feature"]) == set(MODEL_FEATURE_COLUMNS)
    write_document_model_experiment(
        output, predictions=predictions, metrics=metrics,
        manifest={"validation_model_selection_caveat": True}, ebm_model=models["ebm"],
    )
    assert len(pd.read_parquet(output / "document_model_predictions.parquet")) == len(predictions)
