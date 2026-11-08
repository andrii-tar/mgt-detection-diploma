from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from notebooks.utils.model_training_utils import (
    FittedFeatureVectorModel,
    MODEL_FAMILIES,
    fit_feature_vector_model,
    load_model_artifact,
    load_model_metadata,
    save_model_artifact,
    score_feature_vector_model,
)


FEATURES = ("feature_a",)
TRAIN = pd.DataFrame(
    {
        "split": ["HG", "HG", "MG", "MG"],
        "text_id": [1, 2, 3, 4],
        "label": [0, 0, 1, 1],
        "feature_a": [0.0, 0.1, 1.0, 1.1],
        "window_configuration": ["32"] * 4,
    }
)
METADATA = {
    "artifact_version": 1,
    "run_id": "run-1",
    "created_at": "2026-09-11T12:00:00+00:00",
    "model_family": "logistic_regression",
    "scorer_llm": "llm",
    "training_datasets": ["source"],
    "training_regime": "Mixed",
    "training_window": "32",
    "feature_set": "d",
    "feature_columns": ["feature_a"],
    "cohort_seed": 42,
    "stride": 16,
    "train_row_count": 4,
    "validation_row_count": 0,
    "train_parent_count": 4,
    "validation_parent_count": 0,
    "train_hg_row_count": 2,
    "train_mg_row_count": 2,
    "validation_hg_row_count": 0,
    "validation_mg_row_count": 0,
}


def _fit() -> FittedFeatureVectorModel:
    return fit_feature_vector_model("logistic_regression", TRAIN, feature_columns=FEATURES, seed=42)


def test_logistic_training_uses_only_ordered_feature_columns() -> None:
    fitted = _fit()
    scored = score_feature_vector_model(fitted, TRAIN)

    assert fitted.feature_columns == FEATURES
    assert {"mg_score", "prediction"}.issubset(scored.columns)
    assert "window_configuration" not in fitted.estimator.feature_names_in_
    assert isinstance(fitted.estimator, Pipeline)
    assert isinstance(fitted.estimator.named_steps["impute"], SimpleImputer)
    assert isinstance(fitted.estimator.named_steps["scale"], StandardScaler)
    assert fitted.estimator.named_steps["model"].class_weight == "balanced"


def test_random_forest_training_scores_without_feature_scaling() -> None:
    fitted = fit_feature_vector_model("random_forest", TRAIN, feature_columns=FEATURES, seed=42)
    scored = score_feature_vector_model(fitted, TRAIN)

    assert "random_forest" in MODEL_FAMILIES
    assert isinstance(fitted.estimator.named_steps["impute"], SimpleImputer)
    assert "scale" not in fitted.estimator.named_steps
    assert isinstance(fitted.estimator.named_steps["model"], RandomForestClassifier)
    assert fitted.estimator.named_steps["model"].random_state == 42
    assert fitted.estimator.named_steps["model"].class_weight == "balanced"
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert scored["prediction"].isin([0, 1]).all()


@pytest.mark.parametrize(
    ("model_family", "kernel"),
    [
        ("svm_linear", "linear"),
        ("svm_rbf", "rbf"),
        ("svm_poly", "poly"),
        ("svm_sigmoid", "sigmoid"),
    ],
)
def test_svm_training_scores_with_scaled_probabilistic_estimator(
    model_family: str, kernel: str,
) -> None:
    fitted = fit_feature_vector_model(model_family, TRAIN, feature_columns=FEATURES, seed=42)
    scored = score_feature_vector_model(fitted, TRAIN)

    assert model_family in MODEL_FAMILIES
    assert isinstance(fitted.estimator.named_steps["impute"], SimpleImputer)
    assert isinstance(fitted.estimator.named_steps["scale"], StandardScaler)
    assert isinstance(fitted.estimator.named_steps["model"], SVC)
    assert fitted.estimator.named_steps["model"].kernel == kernel
    assert fitted.estimator.named_steps["model"].probability is True
    assert fitted.estimator.named_steps["model"].class_weight == "balanced"
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert scored["prediction"].isin([0, 1]).all()


def test_random_forest_artifact_round_trip(tmp_path: Path) -> None:
    fitted = fit_feature_vector_model("random_forest", TRAIN, feature_columns=FEATURES, seed=42)
    metadata = {**METADATA, "model_family": "random_forest"}
    artifact = save_model_artifact(
        tmp_path,
        fitted,
        metadata=metadata,
        run_id="run-1",
        training_metrics=pd.DataFrame(),
        training_coverage=pd.DataFrame(),
    )

    loaded = load_model_artifact(artifact, expected=metadata, trust_artifact=True)

    assert loaded.model_family == "random_forest"
    assert isinstance(loaded.estimator.named_steps["model"], RandomForestClassifier)


@pytest.mark.parametrize(
    "model_family", ["svm_linear", "svm_rbf", "svm_poly", "svm_sigmoid"],
)
def test_svm_artifact_round_trip(tmp_path: Path, model_family: str) -> None:
    fitted = fit_feature_vector_model(model_family, TRAIN, feature_columns=FEATURES, seed=42)
    metadata = {**METADATA, "model_family": model_family}
    artifact = save_model_artifact(
        tmp_path,
        fitted,
        metadata=metadata,
        run_id="run-1",
        training_metrics=pd.DataFrame(),
        training_coverage=pd.DataFrame(),
    )

    loaded = load_model_artifact(artifact, expected=metadata, trust_artifact=True)

    assert loaded.model_family == model_family
    assert isinstance(loaded.estimator.named_steps["model"], SVC)


def test_training_coerces_non_numeric_values_and_imputes_missing_values() -> None:
    train = TRAIN.copy()
    train.loc[0, "feature_a"] = None

    fitted = fit_feature_vector_model("logistic_regression", train, feature_columns=FEATURES, seed=42)

    assert fitted.estimator.named_steps["impute"].statistics_[0] == pytest.approx(1.0)


def test_score_requires_exact_feature_schema_and_preserves_provenance() -> None:
    train = TRAIN.assign(feature_b=[0.0, 0.1, 1.0, 1.1])
    fitted = fit_feature_vector_model("logistic_regression", train, feature_columns=("feature_a", "feature_b"), seed=42)
    reordered = train[["split", "text_id", "label", "feature_b", "feature_a", "window_configuration"]]

    with pytest.raises(ValueError, match="feature columns"):
        score_feature_vector_model(fitted, reordered)

    scored = score_feature_vector_model(fitted, train.drop(columns=["window_configuration"]))
    assert scored["split"].tolist() == TRAIN["split"].tolist()
    assert scored["text_id"].tolist() == TRAIN["text_id"].tolist()
    assert scored["prediction"].isin([0, 1]).all()


@pytest.mark.parametrize(
    ("family", "features", "frame"),
    [
        ("unknown", FEATURES, TRAIN),
        ("logistic_regression", (), TRAIN),
        ("logistic_regression", ("feature_a", "feature_a"), TRAIN),
        ("logistic_regression", FEATURES, TRAIN.drop(columns=["feature_a"])),
        ("logistic_regression", FEATURES, TRAIN.assign(feature_a=pd.Series(["x"] * 4, dtype="object"))),
        ("logistic_regression", FEATURES, TRAIN.assign(label=[0] * 4)),
    ],
)
def test_fit_rejects_invalid_model_inputs(family: str, features: tuple[str, ...], frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_feature_vector_model(family, frame, feature_columns=features, seed=42)


def test_model_artifact_round_trip_requires_explicit_trust_and_exact_schema(tmp_path: Path) -> None:
    artifact = save_model_artifact(
        tmp_path,
        _fit(),
        metadata={**METADATA, "extra": "preserved"},
        run_id="run-1",
        training_metrics=pd.DataFrame(),
        training_coverage=pd.DataFrame(),
    )

    with pytest.raises(PermissionError, match="trust_artifact=True"):
        load_model_artifact(artifact, expected=METADATA)
    loaded = load_model_artifact(artifact, expected=METADATA, trust_artifact=True)
    metadata = load_model_metadata(artifact, expected={**METADATA, "extra": "preserved"})

    assert loaded.feature_columns == FEATURES
    assert metadata["extra"] == "preserved"
    assert {"model.pkl", "metadata.json", "training_metrics.parquet", "training_metrics.csv",
            "training_coverage.parquet", "training_coverage.csv"} == {path.name for path in artifact.iterdir()}


def test_artifact_metadata_is_complete_and_matches_fitted_model(tmp_path: Path) -> None:
    incomplete = dict(METADATA)
    del incomplete["validation_mg_row_count"]

    with pytest.raises(ValueError, match="metadata"):
        save_model_artifact(tmp_path, _fit(), metadata=incomplete, run_id="run-1",
                            training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())

    with pytest.raises(ValueError, match="feature columns"):
        save_model_artifact(tmp_path, _fit(), metadata={**METADATA, "feature_columns": ["other"]},
                            run_id="run-1", training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())


def test_load_rejects_expected_metadata_mismatch_and_fitted_schema_conflict(tmp_path: Path) -> None:
    artifact = save_model_artifact(tmp_path, _fit(), metadata=METADATA, run_id="run-1",
                                   training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_model_metadata(artifact, expected={**METADATA, "training_window": "whole_text"})

    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_columns"] = ["other"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature columns"):
        load_model_artifact(artifact, expected={}, trust_artifact=True)


def test_artifact_save_is_atomic_and_rejects_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_csv(self: pd.DataFrame, path: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("csv failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)
    with pytest.raises(RuntimeError, match="csv failed"):
        save_model_artifact(tmp_path, _fit(), metadata=METADATA, run_id="run-1",
                            training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())
    assert not list(tmp_path.rglob("*.tmp"))

    monkeypatch.undo()
    save_model_artifact(tmp_path, _fit(), metadata=METADATA, run_id="run-1",
                        training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())
    with pytest.raises(FileExistsError):
        save_model_artifact(tmp_path, _fit(), metadata=METADATA, run_id="run-1",
                            training_metrics=pd.DataFrame(), training_coverage=pd.DataFrame())


@pytest.mark.parametrize("value", ["../escape", "nested/name", "bad\\name", ""])
def test_artifact_path_components_are_sanitized(tmp_path: Path, value: str) -> None:
    artifact = save_model_artifact(
        tmp_path,
        _fit(),
        metadata={**METADATA, "run_id": "../run/name", "feature_set": value or "valid"},
        run_id="../run/name",
        training_metrics=pd.DataFrame(),
        training_coverage=pd.DataFrame(),
    )

    assert artifact.is_relative_to(tmp_path)
    assert artifact.name == "run_name"
    assert ".." not in artifact.parts
