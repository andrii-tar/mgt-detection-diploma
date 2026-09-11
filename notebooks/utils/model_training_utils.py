from __future__ import annotations

import json
import pickle
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from notebooks.utils.classification_utils import evaluate_scores


MODEL_FAMILIES = ("logistic_regression",)
ARTIFACT_VERSION = 1
REQUIRED_METADATA = (
    "artifact_version", "run_id", "created_at", "model_family", "scorer_llm",
    "training_datasets", "training_regime", "training_window", "feature_set",
    "feature_columns", "cohort_seed", "stride", "train_row_count",
    "validation_row_count", "train_parent_count", "validation_parent_count",
    "train_hg_row_count", "train_mg_row_count", "validation_hg_row_count",
    "validation_mg_row_count",
)
METRIC_COLUMNS = (
    "model_family", "scorer_llm", "feature_set", "training_regime", "training_window",
    "partition", "sample_count", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
)
COVERAGE_COLUMNS = ("dataset", "split", "class", "parent_count", "row_count", "availability_status")


@dataclass(frozen=True)
class FittedFeatureVectorModel:
    model_family: str
    feature_columns: tuple[str, ...]
    estimator: Pipeline


def _feature_columns(columns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result = tuple(columns)
    if not result or any(not isinstance(column, str) or not column for column in result):
        raise ValueError("feature columns must be a non-empty ordered list of unique names")
    if len(set(result)) != len(result):
        raise ValueError("feature columns must be a non-empty ordered list of unique names")
    return result


def _validate_feature_frame(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    positions = [frame.columns.get_loc(column) for column in feature_columns]
    if positions != sorted(positions):
        raise ValueError("feature columns must preserve their ordered schema")


def _numeric_features(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    raw = frame.loc[:, list(feature_columns)]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError("feature columns must be numeric")
    return numeric.replace([np.inf, -np.inf], np.nan)


def fit_feature_vector_model(
    model_family: str,
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    seed: int,
) -> FittedFeatureVectorModel:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family: {model_family!r}")
    features = _feature_columns(feature_columns)
    required = {"split", "text_id", "label"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"missing training columns: {sorted(missing)}")
    _validate_feature_frame(train, features)
    labels = pd.to_numeric(train["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("training labels must contain both classes 0 and 1")
    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )
    estimator.fit(_numeric_features(train, features), labels.astype("int64"))
    return FittedFeatureVectorModel(model_family, features, estimator)


def score_feature_vector_model(fitted: FittedFeatureVectorModel, frame: pd.DataFrame) -> pd.DataFrame:
    features = _feature_columns(fitted.feature_columns)
    required = {"split", "text_id", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    _validate_feature_frame(frame, features)
    result = frame.copy()
    scores = fitted.estimator.predict_proba(_numeric_features(frame, features))[:, 1]
    result["mg_score"] = scores
    result["prediction"] = (scores >= 0.5).astype("int64")
    return result


def _validate_metadata(metadata: Mapping[str, object], *, run_id: str | None = None) -> dict[str, object]:
    result = dict(metadata)
    missing = [key for key in REQUIRED_METADATA if key not in result]
    if missing:
        raise ValueError(f"metadata is missing required fields: {missing}")
    if result["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact version")
    if run_id is not None and result["run_id"] != run_id:
        raise ValueError("artifact metadata run ID does not match run_id")
    result["feature_columns"] = list(_feature_columns(result["feature_columns"]))  # type: ignore[arg-type]
    if result["model_family"] not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family: {result['model_family']!r}")
    return result


def _validate_tabular_inputs(training_metrics: pd.DataFrame, training_coverage: pd.DataFrame) -> None:
    if not isinstance(training_metrics, pd.DataFrame) or not isinstance(training_coverage, pd.DataFrame):
        raise ValueError("training metrics and coverage must be pandas DataFrames")
    if not training_metrics.empty and set(METRIC_COLUMNS).difference(training_metrics.columns):
        raise ValueError("training metrics are missing required columns")
    if not training_coverage.empty and set(COVERAGE_COLUMNS).difference(training_coverage.columns):
        raise ValueError("training coverage is missing required columns")


def _artifact_path(data_root: Path, fitted: FittedFeatureVectorModel, metadata: Mapping[str, object], run_id: str) -> Path:
    def safe(value: object) -> str:
        result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
        return result or "unknown"

    return data_root / "experiment-results" / "classification" / "models" / safe(fitted.model_family) / safe(metadata["scorer_llm"]) / safe(metadata["feature_set"]) / safe(metadata["training_regime"]) / safe(metadata["training_window"]) / safe(run_id)


def save_model_artifact(
    data_root: Path | str,
    fitted: FittedFeatureVectorModel,
    *,
    metadata: Mapping[str, object],
    run_id: str,
    training_metrics: pd.DataFrame,
    training_coverage: pd.DataFrame,
) -> Path:
    artifact_metadata = _validate_metadata(metadata, run_id=run_id)
    features = _feature_columns(fitted.feature_columns)
    if artifact_metadata["model_family"] != fitted.model_family:
        raise ValueError("metadata model family does not match fitted model")
    if artifact_metadata["feature_columns"] != list(features):
        raise ValueError("metadata feature columns do not match fitted model")
    _validate_tabular_inputs(training_metrics, training_coverage)
    destination = _artifact_path(Path(data_root), fitted, artifact_metadata, run_id)
    if destination.exists():
        raise FileExistsError(f"model artifact already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True)
        with (temporary / "model.pkl").open("wb") as handle:
            pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (temporary / "metadata.json").write_text(json.dumps(artifact_metadata, indent=2, sort_keys=True), encoding="utf-8")
        metrics = training_metrics.reindex(columns=list(METRIC_COLUMNS))
        coverage = training_coverage.reindex(columns=list(COVERAGE_COLUMNS))
        metrics.to_parquet(temporary / "training_metrics.parquet", index=False)
        metrics.to_csv(temporary / "training_metrics.csv", index=False)
        coverage.to_parquet(temporary / "training_coverage.parquet", index=False)
        coverage.to_csv(temporary / "training_coverage.csv", index=False)
        if _validate_metadata(json.loads((temporary / "metadata.json").read_text(encoding="utf-8")), run_id=run_id)["feature_columns"] != list(features):
            raise ValueError("persisted metadata feature columns do not match fitted model")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_model_metadata(path: Path | str, *, expected: Mapping[str, object]) -> dict[str, object]:
    root = Path(path)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid model artifact metadata: {root}") from error
    if not isinstance(metadata, dict):
        raise ValueError("model artifact metadata must be an object")
    result = _validate_metadata(metadata)
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"model artifact metadata mismatch for {key}")
    return result


def load_model_artifact(
    path: Path | str,
    *,
    expected: Mapping[str, object],
    trust_artifact: bool = False,
) -> FittedFeatureVectorModel:
    root = Path(path)
    metadata = load_model_metadata(root, expected=expected)
    if not trust_artifact:
        raise PermissionError("loading a model artifact requires trust_artifact=True")
    try:
        with (root / "model.pkl").open("rb") as handle:
            fitted = pickle.load(handle)
    except (FileNotFoundError, pickle.PickleError, EOFError, AttributeError, ImportError) as error:
        raise ValueError("model artifact pickle is missing or invalid") from error
    if not isinstance(fitted, FittedFeatureVectorModel):
        raise ValueError("model artifact does not contain a fitted feature-vector model")
    if fitted.model_family != metadata["model_family"] or list(fitted.feature_columns) != metadata["feature_columns"]:
        raise ValueError("model artifact feature columns do not match metadata")
    return fitted
