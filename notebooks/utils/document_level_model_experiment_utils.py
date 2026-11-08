"""Feature engineering and evaluation for document-level window-profile models."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


DOCUMENT_KEY_COLUMNS = ("source_dataset", "split", "text_id")
MODEL_FEATURE_COLUMNS = (
    "mean_window_score",
    "std_window_score",
    "p90_window_score",
    "fraction_windows_ge_0_5",
    "longest_positive_run_fraction",
    "mean_adjacent_score_abs_diff",
)
DESCRIPTIVE_COLUMNS = ("document_token_length", "window_count")
WINDOW_COLUMNS = (*DOCUMENT_KEY_COLUMNS, "start_token", "stop_token", "label", "mg_score")


def _normalize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    frame = windows.copy()
    if "source_dataset" not in frame and "dataset" in frame:
        frame["source_dataset"] = frame["dataset"]
    missing = [column for column in WINDOW_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing window prediction columns: {missing}")
    if frame.empty:
        raise ValueError("Window prediction rows cannot be empty")
    if frame[list(DOCUMENT_KEY_COLUMNS)].isna().any().any():
        raise ValueError("Document keys cannot be null")
    if frame["mg_score"].isna().any():
        raise ValueError("Window scores must be finite")
    if frame[["start_token", "stop_token", "label"]].isna().any().any():
        raise ValueError("Window boundaries and labels cannot be null")
    frame["start_token"] = pd.to_numeric(frame["start_token"], errors="raise")
    frame["stop_token"] = pd.to_numeric(frame["stop_token"], errors="raise")
    frame["mg_score"] = pd.to_numeric(frame["mg_score"], errors="coerce")
    if not np.isfinite(frame["mg_score"]).all():
        raise ValueError("Window scores must be finite")
    if not frame["mg_score"].between(0, 1).all():
        raise ValueError("Window scores must be between 0 and 1")
    if (frame["stop_token"] <= frame["start_token"]).any():
        raise ValueError("Every window must have positive token length")
    if frame.duplicated([*DOCUMENT_KEY_COLUMNS, "start_token"]).any():
        raise ValueError("Found duplicate window start positions for a document")
    label_counts = frame.groupby(list(DOCUMENT_KEY_COLUMNS), dropna=False)["label"].nunique(dropna=False)
    if (label_counts != 1).any():
        raise ValueError("Documents have conflicting labels")
    return frame.sort_values([*DOCUMENT_KEY_COLUMNS, "start_token"], kind="stable").reset_index(drop=True)


def _longest_run_fraction(scores: np.ndarray, threshold: float) -> float:
    positive = scores >= threshold
    longest = current = 0
    for is_positive in positive:
        current = current + 1 if is_positive else 0
        longest = max(longest, current)
    return float(longest / len(scores))


def build_document_feature_table(windows: pd.DataFrame, window_threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate ordered window scores into document-profile features."""
    if not 0 <= window_threshold <= 1:
        raise ValueError("window_threshold must be between 0 and 1")
    frame = _normalize_windows(windows)
    rows = []
    for key, group in frame.groupby(list(DOCUMENT_KEY_COLUMNS), sort=False, dropna=False):
        scores = group["mg_score"].to_numpy(dtype=float)
        key_tuple = key if isinstance(key, tuple) else (key,)
        start = float(group["start_token"].min())
        end = float(group["stop_token"].max())
        row = dict(zip(DOCUMENT_KEY_COLUMNS, key_tuple))
        row.update({
            "document_key": "::".join(str(value) for value in key_tuple),
            "label": int(group["label"].iloc[0]),
            "mean_window_score": float(np.mean(scores)),
            "std_window_score": float(np.std(scores, ddof=0)),
            "p90_window_score": float(np.quantile(scores, 0.9)),
            "fraction_windows_ge_0_5": float(np.mean(scores >= window_threshold)),
            "longest_positive_run_fraction": _longest_run_fraction(scores, window_threshold),
            "mean_adjacent_score_abs_diff": float(np.mean(np.abs(np.diff(scores)))) if len(scores) > 1 else 0.0,
            "document_token_length": end - start,
            "window_count": int(len(scores)),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(DOCUMENT_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def validate_disjoint_document_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise when fully qualified document identities occur in both partitions."""
    required = set(DOCUMENT_KEY_COLUMNS)
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Both partitions must contain {sorted(required)}")
    train_keys = set(train[list(DOCUMENT_KEY_COLUMNS)].astype(str).itertuples(index=False, name=None))
    test_keys = set(test[list(DOCUMENT_KEY_COLUMNS)].astype(str).itertuples(index=False, name=None))
    overlap = sorted(train_keys & test_keys)
    if overlap:
        raise ValueError(f"Validation/test document overlap: {overlap[:5]}")


def load_validation_test_features(
    prediction_root: Path,
    *,
    run_id: str,
    scorer_llm: str,
    model_family: str,
    feature_set: str,
    training_regime: str,
    training_window: int,
    evaluation_regime: str,
    evaluation_window: int,
    partition_column: str = "partition",
    document_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load finalized, matching first-stage artifacts and return validation/test profiles."""
    import pyarrow.parquet as pq

    selection = {
        "run_id": str(run_id),
        "scorer_llm": str(scorer_llm),
        "model_family": str(model_family),
        "feature_set": str(feature_set),
        "training_regime": str(training_regime),
        "training_window": str(training_window),
        "evaluation_regime": str(evaluation_regime),
        "evaluation_window": str(evaluation_window),
    }
    expected_artifact_id = "__".join((
        selection["model_family"], selection["scorer_llm"], selection["feature_set"],
        selection["training_regime"].replace("+", "_"), selection["training_window"], selection["run_id"],
    ))
    partitions: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}
    for metadata_path in sorted(prediction_root.rglob("metadata.json")):
        relative = metadata_path.relative_to(prediction_root)
        if any(part.startswith(".") or part.endswith(".tmp") for part in relative.parts[:-1]):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        partition = str(metadata.get(partition_column, ""))
        if partition not in partitions:
            continue
        if metadata.get("artifact_id") != expected_artifact_id:
            continue
        if str(metadata.get("evaluation_regime", "")) != evaluation_regime:
            continue
        if str(metadata.get("evaluation_window", "")) != str(evaluation_window):
            continue
        data_path = metadata_path.parent / "data.parquet"
        if not data_path.is_file():
            continue
        try:
            parquet = pq.ParquetFile(data_path)
        except (OSError, ValueError):
            continue
        names = set(parquet.schema_arrow.names)
        required = {*DOCUMENT_KEY_COLUMNS, "start_token", "stop_token", "label", "mg_score", *selection}
        if not required.issubset(names):
            continue
        if any(str(metadata.get(key, "")) != value for key, value in selection.items() if key in metadata):
            continue
        identity = parquet.read_row_group(0, columns=list(selection)).slice(0, 1).to_pandas()
        if identity.empty or any(str(identity[key].iloc[0]) != value for key, value in selection.items()):
            continue
        columns = [*DOCUMENT_KEY_COLUMNS, "start_token", "stop_token", "label", "mg_score", *selection]
        if partition_column in names:
            columns.append(partition_column)
        rows = parquet.read(columns=list(dict.fromkeys(columns))).to_pandas()
        if any(not rows[key].astype(str).eq(value).all() for key, value in selection.items()):
            raise ValueError(f"Selected metadata dimensions vary within {data_path}")
        if partition_column in rows and not rows[partition_column].astype(str).eq(partition).all():
            raise ValueError(f"Partition varies within {data_path}")
        partitions[partition].append(build_document_feature_table(rows, document_threshold))

    missing = [partition for partition, frames in partitions.items() if not frames]
    if missing:
        raise FileNotFoundError(f"No matching finalized artifacts for partition(s): {missing} under {prediction_root}")
    result = []
    for partition, frames in partitions.items():
        combined = pd.concat(frames, ignore_index=True)
        if combined["document_key"].duplicated().any():
            raise ValueError(f"Duplicate document predictions across {partition} artifacts")
        for column, value in selection.items():
            combined[column] = value
        combined[partition_column] = partition
        combined["evaluation_partition"] = partition
        result.append(combined.sort_values(list(DOCUMENT_KEY_COLUMNS), kind="stable").reset_index(drop=True))
    validate_disjoint_document_splits(result[0], result[1])
    return result[0], result[1]


def _metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, model: str) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "model": model,
        "evaluation_split": "test",
        "document_count": int(len(labels)),
        "hg_document_count": int(np.sum(labels == 0)),
        "mg_document_count": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "false_positive_rate": float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else np.nan,
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }


def fit_document_models(validation_documents: pd.DataFrame, *, random_state: int = 42) -> dict:
    """Fit LR and additive EBM using validation document profiles only."""
    required = set(MODEL_FEATURE_COLUMNS) | {"label"}
    missing = sorted(required - set(validation_documents.columns))
    if missing:
        raise ValueError(f"validation features are missing required columns: {missing}")
    if validation_documents.empty:
        raise ValueError("validation document table cannot be empty")
    if not validation_documents["label"].isin([0, 1]).all():
        raise ValueError("validation labels must be binary 0/1")
    if validation_documents["label"].nunique() != 2:
        raise ValueError("validation documents must contain both HG and MG classes")
    train_x = validation_documents.loc[:, MODEL_FEATURE_COLUMNS]
    if not np.isfinite(train_x.to_numpy(dtype=float)).all():
        raise ValueError("validation model features must be finite")
    y_train = validation_documents["label"].to_numpy(dtype=int)
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", random_state=random_state, max_iter=2000),
        )
    }
    from interpret.glassbox import ExplainableBoostingClassifier

    models["ebm"] = ExplainableBoostingClassifier(interactions=0, random_state=random_state)
    for model in models.values():
        model.fit(train_x, y_train)
    return models


def score_document_models(
    documents: pd.DataFrame,
    models: Mapping[str, object],
    *,
    document_threshold: float = 0.5,
) -> pd.DataFrame:
    """Score document profiles with the mean-score baseline and supplied fitted models."""
    required = set(MODEL_FEATURE_COLUMNS) | {"label", "mean_window_score"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"document features are missing required columns: {missing}")
    if documents.empty:
        raise ValueError("document table cannot be empty")
    if not 0 <= document_threshold <= 1:
        raise ValueError("document_threshold must be between 0 and 1")
    if not np.isfinite(documents.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("document model features must be finite")
    expected_models = {"logistic_regression", "ebm"}
    missing_models = expected_models - set(models)
    if missing_models:
        raise ValueError(f"fitted document models are missing: {sorted(missing_models)}")
    frames = []
    baseline_scores = documents["mean_window_score"].to_numpy(dtype=float)
    frames.append(documents.assign(
        model="mean_score_baseline",
        prediction=(baseline_scores >= document_threshold).astype(int),
        model_score=baseline_scores,
    ))
    features = documents.loc[:, MODEL_FEATURE_COLUMNS]
    for model_name in ("logistic_regression", "ebm"):
        scores = models[model_name].predict_proba(features)[:, 1]
        frames.append(documents.assign(
            model=model_name,
            prediction=(scores >= document_threshold).astype(int),
            model_score=scores,
        ))
    return pd.concat(frames, ignore_index=True)


def summarize_mg_detection(
    predictions: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("source_llm",),
) -> pd.DataFrame:
    """Report MG detection counts/rates only; reject any non-MG target records."""
    required = {"label", "prediction", "model", *group_columns}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"MG detection results are missing columns: {missing}")
    if predictions.empty:
        raise ValueError("MG detection results cannot be empty")
    if not predictions["label"].eq(1).all():
        raise ValueError("MG-only detection summary requires MG-only target examples")
    if not predictions["prediction"].isin([0, 1]).all():
        raise ValueError("MG predictions must be binary 0/1")
    groups = [*group_columns, "model"]
    return (
        predictions.groupby(groups, dropna=False, as_index=False)
        .agg(
            document_count=("prediction", "size"),
            detected_mg_count=("prediction", "sum"),
            mg_detection_rate=("prediction", "mean"),
        )
        .sort_values(groups, kind="stable")
        .reset_index(drop=True)
    )


def fit_and_evaluate_document_models(
    train: pd.DataFrame, test: pd.DataFrame, *, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit document models on validation profiles and return held-out test results."""
    validate_disjoint_document_splits(train, test)
    for name, frame in (("validation", train), ("test", test)):
        missing = sorted(set(MODEL_FEATURE_COLUMNS + ("label",)) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} features are missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} document table cannot be empty")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{name} labels must be binary 0/1")
    test_x = test.loc[:, MODEL_FEATURE_COLUMNS]
    y_test = test["label"].to_numpy(dtype=int)
    models = fit_document_models(train, random_state=random_state)
    results = []
    metric_rows = []
    baseline_scores = test["mean_window_score"].to_numpy(dtype=float)
    baseline_predictions = (baseline_scores >= 0.5).astype(int)
    results.append(test.assign(model="mean_score_baseline", prediction=baseline_predictions, model_score=baseline_scores))
    metric_rows.append(_metrics(y_test, baseline_predictions, baseline_scores, "mean_score_baseline"))
    for model_name, model in models.items():
        scores = model.predict_proba(test_x)[:, 1]
        predictions = (scores >= 0.5).astype(int)
        results.append(test.assign(model=model_name, prediction=predictions, model_score=scores))
        metric_rows.append(_metrics(y_test, predictions, scores, model_name))
    return pd.concat(results, ignore_index=True), pd.DataFrame(metric_rows), models


def write_document_model_experiment(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: Mapping,
    ebm_model,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atomically replace report outputs, including global EBM explanation data."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    backup = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.bak")
    temporary.mkdir()
    try:
        predictions.sort_values([*DOCUMENT_KEY_COLUMNS, "model"], kind="stable").to_parquet(
            temporary / "document_model_predictions.parquet", index=False
        )
        metrics.sort_values("model", kind="stable").to_csv(temporary / "document_model_metrics.csv", index=False)
        (temporary / "run_manifest.json").write_text(
            json.dumps(dict(manifest), indent=2, default=str) + "\n", encoding="utf-8"
        )
        explanation = ebm_model.explain_global()
        global_data = explanation.data()
        names = global_data.get(" names", global_data.get("names", []))
        scores = global_data.get("scores", [])
        importance_rows = [{"feature": name, "importance": float(np.mean(np.abs(score)))} for name, score in zip(names, scores)]
        importance = pd.DataFrame(importance_rows).sort_values("importance", ascending=False)
        importance.to_csv(temporary / "ebm_global_feature_importance.csv", index=False)
        figure, axis = plt.subplots(figsize=(8, max(3, 0.4 * len(importance))))
        if not importance.empty:
            axis.barh(importance["feature"], importance["importance"])
            axis.invert_yaxis()
        axis.set_xlabel("Mean absolute additive contribution")
        axis.set_title("EBM global feature importance")
        figure.tight_layout()
        figure.savefig(temporary / "ebm_global_feature_importance.png", dpi=160)
        plt.close(figure)
        for old_path in (backup,):
            if old_path.exists():
                shutil.rmtree(old_path)
        if output_dir.exists():
            output_dir.replace(backup)
        temporary.replace(output_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return importance, pd.DataFrame(global_data)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
