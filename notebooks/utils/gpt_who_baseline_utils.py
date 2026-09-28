from __future__ import annotations

import math
import hashlib
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GPT_WHO_FEATURE_COLUMNS = (
    "uid_mean_surprisal",
    "uid_surprisal_variance",
    "uid_mean_abs_surprisal_delta",
    "uid_mean_squared_surprisal_delta",
    "uid_min_window_surprisal_variance",
    "uid_max_window_surprisal_variance",
    "uid_min_window_abs_delta",
    "uid_max_window_abs_delta",
    "uid_min_window_squared_delta",
    "uid_max_window_squared_delta",
    "uid_valid_token_count",
)

_IDENTITY_COLUMNS = ("id", "text", "dataset", "model", "split", "partition")


def _finite_surprisals(row: Mapping[str, object]) -> np.ndarray:
    raw_tokens = row.get("tokens")
    tokens = [] if raw_tokens is None else list(raw_tokens)
    values = []
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        value = token.get("logprob")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            values.append(-numeric_value)
    return np.asarray(values, dtype="float64")


def _window_statistics(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variances = []
    absolute_deltas = []
    squared_deltas = []
    for start in range(values.size - window_size + 1):
        window = values[start : start + window_size]
        deltas = np.diff(window)
        variances.append(float(np.var(window)))
        absolute_deltas.append(float(np.mean(np.abs(deltas))))
        squared_deltas.append(float(np.mean(deltas**2)))
    return (
        np.asarray(variances, dtype="float64"),
        np.asarray(absolute_deltas, dtype="float64"),
        np.asarray(squared_deltas, dtype="float64"),
    )


def compute_gpt_who_features(row: Mapping[str, object], *, window_size: int = 20) -> dict[str, float]:
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1")

    surprisals = _finite_surprisals(row)
    if surprisals.size == 0:
        raise ValueError("no valid logprobs")
    if surprisals.size < window_size:
        raise ValueError(f"fewer than {window_size} valid tokens")

    deltas = np.diff(surprisals)
    window_variances, window_absolute_deltas, window_squared_deltas = _window_statistics(surprisals, window_size)
    return {
        "uid_mean_surprisal": float(np.mean(surprisals)),
        "uid_surprisal_variance": float(np.var(surprisals)),
        "uid_mean_abs_surprisal_delta": float(np.mean(np.abs(deltas))),
        "uid_mean_squared_surprisal_delta": float(np.mean(deltas**2)),
        "uid_min_window_surprisal_variance": float(np.min(window_variances)),
        "uid_max_window_surprisal_variance": float(np.max(window_variances)),
        "uid_min_window_abs_delta": float(np.min(window_absolute_deltas)),
        "uid_max_window_abs_delta": float(np.max(window_absolute_deltas)),
        "uid_min_window_squared_delta": float(np.min(window_squared_deltas)),
        "uid_max_window_squared_delta": float(np.max(window_squared_deltas)),
        "uid_valid_token_count": int(surprisals.size),
    }


def build_gpt_who_feature_frame(
    rows: Iterable[Mapping[str, object]], *, window_size: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_records = []
    exclusion_records = []
    for row in rows:
        identity = {column: row[column] for column in _IDENTITY_COLUMNS if column in row}
        try:
            features = compute_gpt_who_features(row, window_size=window_size)
        except ValueError as error:
            reason = "no_valid_logprobs" if "no valid" in str(error) else "short_text"
            exclusion_records.append({**identity, "reason": reason})
            continue
        feature_records.append({**identity, **features})

    identity_columns = [column for column in _IDENTITY_COLUMNS if any(column in record for record in feature_records)]
    feature_frame = pd.DataFrame(feature_records, columns=[*identity_columns, *GPT_WHO_FEATURE_COLUMNS])
    exclusion_identity_columns = [
        column for column in _IDENTITY_COLUMNS if any(column in record for record in exclusion_records)
    ]
    exclusions = pd.DataFrame(exclusion_records, columns=[*exclusion_identity_columns, "reason"])
    return feature_frame, exclusions


def _stable_group_score(value: object, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_parent_cohorts(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    group_column: str = "id",
) -> dict[str, pd.DataFrame]:
    if group_column not in frame.columns:
        raise ValueError(f"group_column {group_column!r} is missing")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if frame[group_column].isna().any():
        raise ValueError(f"group_column {group_column!r} contains missing values")

    group_labels: dict[object, object] = {}
    if "label" in frame.columns:
        for group, group_frame in frame.groupby(group_column, sort=False, dropna=False):
            labels = group_frame["label"].dropna().unique()
            if len(labels) > 1:
                raise ValueError(f"group {group!r} contains multiple labels")
            if len(labels) == 1:
                group_labels[group] = labels[0]

    groups = list(frame[group_column].drop_duplicates())
    validation_groups: set[object] = set()
    if group_labels and len(set(group_labels.values())) > 1:
        for label in sorted(set(group_labels.values()), key=str):
            label_groups = [group for group in groups if group_labels.get(group) == label]
            validation_count = max(1, min(len(label_groups) - 1, round(len(label_groups) * validation_fraction)))
            ordered = sorted(label_groups, key=lambda group: _stable_group_score(group, seed))
            validation_groups.update(ordered[:validation_count])
    else:
        validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
        ordered = sorted(groups, key=lambda group: _stable_group_score(group, seed))
        validation_groups.update(ordered[:validation_count])

    validation_mask = frame[group_column].isin(validation_groups)
    return {
        "train": frame.loc[~validation_mask].copy(),
        "validation": frame.loc[validation_mask].copy(),
    }


def make_mixed_parent_cohorts(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    max_per_split: int = 500,
    train_count_per_class: int = 900,
    eval_count_per_class: int = 300,
) -> dict[str, pd.DataFrame]:
    required = {"dataset", "split", "id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing mixed cohort columns: {missing}")
    if max_per_split <= 0 or train_count_per_class <= 0 or eval_count_per_class <= 0:
        raise ValueError("mixed cohort counts must be positive")
    if train_count_per_class % frame["dataset"].nunique() or eval_count_per_class % frame["dataset"].nunique():
        raise ValueError("mixed cohort counts must divide evenly across datasets")

    parents = frame.drop_duplicates(["dataset", "split", "id"]).copy()
    datasets = sorted(parents["dataset"].dropna().unique(), key=str)
    if not datasets:
        raise ValueError("frame must contain at least one dataset")
    if set(parents["label"].dropna().unique()) != {0, 1}:
        raise ValueError("mixed cohort must contain labels 0 and 1")

    selected: dict[str, list[pd.DataFrame]] = {"train": [], "validation": []}
    train_per_source = train_count_per_class // len(datasets)
    eval_per_source = eval_count_per_class // len(datasets)
    for dataset in datasets:
        source = parents.loc[parents["dataset"].eq(dataset)].copy()
        source["_numeric_id"] = pd.to_numeric(source["id"], errors="raise")
        source = (
            source.sort_values(["split", "_numeric_id"], kind="stable")
            .groupby("split", sort=False, group_keys=False)
            .head(max_per_split)
            .drop(columns="_numeric_id")
        )
        for label in (0, 1):
            source_class = source.loc[source["label"].eq(label)].copy()
            source_class = source_class.assign(
                _stable_order=source_class.apply(
                    lambda row: _stable_group_score(f"{dataset}:{row['split']}:{row['id']}", seed),
                    axis=1,
                )
            ).sort_values(["_stable_order", "split", "id"], kind="stable")
            required_count = train_per_source + eval_per_source
            if len(source_class) < required_count:
                raise ValueError(
                    f"dataset {dataset!r} label {label} has {len(source_class)} parents; "
                    f"{required_count} are required"
                )
            selected["train"].append(source_class.iloc[:train_per_source].drop(columns="_stable_order"))
            selected["validation"].append(
                source_class.iloc[train_per_source:required_count].drop(columns="_stable_order")
            )

    return {
        partition: pd.concat(frames, ignore_index=True)
        for partition, frames in selected.items()
    }


def make_pooled_parent_cohorts(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    max_per_split: int = 500,
    train_count_per_class: int = 900,
    eval_count_per_class: int = 300,
) -> dict[str, pd.DataFrame]:
    """Build balanced cohorts from a pooled multi-dataset parent population."""
    required = {"dataset", "split", "id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing pooled cohort columns: {missing}")
    if max_per_split <= 0 or train_count_per_class <= 0 or eval_count_per_class <= 0:
        raise ValueError("pooled cohort counts must be positive")

    parents = frame.drop_duplicates(["dataset", "split", "id"]).copy()
    if parents.empty:
        raise ValueError("pooled cohort must contain at least one parent")
    parents["_stable_order"] = parents.apply(
        lambda row: _stable_group_score(f"{row['dataset']}:{row['split']}:{row['id']}", seed),
        axis=1,
    )
    parents = pd.concat(
        [
            source.sort_values("_stable_order", kind="stable").head(max_per_split)
            for _, source in parents.groupby("split", sort=True)
        ],
        ignore_index=True,
    )

    selected: dict[str, list[pd.DataFrame]] = {"train": [], "validation": []}
    for label in (0, 1):
        source_class = parents.loc[parents["label"].eq(label)].sort_values(
            ["_stable_order", "dataset", "split", "id"], kind="stable"
        )
        required_count = train_count_per_class + eval_count_per_class
        if len(source_class) < required_count:
            raise ValueError(
                f"pooled label {label} has {len(source_class)} parents; {required_count} are required"
            )
        selected["train"].append(source_class.iloc[:train_count_per_class].drop(columns="_stable_order"))
        selected["validation"].append(
            source_class.iloc[train_count_per_class:required_count].drop(columns="_stable_order")
        )

    return {
        partition: pd.concat(frames, ignore_index=True)
        for partition, frames in selected.items()
    }


def make_experiment_cohorts(
    frame: pd.DataFrame,
    *,
    training_datasets: Sequence[str],
    evaluation_datasets: Sequence[str],
    seed: int = 42,
    max_per_split: int = 500,
    train_count_per_class: int = 900,
    eval_count_per_class: int = 300,
) -> dict[str, pd.DataFrame]:
    required = {"dataset", "split", "id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing experiment cohort columns: {missing}")
    if not training_datasets or not evaluation_datasets:
        raise ValueError("training and evaluation datasets must not be empty")

    evaluation_frame = frame.loc[frame["dataset"].isin(evaluation_datasets)].copy()
    evaluation = make_mixed_parent_cohorts(
        evaluation_frame,
        seed=seed,
        max_per_split=max_per_split,
        train_count_per_class=eval_count_per_class,
        eval_count_per_class=eval_count_per_class,
    )["validation"]
    evaluation_keys = set(zip(evaluation["dataset"], evaluation["split"], evaluation["id"]))

    training_frame = frame.loc[frame["dataset"].isin(training_datasets)].copy()
    training_keys = list(zip(training_frame["dataset"], training_frame["split"], training_frame["id"]))
    training_frame = training_frame.loc[
        [key not in evaluation_keys for key in training_keys]
    ]
    training = make_pooled_parent_cohorts(
        training_frame,
        seed=seed,
        max_per_split=max(max_per_split, train_count_per_class + max(1, len(tuple(training_datasets)))),
        train_count_per_class=train_count_per_class,
        eval_count_per_class=max(1, len(tuple(training_datasets))),
    )["train"]
    return {"train": training, "evaluation": evaluation}


def fit_and_score_logistic(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = GPT_WHO_FEATURE_COLUMNS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = tuple(feature_columns)
    missing_columns = [column for column in [*feature_columns, "label"] if column not in train.columns]
    if missing_columns:
        raise ValueError(f"training frame is missing columns: {missing_columns}")
    missing_evaluation_columns = [column for column in [*feature_columns, "label"] if column not in evaluation.columns]
    if missing_evaluation_columns:
        raise ValueError(f"evaluation frame is missing columns: {missing_evaluation_columns}")
    if train.empty or evaluation.empty:
        raise ValueError("training and evaluation frames must not be empty")
    if train["label"].nunique(dropna=True) < 2:
        raise ValueError("training frame must contain both labels")

    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    )
    estimator.fit(train.loc[:, feature_columns], train["label"].astype("int64"))
    probabilities = estimator.predict_proba(evaluation.loc[:, feature_columns])
    classes = estimator.named_steps["classifier"].classes_
    mg_index = int(np.flatnonzero(classes == 1)[0])
    predictions = estimator.predict(evaluation.loc[:, feature_columns]).astype("int64")

    identity_columns = [column for column in _IDENTITY_COLUMNS if column in evaluation.columns]
    scores = evaluation.loc[:, identity_columns].copy()
    scores["label"] = evaluation["label"].to_numpy()
    scores["predicted_label"] = predictions
    scores["mg_probability"] = probabilities[:, mg_index]

    classifier = estimator.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": classifier.coef_[0],
        }
    )
    return scores.reset_index(drop=True), coefficients


def summarize_scores(scores: pd.DataFrame, *, partition: str) -> pd.DataFrame:
    required_columns = {"label", "predicted_label", "mg_probability"}
    missing_columns = sorted(required_columns.difference(scores.columns))
    if missing_columns:
        raise ValueError(f"scores are missing columns: {missing_columns}")
    labels = scores["label"].astype("int64")
    predictions = scores["predicted_label"].astype("int64")
    probabilities = pd.to_numeric(scores["mg_probability"], errors="coerce")
    has_both_labels = labels.nunique() == 2
    row = {
        "partition": partition,
        "sample_count": int(len(scores)),
        "hg_count": int((labels == 0).sum()),
        "mg_count": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_labels and probabilities.notna().all() else np.nan,
    }
    return pd.DataFrame([row])
