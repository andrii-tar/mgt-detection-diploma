from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from notebooks.logprob_metrics_distributions_utils import load_metrics


SCALAR_FEATURE_COLUMNS = (
    "prompt_mu_score",
    "prompt_sigma_score",
    "perplexity",
    "d_score_top20",
    "d_score_top20_coverage",
    "text_len_symbols",
    "text_len_words",
    "text_len_sentences",
)
LABELS = {"HG": 0, "MG": 1}


def available_scalar_features(metrics_df: pd.DataFrame) -> list[str]:
    return [column for column in SCALAR_FEATURE_COLUMNS if column in metrics_df.columns]


def select_model_by_coverage(
    metrics_df: pd.DataFrame,
    dataset: str,
    model: str | None = None,
) -> tuple[str, pd.DataFrame]:
    dataset_df = metrics_df[metrics_df["dataset"] == dataset]
    if dataset_df.empty:
        raise ValueError(f"No metric rows found for dataset={dataset!r}")
    labeled_df = dataset_df[dataset_df["split"].isin(LABELS)]
    coverage = (
        labeled_df.groupby(["model", "split"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["HG", "MG"], fill_value=0)
        .rename(columns={"HG": "hg_rows", "MG": "mg_rows"})
        .reset_index()
    )
    coverage["balanced_rows"] = coverage[["hg_rows", "mg_rows"]].min(axis=1)
    coverage["total_rows"] = coverage[["hg_rows", "mg_rows"]].sum(axis=1)
    coverage = coverage.sort_values(
        ["balanced_rows", "total_rows", "model"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    if coverage.empty or coverage.loc[0, "balanced_rows"] == 0:
        raise ValueError(f"No HG/MG metric coverage found for dataset={dataset!r}")
    if model is None:
        return str(coverage.loc[0, "model"]), coverage
    if model not in set(coverage["model"]):
        raise ValueError(f"No HG/MG metric coverage found for dataset={dataset!r}, model={model!r}")
    if int(coverage.loc[coverage["model"] == model, "balanced_rows"].iat[0]) == 0:
        raise ValueError(f"No HG/MG metric coverage found for dataset={dataset!r}, model={model!r}")
    return model, coverage


def build_canonical_cohort(metrics_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in metrics_df.columns]
    if missing_columns:
        raise ValueError(f"missing feature columns: {missing_columns}")
    cohort = metrics_df[metrics_df["split"].isin(LABELS)].copy()
    cohort["label"] = cohort["split"].map(LABELS).astype("int64")
    numeric_features = cohort[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric_features.to_numpy(dtype="float64")).all(axis=1)
    cohort.loc[:, feature_columns] = numeric_features
    return cohort.loc[finite_mask].reset_index(drop=True)


def load_baseline_cohort(
    processed_root: Path,
    *,
    dataset: str = "kaggle_train",
    model: str | None = None,
) -> tuple[pd.DataFrame, list[str], str, pd.DataFrame]:
    metrics_df = load_metrics(processed_root)
    selected_model, coverage = select_model_by_coverage(metrics_df, dataset, model)
    selected_df = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["model"] == selected_model)].copy()
    feature_columns = available_scalar_features(selected_df)
    if not feature_columns:
        raise ValueError(f"No scalar metric columns available for dataset={dataset!r}, model={selected_model!r}")
    cohort = build_canonical_cohort(selected_df, feature_columns)
    if cohort["label"].nunique() != 2:
        raise ValueError(f"No complete HG/MG scalar-metric cohort found for dataset={dataset!r}, model={selected_model!r}")
    return cohort, feature_columns, selected_model, coverage


def split_cohort(
    cohort: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("train, validation, and test fractions must be greater than 0")
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("train, validation, and test fractions must sum to 1.0")
    if cohort["label"].nunique() != 2:
        raise ValueError("cohort must contain HG and MG labels")
    minimum_class_rows = ceil(1 / min(train_fraction, validation_fraction, test_fraction))
    if cohort["label"].value_counts().min() < minimum_class_rows:
        raise ValueError(f"each label requires at least {minimum_class_rows} rows for the requested stratified partitions")
    development, test = train_test_split(
        cohort,
        test_size=test_fraction,
        random_state=seed,
        stratify=cohort["label"],
    )
    validation_share = validation_fraction / (train_fraction + validation_fraction)
    train, validation = train_test_split(
        development,
        test_size=validation_share,
        random_state=seed,
        stratify=development["label"],
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def fit_single_feature_threshold(y_true: pd.Series, values: pd.Series) -> dict[str, float | str]:
    raw_values = pd.to_numeric(values, errors="raise").to_numpy(dtype="float64")
    labels = pd.Series(y_true).to_numpy(dtype="int64")
    candidates: list[dict[str, float | str]] = []
    for direction, mg_score in (("higher_is_mg", raw_values), ("lower_is_mg", -raw_values)):
        for threshold in np.unique(mg_score):
            macro_f1 = f1_score(labels, mg_score >= threshold, average="macro", zero_division=0)
            candidates.append(
                {
                    "direction": direction,
                    "threshold": float(threshold),
                    "macro_f1": float(macro_f1),
                }
            )
    return max(candidates, key=lambda candidate: (float(candidate["macro_f1"]), -float(candidate["threshold"])))


def evaluate_scores(y_true: pd.Series, mg_score: pd.Series, threshold: float) -> dict[str, float | int]:
    labels = pd.Series(y_true).to_numpy(dtype="int64")
    scores = pd.Series(mg_score).to_numpy(dtype="float64")
    predictions = (scores >= threshold).astype("int64")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "hg_precision": float(precision[0]),
        "hg_recall": float(recall[0]),
        "hg_f1": float(f1[0]),
        "hg_support": int(support[0]),
        "mg_precision": float(precision[1]),
        "mg_recall": float(recall[1]),
        "mg_f1": float(f1[1]),
        "mg_support": int(support[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _single_feature_score(values: pd.Series, direction: str) -> pd.Series:
    if direction == "higher_is_mg":
        return values.astype("float64")
    if direction == "lower_is_mg":
        return -values.astype("float64")
    raise ValueError(f"unknown score direction: {direction!r}")


def run_baseline_experiments(
    partitions: dict[str, pd.DataFrame],
    feature_columns: list[str],
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation = partitions["validation"]
    test = partitions["test"]
    for feature in feature_columns:
        fitted = fit_single_feature_threshold(validation["label"], validation[feature])
        test_scores = _single_feature_score(test[feature], str(fitted["direction"]))
        metrics = evaluate_scores(test["label"], test_scores, float(fitted["threshold"]))
        experiment = f"threshold:{feature}"
        summaries.append(
            {
                "experiment": experiment,
                "classifier": "single_feature_threshold",
                "feature_set": feature,
                "threshold": float(fitted["threshold"]),
                "score_direction": fitted["direction"],
                "validation_macro_f1": float(fitted["macro_f1"]),
                **metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "experiment": experiment,
                    "classifier": "single_feature_threshold",
                    "label": test["label"].to_numpy(),
                    "mg_score": test_scores.to_numpy(),
                    "prediction": (test_scores >= float(fitted["threshold"])).astype("int64"),
                }
            )
        )
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(partitions["train"][feature_columns], partitions["train"]["label"])
    logistic_scores = pd.Series(model.predict_proba(test[feature_columns])[:, 1], index=test.index)
    logistic_metrics = evaluate_scores(test["label"], logistic_scores, 0.5)
    summaries.append(
        {
            "experiment": "logistic_regression:all",
            "classifier": "logistic_regression",
            "feature_set": ", ".join(feature_columns),
            "threshold": 0.5,
            "score_direction": "higher_is_mg",
            "validation_macro_f1": np.nan,
            **logistic_metrics,
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "experiment": "logistic_regression:all",
                "classifier": "logistic_regression",
                "label": test["label"].to_numpy(),
                "mg_score": logistic_scores.to_numpy(),
                "prediction": (logistic_scores >= 0.5).astype("int64"),
            }
        )
    )
    return pd.DataFrame(summaries), pd.concat(prediction_frames, ignore_index=True)


def plot_confusion_matrices(predictions: pd.DataFrame) -> None:
    experiments = sorted(predictions["experiment"].unique())
    figure, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 3.5), squeeze=False)
    for axis, experiment in zip(axes[0], experiments):
        experiment_df = predictions[predictions["experiment"] == experiment]
        matrix = confusion_matrix(experiment_df["label"], experiment_df["prediction"], labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        for row, column in np.ndindex(matrix.shape):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(title=experiment, xlabel="Predicted label", ylabel="True label", xticks=[0, 1], yticks=[0, 1])
        axis.set_xticklabels(["HG", "MG"])
        axis.set_yticklabels(["HG", "MG"])
    figure.colorbar(image, ax=axes.ravel().tolist(), label="count")
    figure.suptitle("Test confusion matrices", y=1.04)
    plt.show()


def plot_score_curves(predictions: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for experiment, experiment_df in predictions.groupby("experiment", sort=True):
        fpr, tpr, _ = roc_curve(experiment_df["label"], experiment_df["mg_score"])
        precision, recall, _ = precision_recall_curve(experiment_df["label"], experiment_df["mg_score"])
        axes[0].plot(fpr, tpr, label=experiment)
        axes[1].plot(recall, precision, label=experiment)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set(title="Test ROC curve (MG positive)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Test precision-recall curve (MG positive)", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.legend(fontsize="small")
    plt.show()
