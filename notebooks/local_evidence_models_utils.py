from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logistic_regression", "gaussian_nb", "pymc_bayesian_logistic")
LOCAL_FEATURE_COLUMNS = tuple(
    f"{metric}__{statistic}"
    for metric in (
        "observed_logprob",
        "mean_candidate_logprob_delta",
        "mean_candidate_similarity",
        "top20_coverage",
    )
    for statistic in ("mean", "std", "q10", "q50", "q90", "min", "max", "slope", "finite_fraction")
)


@dataclass(frozen=True)
class BayesianConfig:
    prior_scale: float = 1.0
    intercept_scale: float = 2.5
    advi_iterations: int = 10_000
    posterior_draws: int = 1_000
    random_seed: int = 42
    credible_interval: tuple[float, float] = (0.025, 0.975)


@dataclass(frozen=True)
class _FittedClassifier:
    model_name: str
    feature_columns: tuple[str, ...]
    estimator: Any


@dataclass
class _PymcEstimator:
    model: Any
    approximation: Any
    preprocessing: Pipeline
    posterior: Any
    config: BayesianConfig
    seed: int


def _load_pymc() -> Any:
    try:
        import pymc
    except ModuleNotFoundError as error:
        if error.name != "pymc":
            raise
        raise ImportError("PyMC is optional; run uv sync --extra bayesian to use the Bayesian adapter") from error
    return pymc


def _validate_features(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    missing_features = [feature for feature in feature_columns if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")


def _deterministic_estimator(model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic_regression":
        steps = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    elif model_name == "gaussian_nb":
        steps = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", GaussianNB()),
        ]
    else:
        raise ValueError(f"unsupported deterministic model: {model_name!r}")
    return Pipeline(steps)


def fit_local_evidence_classifier(
    model_name: str,
    train_windows: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    seed: int,
    bayesian_config: BayesianConfig | None = None,
) -> _FittedClassifier:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model name: {model_name!r}")

    normalized_features = tuple(feature_columns)
    _validate_features(train_windows, normalized_features)
    fit_windows = train_windows
    if "partition" in train_windows.columns:
        train_rows = train_windows["partition"].eq("train")
        if not train_rows.any():
            raise ValueError("partitioned train_windows must contain train rows")
        fit_windows = train_windows.loc[train_rows]
    if "label" not in fit_windows.columns:
        raise ValueError("train_windows must contain a label column")
    if model_name == "pymc_bayesian_logistic":
        pymc = _load_pymc()
        config = bayesian_config or BayesianConfig()
        preprocessing = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        features = preprocessing.fit_transform(fit_windows[list(normalized_features)])
        bayesian_seed = config.random_seed
        model = pymc.Model()
        with model:
            coefficients = pymc.Normal(
                "coefficients",
                mu=0.0,
                sigma=config.prior_scale,
                shape=features.shape[1],
            )
            intercept = pymc.Normal("intercept", mu=0.0, sigma=config.intercept_scale)
            probability = pymc.Deterministic(
                "mg_probability",
                pymc.math.sigmoid(intercept + pymc.math.dot(features, coefficients)),
            )
            pymc.Bernoulli("label", p=probability, observed=fit_windows["label"].to_numpy())
            approximation = pymc.fit(
                n=config.advi_iterations,
                random_seed=bayesian_seed,
                progressbar=False,
            )
            posterior = approximation.sample(
                draws=config.posterior_draws,
                random_seed=bayesian_seed,
                return_inferencedata=True,
            )
        estimator = _PymcEstimator(model, approximation, preprocessing, posterior, config, bayesian_seed)
        return _FittedClassifier(model_name, normalized_features, estimator)

    estimator = _deterministic_estimator(model_name, seed)
    estimator.fit(fit_windows[list(normalized_features)], fit_windows["label"])
    return _FittedClassifier(model_name, normalized_features, estimator)


def score_local_evidence_classifier(
    fitted: _FittedClassifier,
    windows: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    posterior_draws: int | None = None,
) -> pd.DataFrame:
    normalized_features = tuple(feature_columns)
    _validate_features(windows, normalized_features)
    if normalized_features != fitted.feature_columns:
        raise ValueError("feature columns do not match fitted classifier")
    if fitted.model_name == "pymc_bayesian_logistic":
        estimator = fitted.estimator
        features = estimator.preprocessing.transform(windows[list(normalized_features)])
        posterior = estimator.posterior
        if posterior_draws is not None and posterior_draws != estimator.config.posterior_draws:
            posterior = estimator.approximation.sample(
                draws=posterior_draws,
                random_seed=estimator.seed,
                return_inferencedata=True,
            )
        coefficients = np.asarray(posterior.posterior["coefficients"]).reshape(-1, features.shape[1])
        intercept = np.asarray(posterior.posterior["intercept"]).reshape(-1)
        logits = intercept[:, None] + coefficients @ features.T
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
        scores = np.asarray(probabilities.mean(axis=0), dtype="float64")
        lower, upper = np.quantile(probabilities, estimator.config.credible_interval, axis=0)
    else:
        scores = np.asarray(fitted.estimator.predict_proba(windows[list(normalized_features)])[:, 1], dtype="float64")
        lower = upper = None
    scored = windows.copy()
    scored["model"] = fitted.model_name
    scored["mg_score"] = np.clip(scores, 0.0, 1.0)
    scored["mg_score_lower"] = np.nan if lower is None else np.clip(lower, 0.0, 1.0)
    scored["mg_score_upper"] = np.nan if upper is None else np.clip(upper, 0.0, 1.0)
    return scored


def summarize_pymc_coefficients(fitted: _FittedClassifier) -> pd.DataFrame:
    if fitted.model_name != "pymc_bayesian_logistic":
        raise ValueError("coefficient summaries are only available for the PyMC model")
    posterior = fitted.estimator.posterior.posterior["coefficients"]
    samples = np.asarray(posterior).reshape(-1, len(fitted.feature_columns))
    lower, upper = np.quantile(samples, fitted.estimator.config.credible_interval, axis=0)
    return pd.DataFrame(
        {
            "feature": fitted.feature_columns,
            "posterior_mean": samples.mean(axis=0),
            "lower": lower,
            "upper": upper,
            "posterior_probability_positive": (samples > 0).mean(axis=0),
        }
    )


def _summarize_deterministic_coefficients(
    fitted: _FittedClassifier,
    window_length: int,
) -> pd.DataFrame:
    if fitted.model_name != "logistic_regression":
        return pd.DataFrame()
    coefficients = np.asarray(fitted.estimator.named_steps["model"].coef_[0], dtype="float64")
    return pd.DataFrame(
        {
            "model": fitted.model_name,
            "window_length": window_length,
            "feature": fitted.feature_columns,
            "coefficient": coefficients,
            "posterior_mean": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "posterior_probability_positive": np.nan,
        }
    )


def _partition_comparison_windows(
    windows: pd.DataFrame,
    parents: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    parent_labels = parents[["split", "text_id", "label"]].rename(columns={"label": "parent_label"})
    filtered = windows.merge(parent_labels, on=["split", "text_id"], how="inner", validate="many_to_one")
    if not filtered["label"].eq(filtered["parent_label"]).all():
        raise ValueError(f"{partition} partition has parent labels inconsistent with its windows")
    filtered = filtered.drop(columns="parent_label")
    if set(filtered["label"]) != {0, 1}:
        raise ValueError(f"{partition} partition must have window rows for both labels")
    return filtered.assign(partition=partition)


def _aggregate_maximum_window(scored_windows: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["split", "text_id", "label", "window_length"]
    selected = (
        scored_windows.sort_values(
            [*group_columns, "mg_score", "start_token"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(group_columns, as_index=False, sort=False)
        .head(1)
        .copy()
    )
    return selected.rename(columns={"start_token": "selected_start_token", "stop_token": "selected_stop_token"})


def _aggregate_evidence_fraction(scored_windows: pd.DataFrame, window_threshold: float) -> pd.DataFrame:
    group_columns = ["split", "text_id", "label", "window_length"]
    maximum = _aggregate_maximum_window(scored_windows)
    fractions = (
        scored_windows.assign(is_evidence=scored_windows["mg_score"] >= window_threshold)
        .groupby(group_columns, as_index=False)["is_evidence"]
        .mean()
        .rename(columns={"is_evidence": "mg_score"})
    )
    return maximum.drop(columns="mg_score").merge(fractions, on=group_columns, validate="one_to_one")


def _fit_comparison_threshold(y_true: pd.Series, mg_score: pd.Series) -> dict[str, float | str]:
    values = pd.to_numeric(mg_score, errors="raise").to_numpy(dtype="float64")
    labels = pd.Series(y_true).to_numpy(dtype="int64")
    candidates = []
    for direction, decision_scores in (("higher_is_mg", values), ("lower_is_mg", -values)):
        for threshold in np.unique(decision_scores):
            candidates.append(
                {
                    "direction": direction,
                    "threshold": float(threshold),
                    "macro_f1": float(f1_score(labels, decision_scores >= threshold, average="macro", zero_division=0)),
                }
            )
    return max(candidates, key=lambda candidate: (float(candidate["macro_f1"]), -float(candidate["threshold"])))


def _select_comparison_parameters(candidates: list[dict[str, float | str]]) -> dict[str, float | str]:
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["macro_f1"]),
            float(candidate["window_threshold"]),
            float(candidate["threshold"]),
        ),
    )


def _evaluate_comparison_scores(
    y_true: pd.Series,
    decision_scores: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    labels = pd.Series(y_true).to_numpy(dtype="int64")
    scores = pd.Series(decision_scores).to_numpy(dtype="float64")
    predictions = scores >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
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


def run_local_evidence_model_comparison(
    windows: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    model_names: tuple[str, ...] | list[str],
    feature_columns: tuple[str, ...] = LOCAL_FEATURE_COLUMNS,
    seed: int = 42,
    bayesian_config: BayesianConfig | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from tqdm import tqdm

    normalized_features = tuple(feature_columns)
    _validate_features(windows, normalized_features)
    requested_models = tuple(model_names)
    if not requested_models:
        raise ValueError("model_names must not be empty")
    unknown_models = [name for name in requested_models if name not in MODEL_NAMES]
    if unknown_models:
        raise ValueError(f"unknown model name: {unknown_models[0]!r}")
    partitioned = {
        partition: _partition_comparison_windows(windows, partitions[partition], partition)
        for partition in ("train", "validation", "test")
    }
    lengths = sorted(windows["window_length"].unique())
    if show_progress:
        lengths = tqdm(lengths, desc="Local evidence model comparison", unit="length")

    summaries: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    for model_name in requested_models:
        for window_length in lengths:
            train = partitioned["train"].loc[partitioned["train"]["window_length"] == window_length]
            validation = partitioned["validation"].loc[partitioned["validation"]["window_length"] == window_length]
            test = partitioned["test"].loc[partitioned["test"]["window_length"] == window_length]
            if any(set(frame["label"]) != {0, 1} for frame in (train, validation, test)):
                raise ValueError(f"window_length={window_length} must have rows for both labels in every partition")

            fitted = fit_local_evidence_classifier(
                model_name,
                train,
                normalized_features,
                seed,
                bayesian_config,
            )
            coefficient_summary = _summarize_deterministic_coefficients(fitted, window_length)
            if model_name == "pymc_bayesian_logistic":
                coefficient_summary = summarize_pymc_coefficients(fitted).assign(
                    model=model_name,
                    window_length=window_length,
                    coefficient=np.nan,
                )[
                    [
                        "model",
                        "window_length",
                        "feature",
                        "coefficient",
                        "posterior_mean",
                        "lower",
                        "upper",
                        "posterior_probability_positive",
                    ]
                ]
            if not coefficient_summary.empty:
                coefficient_frames.append(coefficient_summary)

            validation_scored = score_local_evidence_classifier(fitted, validation, normalized_features)
            test_scored = score_local_evidence_classifier(fitted, test, normalized_features)
            scored_frames.extend((validation_scored, test_scored))

            validation_maximum = _aggregate_maximum_window(validation_scored)
            maximum_parameters = _fit_comparison_threshold(validation_maximum["label"], validation_maximum["mg_score"])
            test_maximum = _aggregate_maximum_window(test_scored)
            candidates: list[dict[str, float | str]] = []
            for window_threshold in np.percentile(validation_scored["mg_score"], np.arange(5, 100, 5)):
                validation_fraction = _aggregate_evidence_fraction(validation_scored, float(window_threshold))
                document_parameters = _fit_comparison_threshold(
                    validation_fraction["label"], validation_fraction["mg_score"]
                )
                candidates.append({"window_threshold": float(window_threshold), **document_parameters})
            evidence_parameters = _select_comparison_parameters(candidates)
            test_fraction = _aggregate_evidence_fraction(
                test_scored, float(evidence_parameters["window_threshold"])
            )

            for aggregation, document_scores, parameters in (
                ("maximum_window", test_maximum, {"window_threshold": np.nan, **maximum_parameters}),
                ("evidence_fraction", test_fraction, evidence_parameters),
            ):
                threshold = float(parameters["threshold"])
                predictions = document_scores.copy()
                predictions["aggregation"] = aggregation
                predictions["window_threshold"] = parameters["window_threshold"]
                predictions["document_threshold"] = threshold
                predictions["score_direction"] = parameters["direction"]
                decision_scores = predictions["mg_score"] if parameters["direction"] == "higher_is_mg" else -predictions["mg_score"]
                predictions["prediction"] = (decision_scores >= threshold).astype("int64")
                metrics = _evaluate_comparison_scores(predictions["label"], decision_scores, threshold)
                summaries.append(
                    {
                        "model": model_name,
                        "window_length": window_length,
                        "aggregation": aggregation,
                        "window_threshold": parameters["window_threshold"],
                        "document_threshold": threshold,
                        "score_direction": parameters["direction"],
                        "validation_macro_f1": float(parameters["macro_f1"]),
                        **metrics,
                    }
                )
                prediction_frames.append(predictions)

    coefficient_columns = [
        "model",
        "window_length",
        "feature",
        "coefficient",
        "posterior_mean",
        "lower",
        "upper",
        "posterior_probability_positive",
    ]
    coefficient_summaries = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(columns=coefficient_columns)
    )
    return (
        pd.DataFrame(summaries),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(scored_frames, ignore_index=True),
        coefficient_summaries,
    )
