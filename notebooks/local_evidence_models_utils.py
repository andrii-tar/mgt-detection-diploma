from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logistic_regression", "gaussian_nb", "gaussian_mixture_nb", "pymc_bayesian_logistic")
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
class GaussianMixtureNBConfig:
    components: int = 2
    reg_covar: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.components, bool) or not isinstance(self.components, Integral) or self.components <= 0:
            raise ValueError("components must be a positive integer")
        if isinstance(self.reg_covar, bool) or not isinstance(self.reg_covar, Real):
            raise ValueError("reg_covar must be a positive finite number")
        if self.reg_covar <= 0 or not np.isfinite(self.reg_covar):
            raise ValueError("reg_covar must be positive and finite")


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


@dataclass
class _GaussianMixtureNBEstimator:
    preprocessing: Pipeline
    mixtures: dict[tuple[str, int], GaussianMixture]
    class_log_priors: dict[int, float]
    config: GaussianMixtureNBConfig


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


def _normalize_log_scores(log_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(log_scores, dtype="float64")
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("log_scores must be a two-dimensional array with at least one class")

    probabilities = np.full(scores.shape, 1.0 / scores.shape[1], dtype="float64")
    finite_rows = np.isfinite(scores).all(axis=1)
    if not finite_rows.any():
        return probabilities

    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        valid_scores = scores[finite_rows]
        shifted = valid_scores - np.max(valid_scores, axis=1, keepdims=True)
        weights = np.exp(shifted)
        totals = weights.sum(axis=1, keepdims=True)
        normalizable = np.isfinite(totals[:, 0]) & (totals[:, 0] > 0.0)
        normalized = np.full_like(weights, 1.0 / scores.shape[1])
        normalized[normalizable] = weights[normalizable] / totals[normalizable]

    probabilities[finite_rows] = normalized
    return probabilities


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
    gaussian_mixture_config: GaussianMixtureNBConfig | None = None,
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
    if model_name == "gaussian_mixture_nb":
        config = gaussian_mixture_config or GaussianMixtureNBConfig()
        labels = fit_windows["label"].to_numpy()
        class_counts = {label: int(np.count_nonzero(labels == label)) for label in (0, 1)}
        if any(count == 0 for count in class_counts.values()):
            raise ValueError("gaussian_mixture_nb requires training rows for both classes")
        preprocessing = Pipeline(
            [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        )
        features = preprocessing.fit_transform(fit_windows[list(normalized_features)])
        mixtures = {}
        for feature_index, feature in enumerate(normalized_features):
            for label in (0, 1):
                mixture = GaussianMixture(
                    n_components=min(config.components, class_counts[label]),
                    covariance_type="full",
                    reg_covar=config.reg_covar,
                    random_state=seed,
                )
                class_features = features[labels == label, feature_index, None]
                if len(class_features) == 1:
                    class_features = np.repeat(class_features, 2, axis=0)
                mixture.fit(class_features)
                mixtures[(feature, label)] = mixture
        total_rows = len(labels)
        class_log_priors = {label: float(np.log(count / total_rows)) for label, count in class_counts.items()}
        estimator = _GaussianMixtureNBEstimator(preprocessing, mixtures, class_log_priors, config)
        return _FittedClassifier(model_name, normalized_features, estimator)
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
    elif fitted.model_name == "gaussian_mixture_nb":
        estimator = fitted.estimator
        features = estimator.preprocessing.transform(windows[list(normalized_features)])
        log_scores = np.empty((len(windows), 2), dtype="float64")
        with np.errstate(over="ignore", invalid="ignore"):
            for class_index, label in enumerate((0, 1)):
                log_scores[:, class_index] = estimator.class_log_priors[label]
                for feature_index, feature in enumerate(normalized_features):
                    log_scores[:, class_index] += estimator.mixtures[(feature, label)].score_samples(
                        features[:, feature_index, None]
                    )
        probabilities = _normalize_log_scores(log_scores)
        scores = probabilities[:, 1]
        lower = upper = None
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


def run_local_evidence_model_comparison(
    windows: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    model_names: tuple[str, ...] | list[str],
    feature_columns: tuple[str, ...] = LOCAL_FEATURE_COLUMNS,
    seed: int = 42,
    bayesian_config: BayesianConfig | None = None,
    gaussian_mixture_config: GaussianMixtureNBConfig | None = None,
    show_progress: bool = True,
    fitted_models: dict[tuple[str, int], _FittedClassifier] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from tqdm import tqdm
    from notebooks.local_evidence_utils import (
        _decision_scores,
        _fit_document_threshold,
        _partition_windows,
        _select_evidence_parameters,
        aggregate_evidence_fraction,
        aggregate_maximum_window,
        evaluate_scores,
    )

    normalized_features = tuple(feature_columns)
    _validate_features(windows, normalized_features)
    requested_models = tuple(model_names)
    if not requested_models:
        raise ValueError("model_names must not be empty")
    unknown_models = [name for name in requested_models if name not in MODEL_NAMES]
    if unknown_models:
        raise ValueError(f"unknown model name: {unknown_models[0]!r}")
    partitioned = {
        partition: _partition_windows(windows, partitions[partition], partition)
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

            fit_kwargs: dict[str, Any] = {}
            if model_name == "pymc_bayesian_logistic":
                fit_kwargs["bayesian_config"] = bayesian_config
            elif model_name == "gaussian_mixture_nb":
                fit_kwargs["gaussian_mixture_config"] = gaussian_mixture_config
            model_key = (model_name, int(window_length))
            if fitted_models is not None and model_key in fitted_models:
                fitted = fitted_models[model_key]
            else:
                fitted = fit_local_evidence_classifier(
                    model_name,
                    train,
                    normalized_features,
                    seed,
                    **fit_kwargs,
                )
                if fitted_models is not None:
                    fitted_models[model_key] = fitted
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

            validation_maximum = aggregate_maximum_window(validation_scored)
            maximum_parameters = _fit_document_threshold(validation_maximum["label"], validation_maximum["mg_score"])
            test_maximum = aggregate_maximum_window(test_scored)
            candidates: list[dict[str, float | str]] = []
            for window_threshold in np.percentile(validation_scored["mg_score"], np.arange(5, 100, 5)):
                validation_fraction = aggregate_evidence_fraction(validation_scored, float(window_threshold))
                document_parameters = _fit_document_threshold(
                    validation_fraction["label"], validation_fraction["mg_score"]
                )
                candidates.append({"window_threshold": float(window_threshold), **document_parameters})
            evidence_parameters = _select_evidence_parameters(candidates)
            test_fraction = aggregate_evidence_fraction(
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
                decision_scores = _decision_scores(predictions["mg_score"], str(parameters["direction"]))
                predictions["prediction"] = (decision_scores >= threshold).astype("int64")
                metrics = evaluate_scores(predictions["label"], decision_scores, threshold)
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
