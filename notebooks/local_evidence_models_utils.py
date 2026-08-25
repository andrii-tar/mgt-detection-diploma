from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = ("logistic_regression", "gaussian_nb", "pymc_bayesian_logistic")


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
    if "label" not in train_windows.columns:
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
        features = preprocessing.fit_transform(train_windows[list(normalized_features)])
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
            pymc.Bernoulli("label", p=probability, observed=train_windows["label"].to_numpy())
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
    estimator.fit(train_windows[list(normalized_features)], train_windows["label"])
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
