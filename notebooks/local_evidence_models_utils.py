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


def _load_pymc() -> Any:
    try:
        import pymc
    except ModuleNotFoundError as error:
        if error.name != "pymc":
            raise
        raise ImportError("PyMC is optional; install text-detection[bayesian] to use the Bayesian adapter") from error
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
    del bayesian_config
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model name: {model_name!r}")
    if model_name == "pymc_bayesian_logistic":
        _load_pymc()
        raise NotImplementedError("PyMC Bayesian logistic regression is deferred to Task 2")

    normalized_features = tuple(feature_columns)
    _validate_features(train_windows, normalized_features)
    if "label" not in train_windows.columns:
        raise ValueError("train_windows must contain a label column")
    estimator = _deterministic_estimator(model_name, seed)
    estimator.fit(train_windows[list(normalized_features)], train_windows["label"])
    return _FittedClassifier(model_name, normalized_features, estimator)


def score_local_evidence_classifier(
    fitted: _FittedClassifier,
    windows: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    posterior_draws: int | None = None,
) -> pd.DataFrame:
    del posterior_draws
    normalized_features = tuple(feature_columns)
    _validate_features(windows, normalized_features)
    if normalized_features != fitted.feature_columns:
        raise ValueError("feature columns do not match fitted classifier")
    scores = np.asarray(fitted.estimator.predict_proba(windows[list(normalized_features)])[:, 1], dtype="float64")
    scored = windows.copy()
    scored["model"] = fitted.model_name
    scored["mg_score"] = np.clip(scores, 0.0, 1.0)
    scored["mg_score_lower"] = np.nan
    scored["mg_score_upper"] = np.nan
    return scored
