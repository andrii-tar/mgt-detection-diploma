from __future__ import annotations

import builtins

import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

import notebooks.local_evidence_models_utils as models_utils
from notebooks.local_evidence_models_utils import (
    MODEL_NAMES,
    BayesianConfig,
    fit_local_evidence_classifier,
    score_local_evidence_classifier,
    summarize_pymc_coefficients,
)


FEATURES = ("feature_a", "feature_b")


def training_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["HG", "HG", "MG", "MG"],
            "text_id": [0, 1, 2, 3],
            "label": [0, 0, 1, 1],
            "feature_a": [0.0, 0.2, 1.0, 1.2],
            "feature_b": [1.0, 1.2, 0.0, 0.2],
        }
    )


def test_model_names_are_stable() -> None:
    assert MODEL_NAMES == ("logistic_regression", "gaussian_nb", "pymc_bayesian_logistic")


def test_bayesian_config_has_stable_defaults() -> None:
    assert BayesianConfig() == BayesianConfig(
        prior_scale=1.0,
        intercept_scale=2.5,
        advi_iterations=10_000,
        posterior_draws=1_000,
        random_seed=42,
        credible_interval=(0.025, 0.975),
    )


def test_logistic_regression_scores_have_common_schema_and_bounded_probabilities() -> None:
    fitted = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)

    scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)

    assert {"split", "text_id", "label", *FEATURES} <= set(scored.columns)
    assert {"model", "mg_score", "mg_score_lower", "mg_score_upper"} <= set(scored.columns)
    assert scored["model"].eq("logistic_regression").all()
    assert scored["mg_score"].between(0.0, 1.0).all()


def test_gaussian_nb_scores_have_common_schema_and_bounded_probabilities() -> None:
    fitted = fit_local_evidence_classifier("gaussian_nb", training_windows(), FEATURES, seed=42)

    scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)

    assert {"model", "mg_score", "mg_score_lower", "mg_score_upper"} <= set(scored.columns)
    assert scored["model"].eq("gaussian_nb").all()
    assert scored["mg_score"].between(0.0, 1.0).all()


def test_gaussian_nb_preserves_signed_continuous_features() -> None:
    windows = training_windows()
    windows["feature_a"] = [-2.5, -1.0, 0.5, 2.0]
    windows["feature_b"] = [1.5, -0.5, -2.0, 0.25]

    fitted = fit_local_evidence_classifier("gaussian_nb", windows, FEATURES, seed=42)

    assert isinstance(fitted.estimator.named_steps["model"], GaussianNB)
    assert list(fitted.estimator.named_steps) == ["impute", "model"]
    assert "scale" not in fitted.estimator.named_steps
    imputed = fitted.estimator.named_steps["impute"].transform(windows[list(FEATURES)])
    assert imputed.min() < 0.0


def test_gaussian_nb_preprocessing_ignores_mutated_non_training_rows() -> None:
    train = training_windows()
    non_training = pd.DataFrame(
        {
            "feature_a": [10.0, 20.0],
            "feature_b": [-10.0, -20.0],
        }
    )
    fitted = fit_local_evidence_classifier("gaussian_nb", train, FEATURES, seed=42)
    imputer = fitted.estimator.named_steps["impute"]
    transformed_before = imputer.transform(train[list(FEATURES)])
    statistics_before = imputer.statistics_.copy()

    non_training.loc[:, "feature_a"] = [-1_000.0, 1_000.0]
    non_training.loc[:, "feature_b"] = [2_000.0, -2_000.0]

    transformed_after = imputer.transform(train[list(FEATURES)])
    pd.testing.assert_frame_equal(
        pd.DataFrame(transformed_before, columns=FEATURES),
        pd.DataFrame(transformed_after, columns=FEATURES),
    )
    pd.testing.assert_series_equal(
        pd.Series(statistics_before, index=FEATURES),
        train[list(FEATURES)].median(),
        check_names=False,
    )


def test_deterministic_scores_have_missing_uncertainty_columns() -> None:
    for model_name in ("logistic_regression", "gaussian_nb"):
        fitted = fit_local_evidence_classifier(model_name, training_windows(), FEATURES, seed=42)

        scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)

        assert scored["mg_score_lower"].isna().all()
        assert scored["mg_score_upper"].isna().all()


def test_pymc_is_imported_only_when_bayesian_path_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def track_pymc_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pymc":
            imported.append(name)
            raise ModuleNotFoundError("No module named 'pymc'", name="pymc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_pymc_import)
    fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)

    assert imported == []
    with pytest.raises(ImportError, match=r"uv sync --extra bayesian"):
        fit_local_evidence_classifier("pymc_bayesian_logistic", training_windows(), FEATURES, seed=42)
    assert imported == ["pymc"]


def test_pymc_adapter_scores_intervals_and_summarizes_coefficients() -> None:
    pytest.importorskip("pymc")
    config = BayesianConfig(advi_iterations=50, posterior_draws=40, random_seed=42)

    fitted = fit_local_evidence_classifier(
        "pymc_bayesian_logistic",
        training_windows(),
        FEATURES,
        seed=42,
        bayesian_config=config,
    )
    scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)
    summary = summarize_pymc_coefficients(fitted)

    assert scored["model"].eq("pymc_bayesian_logistic").all()
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert scored["mg_score_lower"].between(0.0, 1.0).all()
    assert scored["mg_score_upper"].between(0.0, 1.0).all()
    assert (scored["mg_score_lower"] <= scored["mg_score"]).all()
    assert (scored["mg_score"] <= scored["mg_score_upper"]).all()
    assert list(summary["feature"]) == list(FEATURES)
    assert {
        "feature",
        "posterior_mean",
        "lower",
        "upper",
        "posterior_probability_positive",
    } <= set(summary.columns)
    assert summary[["posterior_mean", "lower", "upper", "posterior_probability_positive"]].apply(
        lambda column: column.map(pd.api.types.is_number).all()
    ).all()
    assert (summary["lower"] <= summary["posterior_mean"]).all()
    assert (summary["posterior_mean"] <= summary["upper"]).all()
