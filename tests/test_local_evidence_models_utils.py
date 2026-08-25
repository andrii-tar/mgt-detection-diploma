from __future__ import annotations

import pandas as pd

from notebooks.local_evidence_models_utils import (
    MODEL_NAMES,
    BayesianConfig,
    fit_local_evidence_classifier,
    score_local_evidence_classifier,
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


def test_deterministic_scores_have_missing_uncertainty_columns() -> None:
    for model_name in ("logistic_regression", "gaussian_nb"):
        fitted = fit_local_evidence_classifier(model_name, training_windows(), FEATURES, seed=42)

        scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)

        assert scored["mg_score_lower"].isna().all()
        assert scored["mg_score_upper"].isna().all()
