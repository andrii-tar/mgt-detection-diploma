from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import notebooks.utils.local_evidence_models_utils as models_utils
from notebooks.utils.local_evidence_models_utils import (
    MODEL_NAMES,
    BayesianConfig,
    GaussianMixtureNBConfig,
    fit_local_evidence_classifier,
    run_local_evidence_model_comparison,
    score_local_evidence_classifier,
    summarize_pymc_coefficients,
    training_input_fingerprint,
)


FEATURES = ("feature_a", "feature_b")


def comparison_fixture() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    parent_rows = []
    window_rows = []
    text_id = 0
    for partition in ("train", "validation", "test"):
        for label in (0, 1):
            parent_rows.append({"split": "HG" if label == 0 else "MG", "text_id": text_id, "label": label})
            for window_length in (2, 4):
                for start_token in (0, 1):
                    signal = float(label * 3 + window_length / 10 + start_token / 100)
                    window_rows.append(
                        {
                            "split": "HG" if label == 0 else "MG",
                            "text_id": text_id,
                            "label": label,
                            "window_length": window_length,
                            "start_token": start_token,
                            "stop_token": start_token + window_length,
                            "feature_a": signal,
                            "feature_b": -signal,
                        }
                    )
            text_id += 1
    windows = pd.DataFrame(window_rows)
    parents = pd.DataFrame(parent_rows)
    return windows, {
        partition: parents.iloc[index * 2 : (index + 1) * 2].reset_index(drop=True)
        for index, partition in enumerate(("train", "validation", "test"))
    }


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


def _cache_manifest() -> dict[str, object]:
    return {"schema_version": 1, "feature_columns": list(FEATURES), "seed": 42}


def test_model_names_are_stable() -> None:
    assert MODEL_NAMES == (
        "logistic_regression",
        "gaussian_nb",
        "gaussian_mixture_nb",
        "pymc_bayesian_logistic",
    )


def test_training_input_fingerprint_is_stable_but_changes_with_training_inputs_or_partitions() -> None:
    windows, partitions = comparison_fixture()

    first = training_input_fingerprint(windows, partitions)
    reordered = training_input_fingerprint(
        windows.sample(frac=1.0, random_state=7).reset_index(drop=True),
        {name: frame.sample(frac=1.0, random_state=8).reset_index(drop=True) for name, frame in partitions.items()},
    )

    assert first == reordered
    assert first != training_input_fingerprint(windows.assign(feature_a=windows["feature_a"] + 1), partitions)
    changed_partitions = {name: frame.copy() for name, frame in partitions.items()}
    changed_partitions["train"].loc[0, "text_id"] = 999
    assert first != training_input_fingerprint(windows, changed_partitions)


def test_gaussian_mixture_config_has_stable_defaults_and_validates_values() -> None:
    assert GaussianMixtureNBConfig() == GaussianMixtureNBConfig(components=2, reg_covar=1e-6)
    with pytest.raises(ValueError, match="components"):
        GaussianMixtureNBConfig(components=0)
    with pytest.raises(ValueError, match="components"):
        GaussianMixtureNBConfig(components=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reg_covar"):
        GaussianMixtureNBConfig(reg_covar=0.0)
    with pytest.raises(ValueError, match="reg_covar"):
        GaussianMixtureNBConfig(reg_covar=float("inf"))


@pytest.mark.parametrize("reg_covar", [-1.0, float("nan"), "invalid"])
def test_gaussian_mixture_config_rejects_invalid_regularization(reg_covar: object) -> None:
    with pytest.raises(ValueError, match="reg_covar"):
        GaussianMixtureNBConfig(reg_covar=reg_covar)  # type: ignore[arg-type]


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


def test_gaussian_mixture_nb_fits_one_dimensional_mixtures_and_scores() -> None:
    config = GaussianMixtureNBConfig(components=2, reg_covar=1e-5)

    fitted = fit_local_evidence_classifier(
        "gaussian_mixture_nb", training_windows(), FEATURES, seed=42, gaussian_mixture_config=config
    )

    assert fitted.estimator.config == config
    assert set(fitted.estimator.mixtures) == {(feature, label) for feature in FEATURES for label in (0, 1)}
    assert all(isinstance(mixture, GaussianMixture) for mixture in fitted.estimator.mixtures.values())
    assert all(mixture.n_components == 2 for mixture in fitted.estimator.mixtures.values())
    assert all(mixture.covariance_type == "full" for mixture in fitted.estimator.mixtures.values())
    scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)
    assert scored["model"].eq("gaussian_mixture_nb").all()
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert scored["mg_score_lower"].isna().all()
    assert scored["mg_score_upper"].isna().all()


def test_gaussian_mixture_nb_falls_back_to_one_component_per_class() -> None:
    windows = training_windows().iloc[[0, 2]].reset_index(drop=True)

    fitted = fit_local_evidence_classifier(
        "gaussian_mixture_nb", windows, FEATURES, seed=42, gaussian_mixture_config=GaussianMixtureNBConfig(components=2)
    )

    assert all(mixture.n_components == 1 for mixture in fitted.estimator.mixtures.values())


def test_gaussian_mixture_nb_requires_rows_for_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        fit_local_evidence_classifier(
            "gaussian_mixture_nb", training_windows().query("label == 0"), FEATURES, seed=42
        )


def test_gaussian_mixture_nb_repeated_fits_with_same_seed_are_deterministic() -> None:
    windows = training_windows().assign(feature_a=[-2.0, 0.0, 1.0, 3.0])
    first = fit_local_evidence_classifier("gaussian_mixture_nb", windows, FEATURES, seed=42)
    second = fit_local_evidence_classifier("gaussian_mixture_nb", windows, FEATURES, seed=42)

    first_scores = score_local_evidence_classifier(first, windows, FEATURES)["mg_score"]
    second_scores = score_local_evidence_classifier(second, windows, FEATURES)["mg_score"]
    np.testing.assert_array_equal(first_scores, second_scores)


def test_gaussian_mixture_nb_scores_finite_fallback_for_pathological_log_scores() -> None:
    fitted = fit_local_evidence_classifier("gaussian_mixture_nb", training_windows(), FEATURES, seed=42)
    for mixture in fitted.estimator.mixtures.values():
        mixture.score_samples = lambda features: np.array([np.nan, np.inf, -np.inf, 1e308])

    scored = score_local_evidence_classifier(fitted, training_windows(), FEATURES)

    assert np.isfinite(scored["mg_score"]).all()
    assert scored["mg_score"].between(0.0, 1.0).all()
    np.testing.assert_array_equal(scored["mg_score"].to_numpy(), np.full(4, 0.5))


def test_gaussian_mixture_nb_finite_scores_preserve_class_preference() -> None:
    windows = pd.DataFrame(
        {
            "split": ["HG"] * 4 + ["MG"] * 4,
            "text_id": range(8),
            "label": [0] * 4 + [1] * 4,
            "feature_a": [-2.2, -1.8, -2.0, -1.9, 1.8, 2.0, 2.2, 1.9],
            "feature_b": [-2.1, -1.9, -2.2, -1.8, 1.9, 2.1, 1.8, 2.0],
        }
    )
    scored = score_local_evidence_classifier(
        fit_local_evidence_classifier(
            "gaussian_mixture_nb",
            windows,
            FEATURES,
            seed=42,
            gaussian_mixture_config=GaussianMixtureNBConfig(components=2),
        ),
        windows,
        FEATURES,
    )

    probabilities = scored["mg_score"].to_numpy()
    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    assert probabilities[:4].mean() < 0.5
    assert probabilities[4:].mean() > 0.5


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
    all_windows = pd.concat(
        [
            training_windows().assign(partition="train"),
            pd.DataFrame(
                {
                    "split": ["HG", "MG"],
                    "text_id": [4, 5],
                    "label": [0, 1],
                    "feature_a": [10.0, 20.0],
                    "feature_b": [-10.0, -20.0],
                    "partition": ["validation", "test"],
                }
            ),
        ],
        ignore_index=True,
    )
    training_rows = all_windows["partition"].eq("train")
    original_fitted = fit_local_evidence_classifier(
        "gaussian_nb", all_windows, FEATURES, seed=42
    )

    mutated = all_windows.copy()
    mutated.loc[~training_rows, FEATURES] = [
        [-1_000.0, 2_000.0],
        [1_000.0, -2_000.0],
    ]
    mutated_fitted = fit_local_evidence_classifier(
        "gaussian_nb", mutated, FEATURES, seed=42
    )

    original_imputer = original_fitted.estimator.named_steps["impute"]
    mutated_imputer = mutated_fitted.estimator.named_steps["impute"]
    np.testing.assert_array_equal(original_imputer.statistics_, mutated_imputer.statistics_)
    original_scores = score_local_evidence_classifier(
        original_fitted, all_windows.loc[training_rows], FEATURES
    )
    mutated_scores = score_local_evidence_classifier(
        mutated_fitted, mutated.loc[training_rows], FEATURES
    )
    pd.testing.assert_series_equal(
        original_scores["mg_score"].reset_index(drop=True),
        mutated_scores["mg_score"].reset_index(drop=True),
        check_names=False,
    )


def test_partitioned_fit_requires_train_rows() -> None:
    with pytest.raises(ValueError, match="partitioned train_windows must contain train rows"):
        fit_local_evidence_classifier(
            "gaussian_nb",
            training_windows().assign(partition="validation"),
            FEATURES,
            seed=42,
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
    partitioned_windows = pd.concat(
        [
            training_windows().assign(partition="train"),
            training_windows().assign(
                partition="validation",
                feature_a=[100.0, 200.0, 300.0, 400.0],
                feature_b=[-100.0, -200.0, -300.0, -400.0],
            ),
        ],
        ignore_index=True,
    )

    fitted = fit_local_evidence_classifier(
        "pymc_bayesian_logistic",
        partitioned_windows,
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


def test_comparison_evaluates_each_model_window_and_aggregation() -> None:
    windows, partitions = comparison_fixture()

    summaries, predictions, scored_windows, coefficient_summaries = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression", "gaussian_nb", "gaussian_mixture_nb"),
        feature_columns=FEATURES,
        show_progress=False,
    )

    assert len(summaries) == 12
    assert summaries.groupby(["model", "window_length", "aggregation"]).size().eq(1).all()
    assert set(summaries["model"]) == {"logistic_regression", "gaussian_nb", "gaussian_mixture_nb"}
    assert {"model", "window_length", "aggregation", "accuracy", "macro_f1", "document_threshold", "window_threshold", "hg_support", "mg_support"} <= set(summaries.columns)
    assert {"model", "mg_score", "prediction", "document_threshold", "window_threshold"} <= set(predictions.columns)
    assert {"partition", "model", "mg_score"} <= set(scored_windows.columns)
    assert np.isfinite(scored_windows["mg_score"]).all()
    assert scored_windows["mg_score"].between(0.0, 1.0).all()
    mixture_scores = scored_windows.loc[scored_windows["model"].eq("gaussian_mixture_nb")]
    assert {"split", "text_id", "label", "model", "mg_score", "mg_score_lower", "mg_score_upper"} <= set(mixture_scores.columns)
    assert mixture_scores["mg_score"].between(0.0, 1.0).all()
    mixture_summaries = summaries.loc[summaries["model"].eq("gaussian_mixture_nb")]
    assert len(mixture_summaries) == 4
    assert mixture_summaries["window_length"].nunique() == 2
    assert mixture_summaries["aggregation"].nunique() == 2
    assert {"model", "window_length", "feature", "coefficient"} <= set(coefficient_summaries.columns)


def test_comparison_can_skip_detail_frames() -> None:
    windows, partitions = comparison_fixture()

    summaries, predictions, scored_windows, coefficient_summaries = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        collect_details=False,
        show_progress=False,
    )

    assert len(summaries) == 4
    assert predictions.empty
    assert scored_windows.empty
    assert not coefficient_summaries.empty


def test_comparison_can_return_fitted_models_for_reuse() -> None:
    windows, partitions = comparison_fixture()
    fitted_models: dict[tuple[str, int], object] = {}

    run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        fitted_models=fitted_models,
        show_progress=False,
    )

    assert set(fitted_models) == {("logistic_regression", 2), ("logistic_regression", 4)}


def test_comparison_reuses_compatible_cache_without_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    windows, partitions = comparison_fixture()
    registry = {}
    for window_length in (2, 4):
        train = windows.loc[windows["window_length"].eq(window_length)]
        registry[("logistic_regression", window_length)] = fit_local_evidence_classifier(
            "logistic_regression", train, FEATURES, seed=42
        )
    manifest = {
        "feature_columns": list(FEATURES),
        "seed": 42,
        "training_input_fingerprint": training_input_fingerprint(windows, partitions),
    }
    models_utils.save_fitted_models(
        tmp_path,
        registry,
        manifest,
    )

    def fail_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("compatible cache should avoid fitting")

    monkeypatch.setattr(models_utils, "fit_local_evidence_classifier", fail_fit)
    result = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        cache_root=tmp_path,
        cache_manifest=manifest,
        reuse_fitted_models=True,
        trust_fitted_models=True,
        show_progress=False,
    )

    assert len(result) == 4


def test_comparison_fits_when_training_inputs_change_with_same_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows, partitions = comparison_fixture()
    manifest = {**_cache_manifest(), "training_input_fingerprint": training_input_fingerprint(windows, partitions)}
    registry = {}
    for window_length in (2, 4):
        train = windows.loc[windows["window_length"].eq(window_length)]
        registry[("logistic_regression", window_length)] = fit_local_evidence_classifier(
            "logistic_regression", train, FEATURES, seed=42
        )
    models_utils.save_fitted_models(tmp_path, registry, manifest)

    changed_windows = windows.copy()
    changed_windows.loc[changed_windows["text_id"].eq(0), "feature_a"] += 1
    fit_calls = 0
    original_fit = models_utils.fit_local_evidence_classifier

    def count_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(models_utils, "fit_local_evidence_classifier", count_fit)
    run_local_evidence_model_comparison(
        changed_windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        cache_root=tmp_path,
        cache_manifest=manifest,
        reuse_fitted_models=True,
        trust_fitted_models=True,
        show_progress=False,
    )

    assert fit_calls == 2


def test_comparison_fits_when_training_inputs_change_with_fingerprintless_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows, partitions = comparison_fixture()
    manifest = _cache_manifest()
    registry = {}
    for window_length in (2, 4):
        train = windows.loc[windows["window_length"].eq(window_length)]
        registry[("logistic_regression", window_length)] = fit_local_evidence_classifier(
            "logistic_regression", train, FEATURES, seed=42
        )
    models_utils.save_fitted_models(tmp_path, registry, manifest)

    changed_windows = windows.copy()
    changed_windows.loc[changed_windows["text_id"].eq(0), "feature_a"] += 1
    fit_calls = 0
    original_fit = models_utils.fit_local_evidence_classifier

    def count_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(models_utils, "fit_local_evidence_classifier", count_fit)
    run_local_evidence_model_comparison(
        changed_windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        cache_root=tmp_path,
        cache_manifest=manifest,
        reuse_fitted_models=True,
        trust_fitted_models=True,
        show_progress=False,
    )

    assert fit_calls == 2


@pytest.mark.parametrize("cache_manifest", [None, {"feature_columns": ["other"], "seed": 42}])
def test_comparison_fits_when_cache_is_missing_or_incompatible(
    tmp_path: Path, cache_manifest: dict[str, object] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows, partitions = comparison_fixture()
    fit_calls = 0
    original_fit = models_utils.fit_local_evidence_classifier

    def count_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(models_utils, "fit_local_evidence_classifier", count_fit)
    result = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        cache_root=tmp_path,
        cache_manifest=cache_manifest or {"feature_columns": list(FEATURES), "seed": 42},
        reuse_fitted_models=True,
        show_progress=False,
    )

    assert len(result) == 4
    assert fit_calls == 2


def test_comparison_saves_only_after_successful_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows, partitions = comparison_fixture()
    manifest = {"feature_columns": list(FEATURES), "seed": 42}
    save_calls = []
    monkeypatch.setattr(
        models_utils,
        "_save_model_cache",
        lambda root, registry, cache_manifest: save_calls.append((root, registry, cache_manifest)),
    )

    run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        cache_root=tmp_path,
        cache_manifest=manifest,
        save_fitted_models=True,
        show_progress=False,
    )
    assert len(save_calls) == 1
    assert set(save_calls[0][1]) == {("logistic_regression", 2), ("logistic_regression", 4)}

    invalid_partitions = {key: value.copy() for key, value in partitions.items()}
    invalid_partitions["test"]["label"] = 0
    with pytest.raises(ValueError, match="labels inconsistent"):
        run_local_evidence_model_comparison(
            windows,
            invalid_partitions,
            ("logistic_regression",),
            feature_columns=FEATURES,
            cache_root=tmp_path,
            cache_manifest=manifest,
            save_fitted_models=True,
            show_progress=False,
        )
    assert len(save_calls) == 1


def test_cache_rejects_registry_with_manifest_feature_or_configuration_mismatch(tmp_path: Path) -> None:
    fitted = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)
    models_utils.save_fitted_models(
        tmp_path,
        {("logistic_regression", 4): fitted},
        {"feature_columns": ["wrong_a", "wrong_b"], "model_names": ["logistic_regression"], "seed": 42},
    )

    assert models_utils.load_fitted_models(
        tmp_path,
        {"feature_columns": ["wrong_a", "wrong_b"], "model_names": ["logistic_regression"], "seed": 42},
        trust_cache=True,
    ) == {}

    mixture = fit_local_evidence_classifier("gaussian_mixture_nb", training_windows(), FEATURES, seed=42)
    mixture_manifest = {
        "features": list(FEATURES),
        "model_names": ["gaussian_mixture_nb"],
        "gaussian_mixture_config": {"components": 3, "reg_covar": 1e-6},
    }
    models_utils.save_fitted_models(tmp_path, {("gaussian_mixture_nb", 4): mixture}, mixture_manifest)
    assert models_utils.load_fitted_models(tmp_path, mixture_manifest, trust_cache=True) == {}


def test_comparison_fits_only_training_parent_windows() -> None:
    windows, partitions = comparison_fixture()
    original = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression", "gaussian_nb"),
        feature_columns=FEATURES,
        show_progress=False,
    )
    mutated = windows.copy()
    non_training = ~mutated["text_id"].isin(partitions["train"]["text_id"])
    mutated.loc[non_training, "label"] = 1 - mutated.loc[non_training, "label"]
    mutated_partitions = {key: value.copy() for key, value in partitions.items()}
    for partition in ("validation", "test"):
        mutated_partitions[partition]["label"] = 1 - mutated_partitions[partition]["label"]
    changed = run_local_evidence_model_comparison(
        mutated,
        mutated_partitions,
        ("logistic_regression", "gaussian_nb"),
        feature_columns=FEATURES,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(
        original[2].loc[:, ["model", "mg_score"]].reset_index(drop=True),
        changed[2].loc[:, ["model", "mg_score"]].reset_index(drop=True),
    )


def test_comparison_propagates_pymc_intervals_to_both_aggregations() -> None:
    pytest.importorskip("pymc")
    windows, partitions = comparison_fixture()
    config = BayesianConfig(advi_iterations=20, posterior_draws=20, random_seed=42)

    _, predictions, scored_windows, _ = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("pymc_bayesian_logistic",),
        feature_columns=FEATURES,
        bayesian_config=config,
        show_progress=False,
    )

    interval_columns = {"mg_score_lower", "mg_score_upper"}
    assert interval_columns <= set(scored_windows.columns)
    assert interval_columns <= set(predictions.columns)
    assert predictions["aggregation"].isin({"maximum_window", "evidence_fraction"}).all()
    for aggregation in ("maximum_window", "evidence_fraction"):
        aggregated = predictions.loc[predictions["aggregation"].eq(aggregation)]
        assert aggregated["mg_score_lower"].notna().all()
        assert aggregated["mg_score_upper"].notna().all()
        assert (aggregated["mg_score_lower"] <= aggregated["mg_score_upper"]).all()


def test_comparison_threshold_selection_ignores_test_labels() -> None:
    windows, partitions = comparison_fixture()
    original = run_local_evidence_model_comparison(
        windows,
        partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        show_progress=False,
    )
    changed_partitions = {key: value.copy() for key, value in partitions.items()}
    changed_partitions["test"]["label"] = 1 - changed_partitions["test"]["label"]
    changed_windows = windows.copy()
    test_ids = changed_partitions["test"]["text_id"]
    changed_windows.loc[changed_windows["text_id"].isin(test_ids), "label"] = 1 - changed_windows.loc[
        changed_windows["text_id"].isin(test_ids), "label"
    ]
    changed = run_local_evidence_model_comparison(
        changed_windows,
        changed_partitions,
        ("logistic_regression",),
        feature_columns=FEATURES,
        show_progress=False,
    )

    threshold_columns = ["model", "window_length", "aggregation", "document_threshold", "window_threshold", "score_direction", "validation_macro_f1"]
    pd.testing.assert_frame_equal(
        original[0][threshold_columns].reset_index(drop=True),
        changed[0][threshold_columns].reset_index(drop=True),
    )


def test_fitted_classifier_round_trips_through_cache_with_identical_predictions(tmp_path: Path) -> None:
    fitted = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)
    registry = {("logistic_regression", 4): fitted}
    manifest = _cache_manifest()

    models_utils.save_fitted_models(tmp_path, registry, manifest)

    loaded = models_utils.load_fitted_models(tmp_path, manifest, trust_cache=True)
    assert set(loaded) == set(registry)
    pd.testing.assert_frame_equal(
        score_local_evidence_classifier(fitted, training_windows(), FEATURES),
        score_local_evidence_classifier(loaded[("logistic_regression", 4)], training_windows(), FEATURES),
    )


def test_cache_loading_is_denied_without_explicit_trust(tmp_path: Path) -> None:
    fitted = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)
    models_utils.save_fitted_models(tmp_path, {("logistic_regression", 4): fitted}, _cache_manifest())

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest()) == {}


def test_cache_loading_without_trust_does_not_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fitted = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)
    models_utils.save_fitted_models(tmp_path, {("logistic_regression", 4): fitted}, _cache_manifest())
    monkeypatch.setattr(models_utils.pickle, "load", lambda _: pytest.fail("cache must not be unpickled"))

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest()) == {}


def test_incompatible_manifest_returns_empty_registry(tmp_path: Path) -> None:
    fitted = fit_local_evidence_classifier("gaussian_nb", training_windows(), FEATURES, seed=42)
    models_utils.save_fitted_models(tmp_path, {("gaussian_nb", 4): fitted}, _cache_manifest())

    incompatible = {**_cache_manifest(), "seed": 99}
    assert models_utils.load_fitted_models(tmp_path, incompatible, trust_cache=True) == {}


@pytest.mark.parametrize("cache_setup", ["missing", "malformed"])
def test_missing_or_malformed_cache_data_returns_empty_registry(tmp_path: Path, cache_setup: str) -> None:
    if cache_setup == "malformed":
        (tmp_path / "manifest.json").write_text("not json", encoding="utf-8")

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest(), trust_cache=True) == {}


def test_saving_replaces_existing_cache_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = fit_local_evidence_classifier("logistic_regression", training_windows(), FEATURES, seed=42)
    second = fit_local_evidence_classifier("gaussian_nb", training_windows(), FEATURES, seed=42)
    manifest = _cache_manifest()

    models_utils.save_fitted_models(tmp_path, {("logistic_regression", 4): first}, manifest)

    def fail_replace(self: Path, target: str | Path) -> Path:
        raise OSError("simulated cache replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated cache replacement failure"):
        models_utils.save_fitted_models(tmp_path, {("gaussian_nb", 4): second}, manifest)

    loaded = models_utils.load_fitted_models(tmp_path, manifest, trust_cache=True)
    assert set(loaded) == {("logistic_regression", 4)}


@pytest.mark.parametrize("model_name", ["logistic_regression", "gaussian_nb"])
def test_invalid_deterministic_pipeline_cache_returns_empty_registry(tmp_path: Path, model_name: str) -> None:
    fitted = fit_local_evidence_classifier(model_name, training_windows(), FEATURES, seed=42)
    estimator = fitted.estimator
    del estimator.named_steps["impute"].statistics_
    models_utils.save_fitted_models(tmp_path, {(model_name, 4): fitted}, _cache_manifest())

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest(), trust_cache=True) == {}


@pytest.mark.parametrize(
    ("model_name", "attribute"),
    [
        ("logistic_regression", "coef_"),
        ("logistic_regression", "intercept_"),
        ("gaussian_nb", "class_count_"),
        ("gaussian_nb", "class_prior_"),
        ("gaussian_nb", "theta_"),
        ("gaussian_nb", "var_"),
    ],
)
def test_non_finite_deterministic_model_state_invalidates_cache(
    tmp_path: Path, model_name: str, attribute: str
) -> None:
    fitted = fit_local_evidence_classifier(model_name, training_windows(), FEATURES, seed=42)
    learned = getattr(fitted.estimator.named_steps["model"], attribute)
    learned.flat[0] = np.nan
    models_utils.save_fitted_models(tmp_path, {(model_name, 4): fitted}, _cache_manifest())

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest(), trust_cache=True) == {}


def test_invalid_gaussian_mixture_cache_returns_empty_registry(tmp_path: Path) -> None:
    fitted = fit_local_evidence_classifier("gaussian_mixture_nb", training_windows(), FEATURES, seed=42)
    del fitted.estimator.mixtures[(FEATURES[0], 1)]
    fitted.estimator.class_log_priors[0] = float("nan")
    models_utils.save_fitted_models(tmp_path, {("gaussian_mixture_nb", 4): fitted}, _cache_manifest())

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest(), trust_cache=True) == {}


def test_invalid_pymc_cache_returns_empty_registry(tmp_path: Path) -> None:
    preprocessing = Pipeline([("scale", StandardScaler())])
    fitted = models_utils._FittedClassifier(
        "pymc_bayesian_logistic",
        FEATURES,
        models_utils._PymcEstimator(
            model=None,
            approximation=None,
            preprocessing=preprocessing,
            posterior=None,
            config=None,
            seed=None,
        ),
    )
    models_utils.save_fitted_models(tmp_path, {("pymc_bayesian_logistic", 4): fitted}, _cache_manifest())

    assert models_utils.load_fitted_models(tmp_path, _cache_manifest(), trust_cache=True) == {}
