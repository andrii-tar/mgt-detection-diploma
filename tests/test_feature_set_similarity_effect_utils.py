from __future__ import annotations

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from notebooks.utils.feature_set_similarity_effect_utils import (
    DEFAULT_MODEL_SIZES_BILLIONS,
    aggregate_feature_set_deltas,
    attach_model_sizes,
    matched_feature_set_deltas,
)


def test_matched_feature_set_deltas_uses_only_shared_conditions() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "small", "model_family": "lr", "training_regime": "Mixed", "training_window": 32, "evaluation_window": 32, "dataset": "d1", "partition": "test", "feature_set": "simple_similarity", "macro_f1": 0.60},
        {"scorer_llm": "small", "model_family": "lr", "training_regime": "Mixed", "training_window": 32, "evaluation_window": 32, "dataset": "d1", "partition": "test", "feature_set": "extended", "macro_f1": 0.70},
        {"scorer_llm": "small", "model_family": "lr", "training_regime": "Mixed", "training_window": 64, "evaluation_window": 64, "dataset": "d1", "partition": "test", "feature_set": "simple_similarity", "macro_f1": 0.80},
    ])

    result = matched_feature_set_deltas(
        metrics, baseline_feature_set="simple_similarity", comparison_feature_sets=["extended"],
    )

    row = result.iloc[0]
    assert row["baseline_score"] == pytest.approx(0.60)
    assert row["comparison_score"] == pytest.approx(0.70)
    assert row["delta"] == pytest.approx(0.10)


def test_aggregate_and_model_size_metadata_keep_unknown_models() -> None:
    deltas = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "extended", "delta": 0.10},
        {"scorer_llm": "small", "feature_set": "extended", "delta": 0.20},
        {"scorer_llm": "future_llm", "feature_set": "extended", "delta": -0.05},
    ])

    summary = attach_model_sizes(
        aggregate_feature_set_deltas(deltas), model_sizes_billions={"small": 0.36},
    )

    assert summary.loc[summary["scorer_llm"].eq("small"), "mean_delta"].iat[0] == pytest.approx(0.15)
    assert bool(summary.loc[summary["scorer_llm"].eq("future_llm"), "size_known"].iat[0]) is False


def test_default_model_sizes_include_littlelamb() -> None:
    assert DEFAULT_MODEL_SIZES_BILLIONS["littlelamb"] == pytest.approx(0.3)


def test_feature_set_comparison_uses_one_color_per_feature_set() -> None:
    deltas = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "extended", "baseline_score": .6, "comparison_score": .7},
        {"scorer_llm": "large", "feature_set": "extended", "baseline_score": .6, "comparison_score": .8},
        {"scorer_llm": "small", "feature_set": "scalar_all", "baseline_score": .6, "comparison_score": .65},
        {"scorer_llm": "large", "feature_set": "scalar_all", "baseline_score": .6, "comparison_score": .75},
    ])

    from notebooks.utils.feature_set_similarity_effect_utils import plot_feature_set_comparison

    figure = plot_feature_set_comparison(deltas, baseline_feature_set="simple_similarity", title="test")
    comparison_colors = [tuple(collection.get_facecolors()[0]) for collection in figure.axes[0].collections[1::2]]

    assert comparison_colors[0] == comparison_colors[1]
    assert comparison_colors[0] != comparison_colors[2]
    assert len(figure.axes[0].get_legend().get_texts()) == 2
    plt.close(figure)


def test_feature_set_comparison_includes_model_family_in_configuration_labels() -> None:
    deltas = pd.DataFrame([
        {"scorer_llm": "small", "model_family": "logistic_regression", "feature_set": "extended", "baseline_score": .6, "comparison_score": .7},
        {"scorer_llm": "small", "model_family": "random_forest", "feature_set": "extended", "baseline_score": .6, "comparison_score": .75},
    ])

    from notebooks.utils.feature_set_similarity_effect_utils import plot_feature_set_comparison

    figure = plot_feature_set_comparison(deltas, baseline_feature_set="scalar_compact", title="test")
    tick_labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]

    assert any("logistic_regression" in label for label in tick_labels)
    assert any("random_forest" in label for label in tick_labels)
    plt.close(figure)
