from __future__ import annotations

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from notebooks.utils.feature_set_similarity_effect_utils import (
    DEFAULT_MODEL_SIZES_BILLIONS,
    aggregate_feature_set_deltas,
    append_external_feature_set_comparison,
    attach_model_sizes,
    gpt_who_experiment_for_training_regime,
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


def test_plot_scorer_llms_are_ordered_by_model_size() -> None:
    from notebooks.utils.feature_set_similarity_effect_utils import plot_new_method_vs_gpt_who

    metrics = pd.DataFrame([
        {"scorer_llm": "gemma_3_1b_it", "feature_set": "simple_similarity", "macro_f1": .80},
        {"scorer_llm": "littlelamb", "feature_set": "simple_similarity", "macro_f1": .75},
        {"scorer_llm": "gemma_3_270m_it", "feature_set": "simple_similarity", "macro_f1": .70},
        {"scorer_llm": "gemma_3_1b_it", "feature_set": "gpt_who", "macro_f1": .78},
        {"scorer_llm": "littlelamb", "feature_set": "gpt_who", "macro_f1": .73},
        {"scorer_llm": "gemma_3_270m_it", "feature_set": "gpt_who", "macro_f1": .68},
    ])

    figure = plot_new_method_vs_gpt_who(metrics, title="test")
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
        "gemma_3_270m_it", "littlelamb", "gemma_3_1b_it",
    ]
    plt.close(figure)


def test_gpt_who_context_mapping_only_matches_mixed_regimes() -> None:
    assert gpt_who_experiment_for_training_regime("Kaggle") is None
    assert gpt_who_experiment_for_training_regime("Kaggle+MAGE+Outfox") == "smollm_mixed"
    assert gpt_who_experiment_for_training_regime("Kaggle+MAGE+Outfox+BBC") == "smollm_mixed_bbc"


def test_external_feature_set_comparison_uses_baseline_score_by_scorer() -> None:
    baseline = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "scalar_compact", "macro_f1": 0.60},
        {"scorer_llm": "small", "feature_set": "scalar_compact", "macro_f1": 0.70},
    ])
    external = pd.DataFrame([
        {"scorer_llm": "small", "macro_f1": 0.80},
    ])

    result = append_external_feature_set_comparison(
        baseline, external, baseline_feature_set="scalar_compact", external_feature_set="gpt_who",
    )

    row = result.iloc[0]
    assert row["feature_set"] == "gpt_who"
    assert row["baseline_score"] == pytest.approx(0.65)
    assert row["comparison_score"] == pytest.approx(0.80)
    assert row["delta"] == pytest.approx(0.15)


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


def test_feature_set_palette_distinguishes_all_named_methods() -> None:
    from notebooks.utils.feature_set_similarity_effect_utils import FEATURE_SET_COLORS

    colors = [
        FEATURE_SET_COLORS[name]
        for name in ("scalar_compact", "scalar_all", "extended", "simple_similarity", "gpt_who")
    ]

    assert len(set(colors)) == len(colors)


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


def test_feature_set_lines_use_scorer_llm_as_x_axis() -> None:
    from notebooks.utils.feature_set_similarity_effect_utils import plot_feature_set_lines

    metrics = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "simple_similarity", "model_family": "logistic_regression", "macro_f1": .70},
        {"scorer_llm": "large", "feature_set": "simple_similarity", "model_family": "logistic_regression", "macro_f1": .80},
        {"scorer_llm": "small", "feature_set": "gpt_who", "model_family": "gpt_who", "macro_f1": .75},
        {"scorer_llm": "large", "feature_set": "gpt_who", "model_family": "gpt_who", "macro_f1": .85},
    ])

    figure = plot_feature_set_lines(metrics, title="test")
    axis = figure.axes[0]

    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["small", "large"]
    assert len(axis.lines) == 2
    assert all(line.get_linestyle() == "None" for line in axis.lines)
    assert {line.get_label() for line in axis.lines} == {"logistic_regression / simple_similarity", "gpt_who"}
    assert {line.get_label(): line.get_marker() for line in axis.lines} == {
        "logistic_regression / simple_similarity": "o",
        "gpt_who": "D",
    }
    plt.close(figure)


def test_classifier_feature_set_plot_facets_model_families() -> None:
    from notebooks.utils.feature_set_similarity_effect_utils import plot_classifier_feature_set_comparison

    metrics = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "scalar_compact", "model_family": "logistic_regression", "macro_f1": .60},
        {"scorer_llm": "small", "feature_set": "simple_similarity", "model_family": "logistic_regression", "macro_f1": .75},
        {"scorer_llm": "small", "feature_set": "scalar_compact", "model_family": "random_forest", "macro_f1": .62},
        {"scorer_llm": "small", "feature_set": "simple_similarity", "model_family": "random_forest", "macro_f1": .78},
        {"scorer_llm": "small", "feature_set": "extended", "model_family": "random_forest", "macro_f1": .76},
        {"scorer_llm": "small", "feature_set": "gpt_who", "model_family": "gpt_who", "macro_f1": .70},
    ])

    figure = plot_classifier_feature_set_comparison(metrics, title="test")

    assert len(figure.axes) == 2
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert len(figure.legends) == 1
    legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert "GPT-who" in legend_labels
    assert "extended similarity" in legend_labels
    gpt_who_collections = [
        collection
        for axis in figure.axes
        for collection in axis.collections
        if collection.get_label() == "GPT-who"
    ]
    assert len(gpt_who_collections) == 2
    plt.close(figure)


def _presentation_plot_metrics() -> pd.DataFrame:
    rows = []
    for scorer_llm, base in (("small", 0.60), ("large", 0.70)):
        for model_family, family_offset in (("logistic_regression", 0.00), ("random_forest", 0.03)):
            for feature_set, feature_offset in (
                ("scalar_compact", 0.00),
                ("simple_similarity", 0.10),
                ("scalar_all", 0.05),
                ("extended", 0.08),
            ):
                rows.append({
                    "scorer_llm": scorer_llm,
                    "feature_set": feature_set,
                    "model_family": model_family,
                    "macro_f1": base + family_offset + feature_offset,
                })
        rows.append({
            "scorer_llm": scorer_llm,
            "feature_set": "gpt_who",
            "model_family": "gpt_who",
            "macro_f1": base + 0.04,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "plotter_name",
    [
        "plot_classifier_feature_set_bars_horizontal",
        "plot_classifier_feature_set_bars_vertical",
        "plot_classifier_feature_set_lollipop",
    ],
)
def test_presentation_plotters_use_external_legend_and_gpt_who_marker(plotter_name: str) -> None:
    from notebooks.utils import feature_set_similarity_effect_utils as utils

    figure = getattr(utils, plotter_name)(_presentation_plot_metrics(), title="test")

    assert len(figure.axes) == 2
    assert len(figure.legends) == 1
    legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert "GPT-who" in legend_labels
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert any(
        collection.get_label() == "GPT-who"
        for axis in figure.axes
        for collection in axis.collections
    ) or any(
        line.get_label() == "GPT-who" and line.get_marker() == "D"
        for axis in figure.axes
        for line in axis.lines
    )
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter_name",
    [
        "plot_classifier_feature_set_bars_horizontal",
        "plot_classifier_feature_set_bars_vertical",
        "plot_classifier_feature_set_lollipop",
    ],
)
def test_presentation_plotters_handle_empty_metrics(plotter_name: str) -> None:
    from notebooks.utils import feature_set_similarity_effect_utils as utils

    figure = getattr(utils, plotter_name)(pd.DataFrame(), title="test")

    assert len(figure.axes) == 1
    assert "No classifier feature-set scores" in figure.axes[0].texts[0].get_text()
    plt.close(figure)


def test_new_method_plot_compares_only_new_method_and_gpt_who() -> None:
    from notebooks.utils.feature_set_similarity_effect_utils import plot_new_method_vs_gpt_who

    metrics = pd.DataFrame([
        {"scorer_llm": "small", "feature_set": "simple_similarity", "model_family": "logistic_regression", "macro_f1": .80},
        {"scorer_llm": "small", "feature_set": "extended", "model_family": "logistic_regression", "macro_f1": .70},
        {"scorer_llm": "small", "feature_set": "gpt_who", "model_family": "gpt_who", "macro_f1": .75},
    ])

    figure = plot_new_method_vs_gpt_who(metrics, title="test")
    axis = figure.axes[0]

    assert {collection.get_label() for collection in axis.collections} == {"simple_similarity", "gpt_who"}
    assert len(axis.collections) == 2
    plt.close(figure)
