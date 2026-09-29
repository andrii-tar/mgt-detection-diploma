from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from notebooks.utils.local_evidence_comparison_utils import (
    expand_model_scores_to_tokens,
    plot_model_score_heatmap,
    plot_comparison_token_profiles,
    plot_normalized_token_metrics_with_model_evidence,
    select_configured_comparison_texts,
    shared_comparison_texts,
    select_comparison_texts,
)


def _scored_windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "lr", "start_token": 0, "stop_token": 3, "mg_score": 0.2},
            {"model": "lr", "start_token": 2, "stop_token": 5, "mg_score": 0.8},
            {"model": "nb", "start_token": 0, "stop_token": 3, "mg_score": 0.6},
            {"model": "nb", "start_token": 2, "stop_token": 5, "mg_score": 0.4},
        ]
    )


def test_expand_model_scores_preserves_identity_and_averages_overlaps() -> None:
    expanded = expand_model_scores_to_tokens(_scored_windows(), token_count=5)

    assert len(expanded) == 10
    assert list(expanded.columns) == ["model", "token_position", "mg_score"]
    assert expanded.set_index(["model", "token_position"]).loc["lr", "mg_score"].tolist() == [
        0.2,
        0.2,
        0.5,
        0.8,
        0.8,
    ]
    assert expanded.set_index(["model", "token_position"]).loc["nb", "mg_score"].tolist() == [
        0.6,
        0.6,
        0.5,
        0.4,
        0.4,
    ]


def test_model_score_heatmap_labels_models_and_uses_shared_score_limits() -> None:
    figure = plot_model_score_heatmap(_scored_windows(), token_count=5, title="scores")

    axis = figure.axes[0]
    assert list(axis.get_yticklabels()[i].get_text() for i in range(2)) == ["lr", "nb"]
    assert axis.images[0].get_clim() == (0.0, 1.0)
    plt.close(figure)


def test_combined_plot_contains_metrics_model_lines_and_finite_interval_band() -> None:
    summary = {
        "observed_logprob": np.arange(5, dtype=float),
        "top20_coverage": np.linspace(0.2, 0.8, 5),
    }
    scored = _scored_windows().assign(
        mg_score_lower=lambda frame: frame["mg_score"] - 0.1,
        mg_score_upper=lambda frame: frame["mg_score"] + 0.1,
    )

    figure = plot_normalized_token_metrics_with_model_evidence(summary, scored, token_count=5, window=2)

    axis = figure.axes[0]
    labels = [line.get_label() for line in axis.lines]
    assert any("observed_logprob" in label for label in labels)
    assert any("local evidence (lr)" in label for label in labels)
    assert any("local evidence (nb)" in label for label in labels)
    assert len(axis.collections) == 2
    assert all("credible interval" in collection.get_label() for collection in axis.collections)
    assert axis.get_legend().get_bbox_to_anchor() is not None
    plt.close(figure)


def test_interval_band_uses_central_score_normalization() -> None:
    summary = {"observed_logprob": np.arange(5, dtype=float)}
    scored = _scored_windows().iloc[:2].assign(
        mg_score_lower=lambda frame: frame["mg_score"] - 0.1,
        mg_score_upper=lambda frame: frame["mg_score"] + 0.1,
    )

    figure = plot_normalized_token_metrics_with_model_evidence(summary, scored, token_count=5)

    vertices = figure.axes[0].collections[0].get_paths()[0].vertices
    assert np.isclose(vertices[:, 1], -1 / 6).any()
    assert np.isclose(vertices[:, 1], 1 / 6).any()
    plt.close(figure)


def test_select_comparison_texts_is_deterministic_and_uses_test_partition() -> None:
    predictions = pd.DataFrame(
        {
            "partition": ["train", "validation", "test", "test", "test"],
            "split": ["HG", "MG", "HG", "MG", "HG"],
            "text_id": [1, 2, 3, 4, 5],
            "model": ["lr"] * 5,
        }
    )

    selected = select_comparison_texts(predictions, count=2, seed=17)

    pd.testing.assert_frame_equal(selected, select_comparison_texts(predictions, count=2, seed=17))
    assert selected["partition"].eq("test").all()
    assert set(selected["text_id"]).issubset({3, 4, 5})
    assert len(selected) == 2
    assert list(predictions["partition"]) == ["train", "validation", "test", "test", "test"]
    assert set(selected["split"]).issubset({"HG", "MG"})


def _prediction_rows(text_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "partition": "test",
                "split": "MG",
                "text_id": text_id,
                "start_token": 0,
                "stop_token": 3,
            }
            for text_id in text_ids
        ]
    )


def _complete_prediction_frames() -> dict[tuple[str, int], pd.DataFrame]:
    return {
        (model, window_length): _prediction_rows([1, 2, 3])
        for model in ("logistic_regression", "random_forest")
        for window_length in (32, 64)
    }


def _complete_scored_windows(text_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text_id": text_id,
                "model": model,
                "window_length": window_length,
                "start_token": 0,
                "stop_token": 3,
                "mg_score": score,
            }
            for model, window_length, score in (
                ("logistic_regression", 32, 0.2),
                ("random_forest", 32, 0.4),
                ("logistic_regression", 64, 0.6),
                ("random_forest", 64, 0.8),
            )
        ]
    )


def test_shared_comparison_texts_intersects_configurations_and_is_deterministic() -> None:
    frames = _complete_prediction_frames()
    frames[("random_forest", 32)] = _prediction_rows([1, 2, 3, 4])

    first = shared_comparison_texts(frames, count=2, seed=42)
    second = shared_comparison_texts(frames, count=2, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["text_id"]).issubset({1, 2, 3})


def test_shared_comparison_texts_rejects_different_scoring_spans() -> None:
    frames = _complete_prediction_frames()
    frames[("random_forest", 32)] = frames[("random_forest", 32)].assign(stop_token=99)

    with pytest.raises(ValueError, match="scoring windows"):
        shared_comparison_texts(frames, count=1, seed=42)


def test_shared_comparison_texts_supports_a_single_model_configuration() -> None:
    frames = {
        ("logistic_regression", window_length): _prediction_rows([1, 2, 3])
        for window_length in (32, 64)
    }

    selected = shared_comparison_texts(
        frames,
        count=2,
        seed=42,
        configurations=tuple(frames),
    )

    assert set(selected["text_id"]).issubset({1, 2, 3})
    assert len(selected) == 2


def test_select_configured_comparison_texts_preserves_explicit_ids() -> None:
    selected = select_configured_comparison_texts(
        _complete_prediction_frames(),
        count=3,
        seed=42,
        selected_text_ids=[3, 1],
    )

    assert list(selected["text_id"]) == [3, 1]


def test_select_configured_comparison_texts_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="not available"):
        select_configured_comparison_texts(
            _complete_prediction_frames(),
            count=1,
            seed=42,
            selected_text_ids=[999],
        )


def test_plot_comparison_token_profiles_uses_four_columns_per_text() -> None:
    figure = plot_comparison_token_profiles(
        text_id=7,
        token_count=5,
        scored_windows=_complete_scored_windows(text_id=7),
        title="Mixed MG / 7",
        smoothing_window=2,
    )

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == ["LR-32", "RF-32", "LR-64", "RF-64"]
    assert all(axis.get_ylim() == (0.0, 1.0) for axis in figure.axes)
    plt.close(figure)


def test_plot_comparison_token_profiles_supports_a_single_model_configuration() -> None:
    scored = _complete_scored_windows(text_id=7).loc[lambda frame: frame["model"].eq("logistic_regression")]

    figure = plot_comparison_token_profiles(
        text_id=7,
        token_count=5,
        scored_windows=scored,
        title="Mixed MG / 7",
        smoothing_window=2,
        configurations=tuple(sorted({("logistic_regression", 32), ("logistic_regression", 64)})),
    )

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["LR-32", "LR-64"]
    plt.close(figure)


def test_plot_comparison_token_profiles_labels_three_semeval_rf_windows() -> None:
    configurations = (
        ("random_forest", 32),
        ("random_forest", 64),
        ("random_forest", 128),
    )
    scored = pd.concat(
        [
            pd.DataFrame(
                {
                    "text_id": [7],
                    "model": ["random_forest"],
                    "window_length": [window_length],
                    "start_token": [0],
                    "stop_token": [5],
                    "mg_score": [score],
                }
            )
            for window_length, score in ((32, 0.2), (64, 0.5), (128, 0.8))
        ],
        ignore_index=True,
    )

    figure = plot_comparison_token_profiles(
        text_id=7,
        token_count=5,
        scored_windows=scored,
        title="SemEval MG / gemma_3_270m_it",
        smoothing_window=2,
        configurations=configurations,
    )

    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == ["RF-32", "RF-64", "RF-128"]
    plt.close(figure)


def test_plot_comparison_token_profiles_marks_the_semeval_hg_mg_boundary() -> None:
    configurations = (
        ("random_forest", 32),
        ("random_forest", 64),
        ("random_forest", 128),
    )
    scored = pd.concat(
        [
            pd.DataFrame(
                {
                    "text_id": [7],
                    "model": ["random_forest"],
                    "window_length": [window_length],
                    "start_token": [0],
                    "stop_token": [5],
                    "mg_score": [score],
                }
            )
            for window_length, score in ((32, 0.2), (64, 0.5), (128, 0.8))
        ],
        ignore_index=True,
    )

    figure = plot_comparison_token_profiles(
        text_id=7,
        token_count=5,
        scored_windows=scored,
        title="SemEval MG / gemma_3_270m_it",
        smoothing_window=2,
        configurations=configurations,
        gold_token_boundary=2,
    )

    for axis in figure.axes:
        assert any(np.array_equal(line.get_xdata(), [2, 2]) for line in axis.lines)
        assert {text.get_text() for text in axis.texts}.issuperset({"HG", "MG"})
    plt.close(figure)
