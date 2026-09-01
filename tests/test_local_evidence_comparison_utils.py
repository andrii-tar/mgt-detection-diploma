from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.local_evidence_comparison_utils import (
    expand_model_scores_to_tokens,
    plot_model_score_heatmap,
    plot_normalized_token_metrics_with_model_evidence,
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
