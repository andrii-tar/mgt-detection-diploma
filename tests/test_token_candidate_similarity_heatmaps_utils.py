from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from notebooks.token_candidate_similarity_heatmaps_utils import (
    plot_focused_token_metrics_with_local_evidence,
)


def _scored_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_token": [0, 2],
            "stop_token": [3, 5],
            "mg_score": [10.0, 20.0],
        }
    )


def test_focused_plot_has_exactly_three_labels_and_lines(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = plot_focused_token_metrics_with_local_evidence(
        {
            "mean_candidate_similarity": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "top20_coverage": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        },
        _scored_windows(),
        token_count=5,
        window=1,
    )

    axis = figure.axes[0]
    assert [line.get_label() for line in axis.lines] == [
        "mean_candidate_similarity",
        "top20_coverage",
        "local mg_score",
    ]
    assert len(axis.lines) == 3
    assert axis.get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_focused_plot_normalizes_each_series_independently(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = plot_focused_token_metrics_with_local_evidence(
        {
            "mean_candidate_similarity": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "top20_coverage": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        },
        _scored_windows(),
        token_count=5,
        window=1,
    )

    assert [line.get_ydata()[[0, -1]].tolist() for line in figure.axes[0].lines] == [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    plt.close(figure)


@pytest.mark.parametrize(
    "summary, match",
    [
        ({"top20_coverage": np.ones(3)}, "missing required metric: mean_candidate_similarity"),
        ({"mean_candidate_similarity": np.ones(3)}, "missing required metric: top20_coverage"),
        (
            {
                "mean_candidate_similarity": np.ones(2),
                "top20_coverage": np.ones(3),
            },
            "mean_candidate_similarity length must match token_count",
        ),
        (
            {
                "mean_candidate_similarity": np.ones(3),
                "top20_coverage": np.ones(2),
            },
            "top20_coverage length must match token_count",
        ),
    ],
)
def test_focused_plot_rejects_missing_or_wrong_length_metrics(summary, match) -> None:
    with pytest.raises(ValueError, match=match):
        plot_focused_token_metrics_with_local_evidence(summary, _scored_windows(), token_count=3)


def test_focused_plot_preserves_existing_constant_normalization(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = plot_focused_token_metrics_with_local_evidence(
        {
            "mean_candidate_similarity": np.full(5, 7.0),
            "top20_coverage": np.full(5, 8.0),
        },
        pd.DataFrame(
            {"start_token": [0], "stop_token": [5], "mg_score": [9.0]}
        ),
        token_count=5,
    )

    assert all(np.array_equal(line.get_ydata(), np.zeros(5)) for line in figure.axes[0].lines)
    plt.close(figure)
