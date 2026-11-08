from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

from notebooks.utils.classifier_window_distribution_utils import (
    aggregate_window_classifications,
    plot_window_distribution,
)


matplotlib.use("Agg")


def test_aggregate_window_classifications_counts_positive_segments() -> None:
    scored = pd.DataFrame(
        {
            "split": ["HG", "HG", "MG"],
            "text_id": ["h1", "h1", "m1"],
            "model_family": ["logistic_regression"] * 3,
            "window_configuration": [32] * 3,
            "mg_score": [0.6, 0.4, 0.9],
        }
    )

    result = aggregate_window_classifications(scored)

    assert result[["mg_window_count", "window_count"]].to_dict("records") == [
        {"mg_window_count": 1, "window_count": 2},
        {"mg_window_count": 1, "window_count": 1},
    ]
    assert result.loc[result["text_id"].eq("h1"), "mg_window_percent"].item() == 50.0


def test_aggregate_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        aggregate_window_classifications(pd.DataFrame({"text_id": ["x"]}))


def test_plot_window_distribution_draws_one_axis_per_configuration() -> None:
    metrics = pd.DataFrame(
        {
            "split": ["HG", "MG"] * 2,
            "text_id": ["h1", "m1", "h2", "m2"],
            "model_family": ["logistic_regression"] * 2 + ["random_forest"] * 2,
            "window_configuration": [32, 32, 64, 64],
            "mg_window_count": [1, 2, 2, 3],
            "window_count": [2] * 4,
            "mg_window_percent": [50.0, 100.0, 100.0, 100.0],
        }
    )

    figure, axes = plot_window_distribution(metrics, "mg_window_count")

    assert len(axes.flat) == 2
    assert {label.get_text() for axis in axes.flat for label in axis.get_legend().texts} == {"HG", "MG"}
    figure.clf()


def test_plot_window_distribution_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="value_column"):
        plot_window_distribution(pd.DataFrame(), "unknown")
