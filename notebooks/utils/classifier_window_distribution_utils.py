from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {
    "split",
    "text_id",
    "model_family",
    "window_configuration",
    "mg_score",
}
PLOT_COLUMNS = {"mg_window_count", "mg_window_percent"}


def aggregate_window_classifications(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level MG scores into per-text window metrics."""
    missing = REQUIRED_COLUMNS.difference(scored.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    values = scored.copy()
    values["mg_score"] = pd.to_numeric(values["mg_score"], errors="raise")
    values["is_mg_window"] = values["mg_score"].ge(0.5)
    result = (
        values.groupby(
            ["split", "text_id", "model_family", "window_configuration"],
            as_index=False,
        )
        .agg(
            mg_window_count=("is_mg_window", "sum"),
            window_count=("is_mg_window", "size"),
        )
    )
    result["mg_window_percent"] = 100.0 * result["mg_window_count"] / result["window_count"]
    return result.sort_values(
        ["split", "text_id", "model_family", "window_configuration"]
    ).reset_index(drop=True)


def plot_window_distribution(
    metrics: pd.DataFrame,
    value_column: str,
    *,
    axes: Sequence[plt.Axes] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot HG/MG distributions for each classifier/window configuration."""
    if value_column not in PLOT_COLUMNS:
        raise ValueError(f"value_column must be one of {sorted(PLOT_COLUMNS)}")
    required = {"split", "model_family", "window_configuration", value_column}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("metrics must not be empty")

    configurations = list(
        metrics[["model_family", "window_configuration"]]
        .drop_duplicates()
        .sort_values(["model_family", "window_configuration"])
        .itertuples(index=False, name=None)
    )
    if axes is None:
        figure, axis_array = plt.subplots(
            1,
            len(configurations),
            figsize=(5 * len(configurations), 4),
            squeeze=False,
        )
        axes_array = axis_array.ravel()
    else:
        axes_array = np.asarray(axes, dtype=object).ravel()
        if len(axes_array) != len(configurations):
            raise ValueError("axes count must match classifier/window configurations")
        figure = axes_array[0].figure

    for axis, (model_family, window_configuration) in zip(axes_array, configurations):
        subset = metrics.loc[
            metrics["model_family"].eq(model_family)
            & metrics["window_configuration"].eq(window_configuration)
        ]
        if value_column == "mg_window_count":
            maximum = int(subset[value_column].max())
            bins = np.arange(-0.5, maximum + 1.5, 1.0)
            axis.set_xlim(-0.5, maximum + 0.5)
            axis.set_xlabel("MG-positive windows per text")
        else:
            bins = np.linspace(0.0, 100.0, 21)
            axis.set_xlim(0.0, 100.0)
            axis.set_xlabel("MG-positive windows (%)")
        sns.histplot(
            data=subset,
            x=value_column,
            hue="split",
            hue_order=["HG", "MG"],
            bins=bins,
            element="step",
            stat="density",
            common_norm=False,
            ax=axis,
        )
        axis.set_title(f"{model_family} / {window_configuration}")
        axis.set_ylabel("Density")

    figure.tight_layout()
    return figure, axes_array
