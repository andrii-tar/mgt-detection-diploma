from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from notebooks.token_candidate_similarity_heatmaps_utils import _normalize_series, rolling_mean


def _validate_token_count(token_count: int) -> None:
    if isinstance(token_count, (bool, np.bool_)) or not isinstance(token_count, (int, np.integer)) or token_count <= 0:
        raise ValueError("token_count must be positive")


def _normalize_with_bounds(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = np.asarray(values, dtype="float64").copy()
    finite = np.isfinite(normalized)
    if minimum == maximum:
        normalized[finite] = 0.0
    else:
        normalized[finite] = (normalized[finite] - minimum) / (maximum - minimum)
    return normalized


def expand_model_scores_to_tokens(
    scored_windows: pd.DataFrame,
    token_count: int,
    model_column: str = "model",
) -> pd.DataFrame:
    required = {model_column, "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    _validate_token_count(token_count)

    score_columns = ["mg_score"]
    interval_columns = [column for column in ("mg_score_lower", "mg_score_upper") if column in scored_windows]
    score_columns.extend(interval_columns)
    models = list(pd.unique(scored_windows[model_column]))
    values = {
        model: {column: np.full(token_count, np.nan, dtype="float64") for column in score_columns}
        for model in models
    }
    sums = {
        model: {column: np.zeros(token_count, dtype="float64") for column in score_columns}
        for model in models
    }
    counts = {
        model: {column: np.zeros(token_count, dtype="int64") for column in score_columns}
        for model in models
    }
    for row in scored_windows.itertuples(index=False):
        model = getattr(row, model_column)
        start = getattr(row, "start_token")
        stop = getattr(row, "stop_token")
        if (
            isinstance(start, (bool, np.bool_))
            or isinstance(stop, (bool, np.bool_))
            or not isinstance(start, (int, np.integer))
            or not isinstance(stop, (int, np.integer))
            or start >= stop
            or start < 0
            or stop > token_count
        ):
            raise ValueError("invalid span")
        for column in score_columns:
            score = pd.to_numeric(pd.Series([getattr(row, column)]), errors="coerce").iloc[0]
            if np.isfinite(score):
                sums[model][column][start:stop] += float(score)
                counts[model][column][start:stop] += 1

    rows = []
    for model in models:
        for position in range(token_count):
            row = {model_column: model, "token_position": position}
            for column in score_columns:
                if counts[model][column][position]:
                    values[model][column][position] = (
                        sums[model][column][position] / counts[model][column][position]
                    )
                row[column] = values[model][column][position]
            rows.append(row)
    return pd.DataFrame(rows, columns=[model_column, "token_position", *score_columns])


def plot_model_score_heatmap(scored_windows: pd.DataFrame, token_count: int, title: str) -> plt.Figure:
    expanded = expand_model_scores_to_tokens(scored_windows, token_count)
    models = list(pd.unique(expanded["model"]))
    matrix = expanded.pivot(index="model", columns="token_position", values="mg_score").reindex(models).to_numpy()
    figure, axis = plt.subplots(figsize=(max(6, token_count / 10), max(2, len(models) * 0.5)), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_title(title)
    axis.set_ylabel("Model")
    axis.set_xlabel("Token position")
    axis.set_yticks(range(len(models)), labels=[str(model) for model in models])
    axis.set_xticks([])
    figure.colorbar(image, ax=axis, label="MG score")
    return figure


def plot_normalized_token_metrics_with_model_evidence(
    summary: dict[str, np.ndarray],
    scored_windows: pd.DataFrame,
    token_count: int,
    *,
    window: int = 10,
) -> plt.Figure:
    expanded = expand_model_scores_to_tokens(scored_windows, token_count)
    positions = np.arange(token_count)
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for name, values in summary.items():
        array = np.asarray(values, dtype="float64")
        if array.ndim != 1 or array.size != token_count:
            raise ValueError(f"{name} length must match token_count ({token_count}); got {array.size}")
        axis.plot(
            positions,
            _normalize_series(rolling_mean(array, window=window)),
            linewidth=1.8,
            alpha=0.5,
            label=f"{name} rolling mean ({window})",
        )

    for model in pd.unique(expanded["model"]):
        model_rows = expanded.loc[expanded["model"].eq(model)].sort_values("token_position")
        scores = model_rows["mg_score"].to_numpy(dtype="float64")
        axis.plot(positions, _normalize_series(scores), linewidth=2.0, label=f"local evidence ({model})")
        if {"mg_score_lower", "mg_score_upper"}.issubset(expanded.columns):
            lower = model_rows["mg_score_lower"].to_numpy(dtype="float64")
            upper = model_rows["mg_score_upper"].to_numpy(dtype="float64")
            finite = np.isfinite(lower) & np.isfinite(upper)
            score_finite = np.isfinite(scores)
            if finite.any() and score_finite.any():
                score_minimum = scores[score_finite].min()
                score_maximum = scores[score_finite].max()
                axis.fill_between(
                    positions,
                    _normalize_with_bounds(lower, score_minimum, score_maximum),
                    _normalize_with_bounds(upper, score_minimum, score_maximum),
                    where=finite,
                    alpha=0.15,
                    label=f"{model} credible interval",
                )

    axis.set_title(f"Normalized token metrics with model evidence (window={window})")
    axis.set_xlabel("Token position")
    axis.set_ylabel("Normalized value")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    return figure


def select_comparison_texts(predictions: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if "partition" not in predictions.columns:
        raise ValueError("predictions must contain a partition column")
    test_rows = predictions.loc[predictions["partition"].eq("test")]
    return test_rows.sample(n=min(int(count), len(test_rows)), random_state=seed).reset_index(drop=True)
