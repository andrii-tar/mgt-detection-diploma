from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from notebooks.utils.token_candidate_similarity_heatmaps_utils import _normalize_series, rolling_mean


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


_COMPARISON_CONFIGURATIONS = (
    ("logistic_regression", 32),
    ("random_forest", 32),
    ("logistic_regression", 64),
    ("random_forest", 64),
)


def shared_comparison_texts(
    predictions_by_configuration: Mapping[tuple[str, int], pd.DataFrame],
    count: int,
    seed: int,
    *,
    partition: str = "test",
    configurations: tuple[tuple[str, int], ...] = _COMPARISON_CONFIGURATIONS,
) -> pd.DataFrame:
    """Select deterministic text keys available with identical scoring spans."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not configurations:
        raise ValueError("configurations must not be empty")
    missing_configurations = set(configurations).difference(predictions_by_configuration)
    if missing_configurations:
        raise ValueError(f"missing comparison configurations: {sorted(missing_configurations)}")

    required = {"partition", "split", "text_id", "start_token", "stop_token"}
    filtered: dict[tuple[str, int], pd.DataFrame] = {}
    key_sets: list[set[tuple[object, object]]] = []
    for configuration in configurations:
        frame = predictions_by_configuration[configuration]
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"prediction frame is missing columns: {', '.join(sorted(missing))}")
        selected = frame.loc[frame["partition"].eq(partition)].copy()
        filtered[configuration] = selected
        key_sets.append(set(selected[["split", "text_id"]].itertuples(index=False, name=None)))

    shared_keys = set.intersection(*key_sets) if key_sets else set()
    if not shared_keys:
        raise ValueError(f"no shared texts found in partition {partition!r}")

    for window_length in sorted({window for _, window in configurations}):
        model_spans: dict[str, dict[tuple[object, object], frozenset[tuple[object, object]]]] = {}
        models = [model for model, window in configurations if window == window_length]
        for model in models:
            frame = filtered[(model, window_length)]
            spans = {}
            for key, group in frame.groupby(["split", "text_id"], sort=False):
                spans[key] = frozenset(group[["start_token", "stop_token"]].itertuples(index=False, name=None))
            model_spans[model] = spans
        for key in sorted(shared_keys, key=lambda value: (str(value[0]), str(value[1]))):
            expected_spans = model_spans[models[0]].get(key)
            if any(model_spans[model].get(key) != expected_spans for model in models[1:]):
                raise ValueError(f"scoring windows do not align for {key!r} at window length {window_length}")

    candidates = pd.DataFrame(sorted(shared_keys), columns=["split", "text_id"])
    return candidates.sample(n=min(int(count), len(candidates)), random_state=seed).reset_index(drop=True)


def select_configured_comparison_texts(
    predictions_by_configuration: Mapping[tuple[str, int], pd.DataFrame],
    count: int,
    seed: int,
    *,
    partition: str = "test",
    configurations: tuple[tuple[str, int], ...] = _COMPARISON_CONFIGURATIONS,
    selected_text_ids: list[object] | tuple[object, ...] | None = None,
) -> pd.DataFrame:
    """Select configured text IDs or fall back to deterministic sampling."""
    available = shared_comparison_texts(
        predictions_by_configuration,
        count=sum(len(frame) for frame in predictions_by_configuration.values()),
        seed=seed,
        partition=partition,
        configurations=configurations,
    )
    if selected_text_ids is None:
        return available.sample(n=min(int(count), len(available)), random_state=seed).reset_index(drop=True)

    requested = list(selected_text_ids)
    rows_by_text_id = {
        row.text_id: {"split": row.split, "text_id": row.text_id}
        for row in available.itertuples(index=False)
    }
    missing = [text_id for text_id in requested if text_id not in rows_by_text_id]
    if missing:
        raise ValueError(f"configured text IDs are not available: {missing}")
    return pd.DataFrame([rows_by_text_id[text_id] for text_id in requested], columns=["split", "text_id"])


def plot_comparison_token_profiles(
    text_id: object,
    token_count: int,
    scored_windows: pd.DataFrame,
    title: str,
    *,
    smoothing_window: int = 10,
    configurations: tuple[tuple[str, int], ...] = _COMPARISON_CONFIGURATIONS,
    gold_token_boundary: int | None = None,
) -> plt.Figure:
    """Plot projected per-token MG scores for the selected configurations."""
    required = {"model", "window_length", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"scored windows are missing columns: {', '.join(sorted(missing))}")
    _validate_token_count(token_count)
    if (
        not isinstance(smoothing_window, (int, np.integer))
        or isinstance(smoothing_window, (bool, np.bool_))
        or smoothing_window <= 0
    ):
        raise ValueError("smoothing_window must be positive")
    if gold_token_boundary is not None and (
        isinstance(gold_token_boundary, (bool, np.bool_))
        or not isinstance(gold_token_boundary, (int, np.integer))
        or not 0 <= gold_token_boundary <= token_count
    ):
        raise ValueError("gold_token_boundary must be an integer within the token range")

    if not configurations:
        raise ValueError("configurations must not be empty")
    figure, axes = plt.subplots(
        1,
        len(configurations),
        figsize=(max(4 * len(configurations), 6), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    positions = np.arange(token_count)
    labels = {
        ("logistic_regression", 32): "LR-32",
        ("random_forest", 32): "RF-32",
        ("logistic_regression", 64): "LR-64",
        ("random_forest", 64): "RF-64",
    }
    for axis, configuration in zip(axes, configurations, strict=True):
        model, window_length = configuration
        selected = scored_windows.loc[
            scored_windows["model"].eq(model) & scored_windows["window_length"].eq(window_length)
        ].copy()
        if selected.empty:
            raise ValueError(f"missing scored windows for {model}, {window_length}")
        projected = expand_model_scores_to_tokens(selected, token_count)
        values = projected["mg_score"].to_numpy(dtype="float64")
        axis.plot(positions, values, color="0.65", linewidth=0.8, label="raw")
        axis.plot(
            positions,
            rolling_mean(values, window=int(smoothing_window)),
            color="#d62728",
            linewidth=1.8,
            label=f"rolling mean ({smoothing_window})",
        )
        model_label = {"logistic_regression": "LR", "random_forest": "RF"}.get(model, model)
        axis.set_title(labels.get(configuration, f"{model_label}-{window_length}"))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Token position")
        axis.grid(axis="y", alpha=0.25)
        if gold_token_boundary is not None:
            axis.axvline(gold_token_boundary, color="0.2", linestyle="--", linewidth=1.2)
            axis.text(
                gold_token_boundary / 2,
                1.02,
                "HG",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="0.2",
            )
            axis.text(
                (gold_token_boundary + token_count) / 2,
                1.02,
                "MG",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="0.2",
            )
    axes[0].set_ylabel("Projected MG score")
    axes[-1].legend(loc="best")
    figure.suptitle(f"{title} (text_id={text_id})")
    return figure
