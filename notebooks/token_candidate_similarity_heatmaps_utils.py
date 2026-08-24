from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display

from scripts.logprob_metrics import compute_top20_coverage, compute_top20_d_score


LOGPROB_CMAP = "coolwarm"
SIMILARITY_CMAP = "viridis"
COVERAGE_CMAP = "magma"


class TextHeatmapMatrices(NamedTuple):
    logprob_matrix: np.ndarray
    similarity_matrix: np.ndarray
    token_labels: list[str]
    logprob_scale: tuple[float, float]
    similarity_scale: tuple[float, float]
    coverage_matrix: np.ndarray
    coverage_scale: tuple[float, float]


def path_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def experiment_name(dataset: str, model: str) -> str:
    return f"{path_slug(dataset)}__{path_slug(model)}"


def split_dir(processed_root: Path, dataset: str, model: str, split: str) -> Path:
    return processed_root / experiment_name(dataset, model) / split


def load_text_row(processed_root: Path, dataset: str, model: str, split: str, text_index: int) -> pd.Series:
    data_paths = sorted((split_dir(processed_root, dataset, model, split) / "data").glob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No stage-1 data parquet files for {dataset=} {model=} {split=}")
    offset = 0
    for path in data_paths:
        frame = pd.read_parquet(path)
        if text_index < offset + len(frame):
            return frame.iloc[text_index - offset]
        offset += len(frame)
    raise IndexError(f"text_index {text_index} is outside {offset} available rows")


def load_similarity_row(processed_root: Path, dataset: str, model: str, split: str, text_id: int) -> pd.Series:
    similarity_paths = sorted((split_dir(processed_root, dataset, model, split) / "token_similarities").glob("*.parquet"))
    if not similarity_paths:
        raise FileNotFoundError(f"No token similarity parquet files for {dataset=} {model=} {split=}")
    for path in similarity_paths:
        frame = pd.read_parquet(path)
        matches = frame[frame["id"] == text_id]
        if not matches.empty:
            return matches.iloc[0]
    raise KeyError(f"No token similarity row for text id {text_id}")


def _iter_stage_one_rows(
    processed_root: Path,
    dataset: str,
    model: str,
    split: str,
) -> Iterator[pd.Series]:
    data_paths = sorted((split_dir(processed_root, dataset, model, split) / "data").glob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No stage-1 data parquet files for {dataset=} {model=} {split=}")
    for path in data_paths:
        yield from (row for _, row in pd.read_parquet(path).iterrows())


def calculate_rolling_statistics(
    processed_root: Path,
    dataset: str,
    model: str,
    *,
    splits: tuple[str, ...] = ("HG", "MG"),
    top_k: int = 20,
    window: int = 5,
) -> pd.DataFrame:
    rows = []
    columns = ["split", "text_id", "metric", "mean", "std", "variance"]
    for split in splits:
        for logprob_row in _iter_stage_one_rows(processed_root, dataset, model, split):
            text_id = int(logprob_row["id"])
            similarity_row = load_similarity_row(processed_root, dataset, model, split, text_id)
            matrices = build_text_heatmap_matrices(logprob_row, similarity_row, top_k=top_k)
            coverage = compute_token_coverage(logprob_row, top_k=top_k)
            summary = token_summary_series(matrices, coverage[: len(matrices.token_labels)])
            for metric, values in summary.items():
                smoothed = rolling_mean(values, window=window)
                finite = smoothed[np.isfinite(smoothed)]
                rows.append(
                    {
                        "split": split,
                        "text_id": text_id,
                        "metric": metric,
                        "mean": float(finite.mean()) if finite.size else math.nan,
                        "std": float(finite.std()) if finite.size else math.nan,
                        "variance": float(finite.var()) if finite.size else math.nan,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def finite_min_max(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


def rolling_mean(
    values: np.ndarray | list[float],
    *,
    window: int = 10,
    min_periods: int = 1,
) -> np.ndarray:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an integer")
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise ValueError("min_periods must be an integer")
    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods <= 0 or min_periods > window:
        raise ValueError("min_periods must be between 1 and window")
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    result = np.full(array.shape, np.nan, dtype="float64")
    left_radius = window // 2
    right_radius = (window - 1) // 2
    for position in range(len(array)):
        start = max(0, position - left_radius)
        stop = min(len(array), position + right_radius + 1)
        finite = array[start:stop][np.isfinite(array[start:stop])]
        if finite.size >= min_periods:
            result[position] = finite.mean()
    return result


def token_summary_series(
    matrices: TextHeatmapMatrices,
    coverage: np.ndarray,
    observed_logprobs: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    coverage_array = np.asarray(coverage, dtype="float64")
    heatmap_width = matrices.logprob_matrix.shape[1]
    if coverage_array.ndim != 1 or coverage_array.size != heatmap_width:
        raise ValueError(
            f"coverage length must match heatmap token width ({heatmap_width}); "
            f"got {coverage_array.size}"
        )
    logprob_candidates = np.where(np.isfinite(matrices.logprob_matrix[1:]), matrices.logprob_matrix[1:], np.nan)
    similarity_candidates = np.where(np.isfinite(matrices.similarity_matrix[1:]), matrices.similarity_matrix[1:], np.nan)
    with np.errstate(invalid="ignore"):
        mean_logprob_delta = np.nanmean(logprob_candidates, axis=0)
        mean_similarity = np.nanmean(similarity_candidates, axis=0)
    mean_logprob_delta = np.where(np.isnan(logprob_candidates).all(axis=0), np.nan, mean_logprob_delta)
    mean_similarity = np.where(np.isnan(similarity_candidates).all(axis=0), np.nan, mean_similarity)
    if observed_logprobs is None:
        observed_array = matrices.logprob_matrix[0].copy()
    else:
        observed_array = np.asarray(observed_logprobs, dtype="float64")
        if observed_array.ndim != 1 or observed_array.size != heatmap_width:
            raise ValueError(
                f"observed_logprobs length must match heatmap token width ({heatmap_width}); "
                f"got {observed_array.size}"
            )
        observed_array = np.where(np.isfinite(observed_array), observed_array, np.nan)
    return {
        "observed_logprob": observed_array,
        "mean_candidate_logprob_delta": mean_logprob_delta,
        "mean_candidate_similarity": mean_similarity,
        "top20_coverage": coverage_array.copy(),
    }


def _finite_candidate_logprobs(token: dict) -> list[float]:
    candidates = token.get("top_logprobs")
    candidates = [] if candidates is None else list(candidates)
    values = []
    for candidate in candidates:
        value = candidate.get("logprob")
        values.append(float(value) if value is not None and math.isfinite(float(value)) else math.nan)
    return values


def compute_token_coverage(
    logprob_row: pd.Series,
    top_k: int = 20,
    token_range: str | tuple[int | None, int | None] | slice | None = None,
) -> np.ndarray:
    """Return cumulative top-k candidate probability mass per token position."""
    coverage = []
    for token in list(logprob_row["tokens"])[parse_token_range(token_range)]:
        actual_logprob = token.get("logprob")
        if actual_logprob is None or not math.isfinite(float(actual_logprob)):
            coverage.append(math.nan)
            continue
        candidate_logprobs = _finite_candidate_logprobs(token)[:top_k]
        coverage.append(math.fsum(math.exp(value) for value in candidate_logprobs) if candidate_logprobs else math.nan)
    return np.asarray(coverage, dtype="float64")


def text_d_score_summary(logprob_row: pd.Series) -> dict[str, float | int]:
    tokens = list(logprob_row.get("tokens", []))
    d_score_tokens = [
        token
        for token in tokens
        if token.get("logprob") is not None
        and token.get("top_logprobs") is not None
        and len(token["top_logprobs"]) > 0
    ]
    return {
        "d_score_top20": compute_top20_d_score(d_score_tokens),
        "d_score_top20_coverage": compute_top20_coverage(d_score_tokens),
        "d_score_top20_num_tokens": sum(
            any(candidate.get("logprob") is not None for candidate in token.get("top_logprobs", []))
            for token in d_score_tokens
        ),
    }


def parse_token_range(token_range: str | tuple[int | None, int | None] | slice | None) -> slice:
    if token_range is None:
        return slice(None)
    if isinstance(token_range, slice):
        return token_range
    if isinstance(token_range, tuple):
        return slice(*token_range)
    if isinstance(token_range, str):
        if "-" in token_range and ":" not in token_range:
            start, stop = token_range.split("-", maxsplit=1)
        else:
            start, stop = token_range.split(":", maxsplit=1)
        return slice(int(start) if start else None, int(stop) if stop else None)
    raise TypeError("token_range must be None, slice, tuple, or string like '0-100', '50:200', or '-30:'")


def build_text_heatmap_matrices(
    logprob_row: pd.Series,
    similarity_row: pd.Series,
    top_k: int = 20,
    token_range: str | tuple[int | None, int | None] | slice | None = None,
) -> TextHeatmapMatrices:
    tokens = list(logprob_row["tokens"])[parse_token_range(token_range)]
    similarities = list(similarity_row["token_similarities"])[parse_token_range(token_range)]
    width = min(len(tokens), len(similarities))
    height = top_k + 1
    logprob_matrix = np.full((height, width), np.nan, dtype="float64")
    similarity_matrix = np.full((height, width), np.nan, dtype="float64")
    coverage_matrix = np.full((top_k, width), np.nan, dtype="float64")
    token_labels = []

    for column, (token, similarity_token) in enumerate(zip(tokens[:width], similarities[:width])):
        token_labels.append(str(token.get("token", "")))

        gt_logprob = token.get("logprob")
        candidate_logprobs = _finite_candidate_logprobs(token)[:top_k]
        logprob_values = [gt_logprob] + candidate_logprobs + [None] * (top_k - len(candidate_logprobs))
        logprob_matrix[:, column] = [
            np.nan if value is None or gt_logprob is None else float(value) - float(gt_logprob)
            for value in logprob_values
        ]
        coverage_matrix[:, column] = [math.exp(value) if math.isfinite(value) else math.nan for value in candidate_logprobs[:top_k]] + [
            np.nan
        ] * (top_k - len(candidate_logprobs))

        similarity_top_logprobs = similarity_token.get("top_logprobs")
        similarity_top_logprobs = [] if similarity_top_logprobs is None else list(similarity_top_logprobs)
        candidate_similarities = [candidate.get("similarity_to_gt") for candidate in similarity_top_logprobs[:top_k]]
        similarity_values = [1.0] + candidate_similarities + [None] * (top_k - len(candidate_similarities))
        similarity_matrix[:, column] = [np.nan if value is None else float(value) for value in similarity_values]

    logprob_scale = finite_min_max(logprob_matrix)
    similarity_scale = finite_min_max(similarity_matrix)
    coverage_scale = finite_min_max(coverage_matrix)
    return TextHeatmapMatrices(
        logprob_matrix,
        similarity_matrix,
        token_labels,
        logprob_scale,
        similarity_scale,
        coverage_matrix,
        coverage_scale,
    )


def draw_square_heatmap(
    matrix: np.ndarray,
    title: str,
    *,
    value_scale: tuple[float, float],
    cmap: str,
    center_zero: bool = False,
    y_label: str = "GT + top candidates",
    row_labels: list[str] | None = None,
    square_size_px: int = 10,
    dpi: int = 100,
) -> None:
    height, width = matrix.shape
    fig_width = max(width * square_size_px / dpi, 6)
    fig_height = max(height * square_size_px / dpi, 1.5)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if center_zero:
        bound = max(abs(value_scale[0]), abs(value_scale[1]))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound) if bound > 0 else None
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    else:
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=value_scale[0], vmax=value_scale[1])
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel("Token position")
    if row_labels is None:
        tick_positions = sorted(set([0, min(1, height - 1), height - 1]))
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(["GT" if position == 0 else f"top {position}" for position in tick_positions])
    else:
        axis.set_yticks(range(height))
        axis.set_yticklabels(row_labels)
    axis.set_xticks([])
    colorbar = fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    colorbar.set_label(f"Value range: min={value_scale[0]:.4g}, max={value_scale[1]:.4g}")
    if value_scale[0] == value_scale[1]:
        colorbar.set_ticks([value_scale[0]])
        colorbar.set_ticklabels([f"{value_scale[0]:.4g}"])
    else:
        colorbar.set_ticks([value_scale[0], value_scale[1]])
        colorbar.set_ticklabels([f"min {value_scale[0]:.4g}", f"max {value_scale[1]:.4g}"])
    plt.show()


def draw_token_coverage_scatter(coverage: np.ndarray) -> None:
    positions = np.flatnonzero(np.isfinite(coverage))
    values = coverage[positions]
    fig, axis = plt.subplots(figsize=(max(len(coverage) * 10 / 100, 6), 3), dpi=100)
    axis.scatter(positions, values, s=12)
    axis.set_title("Top-20 candidate probability-mass coverage by token")
    axis.set_xlabel("Token position")
    axis.set_ylabel("Cumulative top-20 probability mass")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    plt.show()


def plot_smoothed_token_summaries(
    summary: dict[str, np.ndarray],
    *,
    window: int = 10,
) -> None:
    figure, axes = plt.subplots(
        len(summary),
        1,
        figsize=(12, 2.8 * len(summary)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (name, values) in zip(axes[:, 0], summary.items()):
        positions = np.arange(len(values))
        smoothed = rolling_mean(values, window=window)
        finite_smoothed = smoothed[np.isfinite(smoothed)]
        if finite_smoothed.size:
            mean = float(finite_smoothed.mean())
            std = float(finite_smoothed.std())
            variance = float(finite_smoothed.var())
        else:
            mean = std = variance = float("nan")
        axis.plot(positions, values, color="0.65", linewidth=0.8, label="raw")
        axis.plot(
            positions,
            smoothed,
            color="#d62728",
            linewidth=1.8,
            label=f"rolling mean ({window})",
        )
        axis.set_title(
            f"{name} (window={window}): "
            f"mean={mean:.4f}, std={std:.4f}, variance={variance:.4f}"
        )
        axis.set_ylabel(name)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Token position")
    figure.suptitle(f"Token-level metrics with rolling smoothing (window={window})")
    plt.show()


def plot_rolling_statistics_histograms(
    statistics: pd.DataFrame,
    *,
    bins: int = 30,
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    metrics = sorted(statistics["metric"].dropna().unique())
    splits = ("HG", "MG")
    figures = []
    statistic_labels = {
        "mean": "mean",
        "std": "standard deviation",
        "variance": "variance",
    }

    for statistic, label in statistic_labels.items():
        figure, axes = plt.subplots(
            len(metrics),
            len(splits),
            squeeze=False,
            sharex="row",
            constrained_layout=True,
        )
        for row_index, metric in enumerate(metrics):
            for column_index, split in enumerate(splits):
                axis = axes[row_index, column_index]
                values = statistics.loc[
                    (statistics["metric"] == metric) & (statistics["split"] == split),
                    statistic,
                ].to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    axis.hist(finite_values, bins=bins)
                else:
                    axis.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axis.transAxes)
                if row_index == 0:
                    axis.set_title(split)
                if column_index == 0:
                    axis.set_ylabel(metric)
                if row_index == len(metrics) - 1:
                    axis.set_xlabel(label)
        figure.suptitle(f"Per-text rolling {label} distributions")
        plt.show()
        figures.append(figure)

    return tuple(figures)


def visualize_text_similarities(
    processed_root: Path,
    dataset: str,
    model: str,
    split: str,
    text_index: int,
    *,
    top_k: int = 20,
    token_range: str | tuple[int | None, int | None] | slice | None = None,
    smoothing_window: int = 10,
    show_smoothed: bool = True,
) -> None:
    logprob_row = load_text_row(processed_root, dataset, model, split, text_index)
    similarity_row = load_similarity_row(processed_root, dataset, model, split, int(logprob_row["id"]))
    matrices = build_text_heatmap_matrices(
        logprob_row,
        similarity_row,
        top_k=top_k,
        token_range=token_range,
    )

    print(f"dataset={dataset}, model={model}, split={split}, text_index={text_index}, id={int(logprob_row['id'])}, token_range={token_range}")
    print(f"tokens={len(matrices.token_labels)}, rows={top_k + 1}")
    summary = text_d_score_summary(logprob_row)
    print(f"d_score_top20_coverage={summary['d_score_top20_coverage']:.6g}")
    summary["text_id"] = int(logprob_row["id"])
    display(pd.DataFrame([summary], columns=[
        "text_id",
        "d_score_top20",
        "d_score_top20_coverage",
        "d_score_top20_num_tokens",
    ]))
    display(pd.DataFrame({"text": [logprob_row.get("text", "")]}))
    draw_square_heatmap(
        matrices.logprob_matrix,
        "Signed candidate logprob - GT logprob",
        value_scale=matrices.logprob_scale,
        cmap=LOGPROB_CMAP,
        center_zero=True,
    )
    draw_square_heatmap(
        matrices.similarity_matrix,
        "Candidate similarity to GT token",
        value_scale=matrices.similarity_scale,
        cmap=SIMILARITY_CMAP,
    )
    draw_square_heatmap(
        matrices.coverage_matrix,
        "Candidate probability-mass coverage",
        value_scale=matrices.coverage_scale,
        cmap=COVERAGE_CMAP,
        y_label="Candidate rank",
        row_labels=[f"top {rank}" for rank in range(1, top_k + 1)],
    )
    draw_token_coverage_scatter(compute_token_coverage(logprob_row, top_k=top_k, token_range=token_range))
    if show_smoothed:
        coverage = compute_token_coverage(logprob_row, top_k=top_k, token_range=token_range)
        observed_logprobs = np.asarray(
            [
                float(token["logprob"])
                if token.get("logprob") is not None and math.isfinite(float(token["logprob"]))
                else math.nan
                for token in list(logprob_row["tokens"])[parse_token_range(token_range)]
            ],
            dtype="float64",
        )
        matrix_width = len(matrices.token_labels)
        summary_series = token_summary_series(
            matrices,
            coverage[:matrix_width],
            observed_logprobs=observed_logprobs[:matrix_width],
        )
        plot_smoothed_token_summaries(summary_series, window=smoothing_window)
