from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


LOGPROB_CMAP = "coolwarm"
SIMILARITY_CMAP = "viridis"


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


def finite_min_max(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


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
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[float, float], tuple[float, float]]:
    tokens = list(logprob_row["tokens"])[parse_token_range(token_range)]
    similarities = list(similarity_row["token_similarities"])[parse_token_range(token_range)]
    width = min(len(tokens), len(similarities))
    height = top_k + 1
    logprob_matrix = np.full((height, width), np.nan, dtype="float64")
    similarity_matrix = np.full((height, width), np.nan, dtype="float64")
    token_labels = []

    for column, (token, similarity_token) in enumerate(zip(tokens[:width], similarities[:width])):
        token_labels.append(str(token.get("token", "")))

        gt_logprob = token.get("logprob")
        top_logprobs = token.get("top_logprobs")
        top_logprobs = [] if top_logprobs is None else list(top_logprobs)
        candidate_logprobs = [candidate.get("logprob") for candidate in top_logprobs[:top_k]]
        logprob_values = [gt_logprob] + candidate_logprobs + [None] * (top_k - len(candidate_logprobs))
        logprob_matrix[:, column] = [
            np.nan if value is None or gt_logprob is None else float(value) - float(gt_logprob)
            for value in logprob_values
        ]

        similarity_top_logprobs = similarity_token.get("top_logprobs")
        similarity_top_logprobs = [] if similarity_top_logprobs is None else list(similarity_top_logprobs)
        candidate_similarities = [candidate.get("similarity_to_gt") for candidate in similarity_top_logprobs[:top_k]]
        similarity_values = [1.0] + candidate_similarities + [None] * (top_k - len(candidate_similarities))
        similarity_matrix[:, column] = [np.nan if value is None else float(value) for value in similarity_values]

    logprob_scale = finite_min_max(logprob_matrix)
    similarity_scale = finite_min_max(similarity_matrix)
    return (
        np.nan_to_num(logprob_matrix, nan=0.0),
        np.nan_to_num(similarity_matrix, nan=0.0),
        token_labels,
        logprob_scale,
        similarity_scale,
    )


def draw_square_heatmap(
    matrix: np.ndarray,
    title: str,
    *,
    value_scale: tuple[float, float],
    cmap: str,
    center_zero: bool = False,
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
    axis.set_ylabel("GT + top candidates")
    axis.set_xlabel("Token position")
    axis.set_yticks([0, 1, height - 1])
    axis.set_yticklabels(["GT", "top 1", f"top {height - 1}"])
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


def visualize_text_similarities(
    processed_root: Path,
    dataset: str,
    model: str,
    split: str,
    text_index: int,
    *,
    top_k: int = 20,
    token_range: str | tuple[int | None, int | None] | slice | None = None,
) -> None:
    logprob_row = load_text_row(processed_root, dataset, model, split, text_index)
    similarity_row = load_similarity_row(processed_root, dataset, model, split, int(logprob_row["id"]))
    logprob_matrix, similarity_matrix, token_labels, logprob_scale, similarity_scale = build_text_heatmap_matrices(
        logprob_row,
        similarity_row,
        top_k=top_k,
        token_range=token_range,
    )

    print(f"dataset={dataset}, model={model}, split={split}, text_index={text_index}, id={int(logprob_row['id'])}, token_range={token_range}")
    print(f"tokens={len(token_labels)}, rows={top_k + 1}")
    display(pd.DataFrame({"text": [logprob_row.get("text", "")]}))
    draw_square_heatmap(
        logprob_matrix,
        "Signed candidate logprob - GT logprob",
        value_scale=logprob_scale,
        cmap=LOGPROB_CMAP,
        center_zero=True,
    )
    draw_square_heatmap(
        similarity_matrix,
        "Candidate similarity to GT token",
        value_scale=similarity_scale,
        cmap=SIMILARITY_CMAP,
    )
