from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from notebooks.utils.token_candidate_similarity_heatmaps_utils import (
    split_dir,
    validated_text_summary,
)


TOKEN_FEATURE_COLUMNS = (
    "observed_logprob",
    "mean_candidate_logprob_delta",
    "mean_candidate_similarity",
    "top20_coverage",
)


def build_token_feature_frame(
    stage1_row: pd.Series,
    stage2_row: pd.Series,
    *,
    split: str,
    text_index: int | None = None,
) -> pd.DataFrame:
    summary, _ = validated_text_summary(stage1_row, stage2_row)
    token_count = len(next(iter(summary.values())))
    data: dict[str, object] = {
        "text_id": int(stage1_row["id"]),
        "split": split,
        "token_position": np.arange(token_count),
    }
    if text_index is not None:
        data["text_index"] = text_index
    data.update({feature: summary[feature] for feature in TOKEN_FEATURE_COLUMNS})
    return pd.DataFrame(data)


def load_balanced_token_feature_frame(
    processed_root: Path,
    dataset: str,
    model: str,
    *,
    texts_per_split: int = 100,
    random_state: int = 42,
    batch_size: int = 256,
    splits: tuple[str, ...] = ("HG", "MG"),
    show_progress: bool = False,
) -> pd.DataFrame:
    if texts_per_split <= 0:
        raise ValueError("texts_per_split must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not splits:
        raise ValueError("splits must not be empty")

    frames = []
    for split in splits:
        root = split_dir(processed_root, dataset, model, split)
        data_paths = sorted((root / "data").glob("*.parquet"))
        similarity_paths = sorted((root / "token_similarities").glob("*.parquet"))
        if not data_paths or not similarity_paths:
            raise FileNotFoundError(f"Missing paired data for {dataset}/{model}/{split}")

        similarity_ids = {
            int(row["id"])
            for row in _iter_parquet_rows(similarity_paths, columns=["id"], batch_size=batch_size)
        }
        reservoir: list[dict] = []
        eligible_count = 0
        rng = random.Random(random_state + (0 if split == "HG" else 1))
        for row in _iter_parquet_rows(data_paths, columns=["id", "tokens"], batch_size=batch_size):
            if int(row["id"]) not in similarity_ids:
                continue
            if len(reservoir) < texts_per_split:
                reservoir.append(row)
            else:
                replacement = rng.randrange(eligible_count + 1)
                if replacement < texts_per_split:
                    reservoir[replacement] = row
            eligible_count += 1
        if eligible_count < texts_per_split:
            raise ValueError(
                f"Need {texts_per_split} paired texts in {split}, found {eligible_count}"
            )
        selected_ids = {int(row["id"]) for row in reservoir}
        similarity_rows = {}
        for row in _iter_parquet_rows(
            similarity_paths,
            columns=["id", "token_similarities"],
            batch_size=batch_size,
        ):
            text_id = int(row["id"])
            if text_id in selected_ids:
                if text_id in similarity_rows:
                    raise ValueError(f"Duplicate token similarity row for text id {text_id}")
                similarity_rows[text_id] = row
        for text_index, stage1_row in enumerate(reservoir):
            text_id = int(stage1_row["id"])
            try:
                stage2_row = similarity_rows[text_id]
            except KeyError as error:
                raise KeyError(f"No token similarity row for text id {text_id}") from error
            frames.append(
                build_token_feature_frame(
                    pd.Series(stage1_row),
                    pd.Series(stage2_row),
                    split=split,
                    text_index=text_index,
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_balanced_token_feature_group_frame(
    processed_root: Path,
    datasets: tuple[str, ...],
    model: str,
    *,
    texts_per_split: int = 100,
    random_state: int = 42,
    batch_size: int = 256,
    dataset_splits: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    if not datasets:
        raise ValueError("datasets must not be empty")
    split_map = dataset_splits or {dataset: ("HG", "MG") for dataset in datasets}
    source_frames = []
    for dataset_index, dataset in enumerate(datasets):
        source = load_balanced_token_feature_frame(
            processed_root,
            dataset,
            model,
            texts_per_split=texts_per_split,
            random_state=random_state + dataset_index,
            batch_size=batch_size,
            splits=split_map.get(dataset, ("HG", "MG")),
        )
        source["source_dataset"] = dataset
        source_frames.append(source)

    combined = pd.concat(source_frames, ignore_index=True)
    selected_frames = []
    for split_index, split in enumerate(("HG", "MG")):
        available = combined.loc[combined["split"].eq(split)]
        if len(available[["source_dataset", "text_id"]].drop_duplicates()) < texts_per_split:
            raise ValueError(f"Need {texts_per_split} paired texts in group split {split}")
        selected_ids = (
            available[["source_dataset", "text_id"]]
            .drop_duplicates()
            .sample(n=texts_per_split, random_state=random_state + 100 + split_index)
        )
        selected_frames.append(
            available.merge(selected_ids, on=["source_dataset", "text_id"], how="inner")
        )
    return pd.concat(selected_frames, ignore_index=True)


def _iter_parquet_rows(
    paths: list[Path],
    *,
    columns: list[str],
    batch_size: int,
):
    for path in paths:
        for record_batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            yield from record_batch.to_pylist()


def plot_all_feature_pairs(
    token_features: pd.DataFrame,
    *,
    alpha: float = 0.18,
    point_size: float = 8,
) -> tuple[plt.Figure, list[plt.Axes]]:
    missing = set(TOKEN_FEATURE_COLUMNS).difference(token_features.columns)
    if missing:
        raise ValueError(f"token feature frame is missing columns: {sorted(missing)}")

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_list = list(axes.flat)
    palette = {"HG": "#0072B2", "MG": "#D55E00"}
    for axis, (x_feature, y_feature) in zip(axes_list, combinations(TOKEN_FEATURE_COLUMNS, 2)):
        plot_data = token_features[["split", x_feature, y_feature]].dropna()
        sns.scatterplot(
            data=plot_data,
            x=x_feature,
            y=y_feature,
            hue="split",
            hue_order=["HG", "MG"],
            palette=palette,
            alpha=alpha,
            s=point_size,
            linewidth=0,
            ax=axis,
        )
        axis.set_title(f"{x_feature} vs {y_feature}")
        axis.legend(title="Class")
    figure.suptitle("Token Feature Pair Scatterplots", fontsize=16)
    return figure, axes_list
