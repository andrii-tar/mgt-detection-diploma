from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


TRAINING_DATASETS = {
    "Kaggle": {"kaggle_train"},
    "MAGE": {"mage_train"},
    "Outfox": {"outfox_chatgpt_train"},
    "BBC": {"bbc_news_content"},
    "Mixed": {"kaggle_train", "mage_train", "outfox_chatgpt_train"},
    "Mixed+BBC": {
        "kaggle_train",
        "mage_train",
        "outfox_chatgpt_train",
        "bbc_news_content",
    },
}


def prediction_parent_lengths(path: Path | str) -> pd.DataFrame:
    batches = []
    parquet_path = Path(path) / "data.parquet"
    for batch in pq.ParquetFile(parquet_path).iter_batches(
        columns=["split", "text_id", "stop_token"]
    ):
        batches.append(batch.to_pandas())
    if not batches:
        return pd.DataFrame(columns=["split", "text_id", "token_count"])
    frame = pd.concat(batches, ignore_index=True)
    return (
        frame.groupby(["split", "text_id"], as_index=False)["stop_token"]
        .max()
        .rename(columns={"stop_token": "token_count"})
    )


def filter_prediction_cohort(
    predictions: pd.DataFrame, *, min_text_tokens: int | None
) -> pd.DataFrame:
    if min_text_tokens is None:
        return predictions.copy()
    if min_text_tokens < 0:
        raise ValueError("min_text_tokens must be non-negative")
    required = {"split", "text_id", "token_count"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing cohort columns: {sorted(missing)}")
    parents = predictions[["split", "text_id", "token_count"]].drop_duplicates()
    eligible = parents.loc[parents["token_count"] >= min_text_tokens, ["split", "text_id"]]
    return predictions.merge(eligible, on=["split", "text_id"], how="inner")


def held_out_binary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"training_regime", "dataset", "partition", "hg_support", "mg_support"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"missing metric columns: {sorted(missing)}")
    result = metrics.loc[
        (metrics["partition"] == "test")
        & (metrics["hg_support"] > 0)
        & (metrics["mg_support"] > 0)
    ].copy()
    result = result.loc[
        ~result.apply(
            lambda row: row["dataset"] in TRAINING_DATASETS.get(row["training_regime"], set()),
            axis=1,
        )
    ]
    return result.reset_index(drop=True)


def select_comparison_metrics(
    metrics: pd.DataFrame, *, min_text_tokens: int | None
) -> pd.DataFrame:
    """Select binary test rows for descriptive cohort plots."""
    required = {"dataset", "partition", "macro_f1"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"missing comparison columns: {sorted(missing)}")
    selected = metrics.loc[
        metrics["partition"].astype(str).eq("test")
        & metrics["dataset"].astype(str).ne("bbc_news_content")
    ].copy()
    if min_text_tokens is not None and "token_count" in selected.columns:
        selected = selected.loc[selected["token_count"] >= min_text_tokens]
    return selected.reset_index(drop=True)


def add_candidate_column(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add a stable configuration label used by comparison summaries."""
    dimensions = [
        column
        for column in (
            "scorer_llm",
            "feature_set",
            "training_regime",
            "training_window",
            "evaluation_window",
            "model_family",
        )
        if column in metrics.columns
    ]
    if not dimensions:
        raise ValueError("metrics has no configuration dimensions")
    result = metrics.copy()
    result["candidate"] = result[dimensions].fillna("NA").astype(str).agg(" | ".join, axis=1)
    return result


def rank_generalization_candidates(
    metrics: pd.DataFrame,
    semeval: pd.DataFrame,
    *,
    top_tier_gap: float,
) -> pd.DataFrame:
    if top_tier_gap < 0:
        raise ValueError("top_tier_gap must be non-negative")
    if metrics.empty:
        result = pd.DataFrame()
        result.attrs["evidence_status"] = "insufficient_binary_holdout"
        return result
    grouped = (
        metrics.groupby("candidate", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            worst_macro_f1=("macro_f1", "min"),
            dataset_count=("dataset", "nunique"),
        )
        .sort_values(
            ["mean_macro_f1", "worst_macro_f1", "dataset_count", "candidate"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    if not semeval.empty and "candidate" in semeval and "mean_absolute_token_error" in semeval:
        grouped = grouped.merge(
            semeval.groupby("candidate", as_index=False)["mean_absolute_token_error"].mean(),
            on="candidate",
            how="left",
        )
    else:
        grouped["mean_absolute_token_error"] = float("nan")
    top_mean = grouped.iloc[0]["mean_macro_f1"]
    grouped["top_tier"] = grouped["mean_macro_f1"] >= top_mean - top_tier_gap
    grouped["rank"] = range(1, len(grouped) + 1)
    grouped.attrs["evidence_status"] = "available"
    return grouped


def _empty_figure(title: str) -> plt.Figure:
    figure, axis = plt.subplots()
    axis.set_title(title)
    return figure


def plot_generalization_leaderboard(ranking: pd.DataFrame, *, title: str) -> plt.Figure:
    if ranking.empty:
        return _empty_figure(title)
    figure, axis = plt.subplots(figsize=(10, 5))
    ordered = ranking.sort_values("rank")
    axis.bar(ordered["candidate"].astype(str), ordered["mean_macro_f1"])
    axis.set_ylabel("Mean held-out macro-F1")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_dataset_heatmap(metrics: pd.DataFrame, *, title: str, limit: int = 15) -> plt.Figure:
    if metrics.empty:
        return _empty_figure(title)
    plot_metrics = metrics if "candidate" in metrics.columns else add_candidate_column(metrics)
    table = plot_metrics.pivot_table(index="candidate", columns="dataset", values="macro_f1", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(table.iloc[:limit].to_numpy(), aspect="auto", vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_yticks(range(len(table.iloc[:limit])), table.iloc[:limit].index.astype(str))
    axis.set_xticks(range(len(table.columns)), table.columns.astype(str), rotation=45, ha="right")
    figure.colorbar(image, ax=axis, label="Macro-F1")
    figure.tight_layout()
    return figure


def plot_f1_mae_frontier(ranking: pd.DataFrame, *, title: str) -> plt.Figure:
    if ranking.empty:
        return _empty_figure(title)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(ranking["mean_absolute_token_error"], ranking["mean_macro_f1"])
    for _, row in ranking.iterrows():
        axis.annotate(str(row["candidate"]), (row["mean_absolute_token_error"], row["mean_macro_f1"]))
    axis.set_xlabel("SemEval boundary MAE")
    axis.set_ylabel("Mean held-out macro-F1")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_parameter_effects(metrics: pd.DataFrame, *, parameter: str, title: str) -> plt.Figure:
    if metrics.empty or parameter not in metrics:
        return _empty_figure(title)
    summary = metrics.groupby(parameter, as_index=False)["macro_f1"].mean()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(summary[parameter].astype(str), summary["macro_f1"])
    axis.set_ylabel("Mean macro-F1")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_safety_frontier(ranking: pd.DataFrame, safety: pd.DataFrame, *, title: str) -> plt.Figure:
    if ranking.empty:
        return _empty_figure(title)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(ranking["worst_macro_f1"], ranking["mean_macro_f1"])
    axis.set_xlabel("Worst held-out macro-F1")
    axis.set_ylabel("Mean held-out macro-F1")
    axis.set_title(title)
    figure.tight_layout()
    return figure
