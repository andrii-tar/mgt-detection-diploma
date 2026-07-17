from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm


DEFAULT_METRICS = [
    "prompt_mu_score",
    "prompt_sigma_score",
    "perplexity",
    "num_tokens",
    "text_len_words",
]


def parse_experiment_name(experiment_name: str) -> tuple[str, str]:
    if "__" not in experiment_name:
        return experiment_name, "unknown"
    dataset, model = experiment_name.split("__", maxsplit=1)
    return dataset, model


def discover_metric_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.glob("*/*/metrics/metrics.parquet"))


def load_metrics(processed_root: Path) -> pd.DataFrame:
    frames = []
    for path in discover_metric_files(processed_root):
        experiment_name = path.parents[2].name
        split = path.parents[1].name
        dataset, model = parse_experiment_name(experiment_name)
        frame = pd.read_parquet(path)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "model", model)
        frame.insert(2, "experiment_name", experiment_name)
        frame["split"] = frame.get("split", split).fillna(split)
        frame["metrics_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metrics files found under {processed_root}")
    return pd.concat(frames, ignore_index=True)


def metric_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["dataset", "model", "split"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["dataset", "model", "split"])
    )


def filter_metrics(
    metrics_df: pd.DataFrame,
    selected_datasets: list[str] | None = None,
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    plot_df = metrics_df.copy()
    if selected_datasets is not None:
        plot_df = plot_df[plot_df["dataset"].isin(selected_datasets)]
    if selected_models is not None:
        plot_df = plot_df[plot_df["model"].isin(selected_models)]
    return plot_df


def available_metric_columns(metrics_df: pd.DataFrame) -> list[str]:
    candidates = [
        "prompt_mu_score",
        "prompt_sigma_score",
        "perplexity",
        "num_tokens",
        "text_len_symbols",
        "text_len_words",
        "text_len_sentences",
    ]
    return [metric for metric in candidates if metric in metrics_df.columns]


def plot_metric_distributions(data: pd.DataFrame, metric: str, *, bins: int = 50) -> None:
    grid = sns.displot(
        data=data,
        x=metric,
        hue="split",
        row="dataset",
        col="model",
        kind="hist",
        bins=bins,
        stat="density",
        common_norm=False,
        element="step",
        height=3.0,
        aspect=1.25,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.set_axis_labels(metric, "density")
    grid.figure.suptitle(f"Distribution of {metric} by dataset/model/split", y=1.03)
    plt.show()


def plot_experiment_metric_row(data: pd.DataFrame, dataset: str, model: str, metrics: list[str]) -> None:
    experiment_df = data[(data["dataset"] == dataset) & (data["model"] == model)]
    if experiment_df.empty:
        raise ValueError(f"No rows for dataset={dataset!r}, model={model!r}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        sns.histplot(
            data=experiment_df,
            x=metric,
            hue="split",
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            ax=axis,
        )
        axis.set_title(metric)
        axis.set_ylabel("density")
    fig.suptitle(f"{dataset} / {model}", y=1.08)
    plt.show()


def discover_similarity_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.glob("*/*/token_similarities/*.parquet"))


def token_similarity_stats(row: pd.Series) -> dict[str, float | int | str | None]:
    selected_similarities = []
    token_candidate_means = []
    token_candidate_stds = []
    for token in row["token_similarities"]:
        top_logprobs = token.get("top_logprobs")
        if top_logprobs is None or len(top_logprobs) == 0:
            continue
        selected_similarity = top_logprobs[0].get("similarity_to_gt")
        if selected_similarity is not None:
            selected_similarities.append(float(selected_similarity))
        candidate_similarities = [
            float(candidate["similarity_to_gt"])
            for candidate in top_logprobs
            if candidate.get("similarity_to_gt") is not None
        ]
        if candidate_similarities:
            series = pd.Series(candidate_similarities, dtype="float64")
            token_candidate_means.append(float(series.mean()))
            token_candidate_stds.append(float(series.std(ddof=0)))

    stats = {
        "id": int(row["id"]),
        "dataset": row["dataset"],
        "model": row["model"],
        "experiment_name": row["experiment_name"],
        "split": row["split"],
        "selected_similarity_count": len(selected_similarities),
        "selected_similarity_mean": None,
        "selected_similarity_std": None,
        "candidate_similarity_token_count": len(token_candidate_means),
        "candidate_similarity_mean": None,
        "candidate_similarity_std": None,
    }
    if selected_similarities:
        selected_series = pd.Series(selected_similarities, dtype="float64")
        stats["selected_similarity_mean"] = float(selected_series.mean())
        stats["selected_similarity_std"] = float(selected_series.std(ddof=0))
    if token_candidate_means:
        stats["candidate_similarity_mean"] = float(pd.Series(token_candidate_means, dtype="float64").mean())
        stats["candidate_similarity_std"] = float(pd.Series(token_candidate_stds, dtype="float64").mean())
    return stats


def load_token_similarity_stats(processed_root: Path) -> pd.DataFrame:
    rows = []
    paths = discover_similarity_files(processed_root)
    for path in tqdm(paths, desc="Similarity files", unit="file"):
        frame = pd.read_parquet(path)
        row_iter = frame.iterrows()
        rows.extend(
            token_similarity_stats(row)
            for _, row in tqdm(row_iter, total=len(frame), desc=path.parent.parent.name, unit="text", leave=False)
        )
    if not rows:
        raise FileNotFoundError(f"No token similarity files found under {processed_root}")
    return pd.DataFrame(rows)


def _similarity_plot_specs() -> list[dict[str, str]]:
    return [
        {
            "x": "selected_similarity_mean",
            "y": "selected_similarity_std",
            "title": "Top-1 selected token",
            "xlabel": "Mean top-1 similarity",
            "ylabel": "Std top-1 similarity",
        },
        {
            "x": "candidate_similarity_mean",
            "y": "candidate_similarity_std",
            "title": "Top-k candidate tokens",
            "xlabel": "Mean token candidate similarity",
            "ylabel": "Mean token candidate similarity std",
        },
    ]


def plot_similarity_scatter_pair(data: pd.DataFrame, *, alpha: float = 0.65) -> None:
    for dataset, model in data[["dataset", "model"]].drop_duplicates().itertuples(index=False):
        experiment_df = data[(data["dataset"] == dataset) & (data["model"] == model)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, spec in zip(axes, _similarity_plot_specs()):
            scatter_df = experiment_df.dropna(subset=[spec["x"], spec["y"]])
            sns.scatterplot(
                data=scatter_df,
                x=spec["x"],
                y=spec["y"],
                hue="split",
                alpha=alpha,
                s=35,
                edgecolor=None,
                ax=axis,
            )
            axis.set_title(spec["title"])
            axis.set_xlabel(spec["xlabel"])
            axis.set_ylabel(spec["ylabel"])
            axis.legend(title="split")
        fig.suptitle(f"Token similarity scatter: {dataset} / {model}", y=1.04)
        plt.show()


def plot_aggregate_similarity_scatter(data: pd.DataFrame, group_column: str, *, alpha: float = 0.65) -> None:
    for group_value in sorted(data[group_column].dropna().unique()):
        group_df = data[data[group_column] == group_value]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, spec in zip(axes, _similarity_plot_specs()):
            scatter_df = group_df.dropna(subset=[spec["x"], spec["y"]])
            sns.scatterplot(
                data=scatter_df,
                x=spec["x"],
                y=spec["y"],
                hue="split",
                alpha=alpha,
                s=35,
                edgecolor=None,
                ax=axis,
            )
            axis.set_title(spec["title"])
            axis.set_xlabel(spec["xlabel"])
            axis.set_ylabel(spec["ylabel"])
            axis.legend(title="split")
        fig.suptitle(f"Token similarity scatter aggregated by {group_column}: {group_value}", y=1.04)
        plt.show()


def split_pair_palette(data: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    palette = {}
    base_palettes = {"HG": "Blues", "MG": "Oranges"}
    for split, split_df in data.groupby("split"):
        pairs = sorted(split_df["dataset_model"].dropna().unique())
        cmap_name = base_palettes.get(split, "Greys")
        colors = sns.color_palette(cmap_name, n_colors=max(len(pairs) + 2, 3))[1 : len(pairs) + 1]
        for pair, color in zip(pairs, colors):
            palette[f"{split} | {pair}"] = color
    return palette


def plot_global_similarity_scatter(data: pd.DataFrame, *, alpha: float = 0.65) -> None:
    plot_df = data.copy()
    plot_df["dataset_model"] = plot_df["dataset"] + " / " + plot_df["model"]
    plot_df["split_dataset_model"] = plot_df["split"] + " | " + plot_df["dataset_model"]
    palette = split_pair_palette(plot_df)
    for spec in _similarity_plot_specs():
        scatter_df = plot_df.dropna(subset=[spec["x"], spec["y"]])
        fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        sns.scatterplot(
            data=scatter_df,
            x=spec["x"],
            y=spec["y"],
            hue="split_dataset_model",
            palette=palette,
            alpha=alpha,
            s=35,
            edgecolor=None,
            ax=axis,
        )
        axis.set_title(f"{spec['title']}: all datasets and models")
        axis.set_xlabel(spec["xlabel"])
        axis.set_ylabel(spec["ylabel"])
        axis.legend(title="split | dataset / model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.show()
