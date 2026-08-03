from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from scripts.logprob_metrics import compute_logprob_metrics


DEFAULT_METRICS = [
    "prompt_mu_score",
    "prompt_sigma_score",
    "perplexity",
    "num_tokens",
    "text_len_words",
    "d_score_top20",
    "d_score_top20_coverage",
]

TOKEN_DERIVED_METRICS = [
    "prompt_mu_score",
    "prompt_sigma_score",
    "perplexity",
    "num_tokens",
    "d_score_top20",
    "d_score_top20_coverage",
]


def iter_token_lengths(token_count: int, step: int = 50) -> range:
    if step <= 0:
        raise ValueError("step must be positive")
    return range(step, token_count + 1, step)


def compute_prefix_metrics(row: dict, *, step: int = 50) -> list[dict]:
    records = []
    tokens = list(row.get("tokens", []))
    for token_length in iter_token_lengths(len(tokens), step):
        prefix_row = {**row, "tokens": tokens[:token_length]}
        metrics = compute_logprob_metrics(prefix_row)
        metrics["token_length"] = token_length
        records.append(metrics)
    return records


def parse_experiment_name(experiment_name: str) -> tuple[str, str]:
    if "__" not in experiment_name:
        return experiment_name, "unknown"
    dataset, model = experiment_name.split("__", maxsplit=1)
    return dataset, model


def load_logprob_rows(
    processed_root: Path,
    selected_datasets: list[str] | None = None,
    selected_models: list[str] | None = None,
) -> list[dict]:
    rows = []
    for path in sorted(processed_root.glob("*/*/data/*.parquet")):
        experiment_name = path.parents[2].name
        split = path.parents[1].name
        dataset, model = parse_experiment_name(experiment_name)
        if selected_datasets is not None and dataset not in selected_datasets:
            continue
        if selected_models is not None and model not in selected_models:
            continue
        for row in pd.read_parquet(path).to_dict(orient="records"):
            row["dataset"] = dataset
            row["model"] = model
            if row.get("split") is None or pd.isna(row.get("split")):
                row["split"] = split
            rows.append(row)
    return rows


def default_length_metrics_path(processed_root: Path) -> Path:
    return processed_root / "length_metric_distributions" / "metrics.parquet"


def save_length_metric_table(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, compression="zstd", index=False)
    return output_path


def load_length_metric_table(output_path: Path) -> pd.DataFrame:
    return pd.read_parquet(output_path)


def compute_length_metric_table(
    processed_root: Path,
    *,
    step: int = 50,
    selected_datasets: list[str] | None = None,
    selected_models: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    columns = ["dataset", "model", "split", "token_length", *TOKEN_DERIVED_METRICS]
    records = []
    rows = load_logprob_rows(processed_root, selected_datasets, selected_models)
    source_rows = tqdm(rows, desc="Computing prefix metrics", unit="row") if show_progress else rows
    for row in source_rows:
        for metrics in compute_prefix_metrics(row, step=step):
            records.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "split": row.get("split"),
                    "token_length": metrics["token_length"],
                    **{metric: metrics[metric] for metric in TOKEN_DERIVED_METRICS},
                }
            )
    return pd.DataFrame(records, columns=columns)


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
        "d_score_top20",
        "d_score_top20_coverage",
    ]
    return [metric for metric in candidates if metric in metrics_df.columns]


def split_metric_summary(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = data.groupby(["dataset", "model", "split"], dropna=False)[metrics]
    summary = grouped.agg(["count", "mean", "median", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.sort_values(["dataset", "model", "split"])


def plot_split_metric_comparison(data: pd.DataFrame, metrics: list[str]) -> None:
    pairs = data[["dataset", "model"]].drop_duplicates().sort_values(["dataset", "model"])
    split_palette = {"HG": "#1f77b4", "MG": "#ff7f0e"}
    fig, axes = plt.subplots(
        len(metrics),
        len(pairs),
        figsize=(4.2 * len(pairs), 3.2 * len(metrics)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, metric in enumerate(metrics):
        metric_available = metric in data.columns
        for column, (dataset, model) in enumerate(pairs.itertuples(index=False)):
            axis = axes[row, column]
            if not metric_available:
                axis.set_title(f"{metric} (missing)")
                axis.set_axis_off()
                continue

            experiment_df = data[(data["dataset"] == dataset) & (data["model"] == model)]
            sns.histplot(
                data=experiment_df,
                x=metric,
                hue="split",
                palette=split_palette,
                bins=50,
                stat="density",
                common_norm=False,
                element="step",
                ax=axis,
            )
            axis.set_title(f"{dataset} / {model}: {metric}")
            axis.set_ylabel("density")

    fig.suptitle("HG/MG metric comparison by dataset/model", y=1.02)
    plt.show()


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


def plot_length_metric_distributions(data: pd.DataFrame, metric: str, *, bins: int = 50) -> None:
    if metric not in data.columns:
        raise ValueError(f"Metric column not found: {metric}")

    split_palette = {"HG": "#1f77b4", "MG": "#ff7f0e"}
    pairs = data[["dataset", "model"]].drop_duplicates().sort_values(["dataset", "model"])
    for dataset, model in pairs.itertuples(index=False):
        experiment_df = data[(data["dataset"] == dataset) & (data["model"] == model)]
        figure = plt.figure(figsize=(10, 7), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        for split in sorted(experiment_df["split"].dropna().unique()):
            split_df = experiment_df[experiment_df["split"] == split]
            for token_length, length_df in split_df.groupby("token_length", sort=True):
                values = pd.to_numeric(length_df[metric], errors="coerce").to_numpy(dtype="float64")
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                density, edges = np.histogram(values, bins=bins, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                axis.plot(
                    centers,
                    np.full(centers.shape, token_length),
                    density,
                    color=split_palette.get(split, "#7f7f7f"),
                    alpha=0.8,
                )

        handles = [
            plt.Line2D([], [], color=color, label=split)
            for split, color in split_palette.items()
            if split in experiment_df["split"].values
        ]
        if handles:
            axis.legend(handles=handles, title="split")
        axis.set_title(f"{metric} distribution by token length: {dataset} / {model}")
        axis.set_xlabel(metric)
        axis.set_ylabel("token length")
        axis.set_zlabel("density")
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


def discover_token_similarity_stats_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.glob("*/*/token_similarity_stats/token_similarity_stats.parquet"))


def load_token_similarity_stats(processed_root: Path) -> pd.DataFrame:
    paths = discover_token_similarity_stats_files(processed_root)
    if not paths:
        raise FileNotFoundError(f"No token similarity stats files found under {processed_root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


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


def plot_similarity_kde_pair(data: pd.DataFrame, *, fill: bool = False, thresh: float = 0.05) -> None:
    split_palette = {"HG": "#1f77b4", "MG": "#ff7f0e"}
    for dataset, model in data[["dataset", "model"]].drop_duplicates().itertuples(index=False):
        experiment_df = data[(data["dataset"] == dataset) & (data["model"] == model)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, spec in zip(axes, _similarity_plot_specs()):
            kde_df = experiment_df.dropna(subset=[spec["x"], spec["y"]])
            if kde_df.empty:
                axis.set_title(spec["title"])
                axis.set_xlabel(spec["xlabel"])
                axis.set_ylabel(spec["ylabel"])
                continue

            sns.kdeplot(
                data=kde_df,
                x=spec["x"],
                y=spec["y"],
                hue="split",
                palette={split: split_palette.get(split, "#7f7f7f") for split in sorted(kde_df["split"].dropna().unique())},
                common_norm=False,
                fill=fill,
                thresh=thresh,
                ax=axis,
            )
            axis.set_title(spec["title"])
            axis.set_xlabel(spec["xlabel"])
            axis.set_ylabel(spec["ylabel"])
        fig.suptitle(f"Token similarity KDE: {dataset} / {model}", y=1.04)
        plt.show()


def plot_token_similarity(
    data: pd.DataFrame,
    *,
    plot_type: str,
    selected_datasets: list[str] | None = None,
    selected_models: list[str] | None = None,
) -> None:
    plot_df = filter_metrics(data, selected_datasets, selected_models)
    if plot_df.empty:
        raise ValueError("No token similarity rows match selected datasets/models")

    if plot_type == "scatter":
        plot_similarity_scatter_pair(plot_df)
        return
    if plot_type == "kde":
        plot_similarity_kde_pair(plot_df)
        return
    raise ValueError("plot_type must be one of: scatter, kde")


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


def plot_hg_metric_mean_std_kde_by_model(
    metrics_df: pd.DataFrame,
    similarity_stats_df: pd.DataFrame,
    *,
    split: str = "HG",
    fill: bool = False,
    thresh: float = 0.05,
) -> None:
    specs = [
        {
            "title": "Token similarity",
            "data": similarity_stats_df,
            "mean_column": "selected_similarity_mean",
            "std_column": "selected_similarity_std",
        },
        {"title": "Prompt mu", "data": metrics_df, "mean_column": "prompt_mu_score", "std_column": "prompt_sigma_score"},
        {"title": "Prompt sigma", "data": metrics_df, "mean_column": "prompt_sigma_score", "std_column": "prompt_mu_score"},
        {"title": "Perplexity", "data": metrics_df, "mean_column": "perplexity", "std_column": "prompt_sigma_score"},
        {
            "title": "D-score vs coverage",
            "data": metrics_df,
            "mean_column": "d_score_top20",
            "std_column": "d_score_top20_coverage",
        },
        {
            "title": "D-score vs prompt mu",
            "data": metrics_df,
            "mean_column": "d_score_top20",
            "std_column": "prompt_mu_score",
        },
        {
            "title": "Coverage vs prompt mu",
            "data": metrics_df,
            "mean_column": "d_score_top20_coverage",
            "std_column": "prompt_mu_score",
        },
    ]
    models = sorted(set(metrics_df["model"].dropna().unique()) | set(similarity_stats_df["model"].dropna().unique()))

    for model in models:
        fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4.5), constrained_layout=True)
        for axis, spec in zip(axes, specs):
            source_df = spec["data"]
            model_df = source_df[(source_df["split"] == split) & (source_df["model"] == model)]
            columns = [spec["mean_column"], spec["std_column"]]
            model_df = model_df.dropna(subset=columns)
            if model_df.empty:
                axis.set_title(spec["title"])
                axis.set_xlabel("mean")
                axis.set_ylabel("std")
                continue

            plot_df = model_df.rename(columns={spec["mean_column"]: "mean", spec["std_column"]: "std"})
            sns.kdeplot(
                data=plot_df,
                x="mean",
                y="std",
                hue="dataset",
                common_norm=False,
                fill=fill,
                thresh=thresh,
                warn_singular=False,
                ax=axis,
            )
            axis.set_title(spec["title"])
            axis.set_xlabel("mean")
            axis.set_ylabel("std")
        fig.suptitle(f"{split} metric mean/std KDE by dataset: {model}", y=1.04)
        plt.show()
