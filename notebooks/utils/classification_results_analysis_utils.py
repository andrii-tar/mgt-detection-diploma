from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import seaborn as sns


_RESULT_TABLES = (
    "evaluation_metrics",
    "evaluation_coverage",
    "run_summary",
    "semeval_summary",
    "semeval_evaluations",
    "semeval_boundary_summary",
    "semeval_boundary_exclusions",
)
_METRIC_COLUMNS = (
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "hg_precision", "hg_recall", "hg_f1", "hg_support", "mg_precision", "mg_recall",
    "mg_f1", "mg_support", "tn", "fp", "fn", "tp", "false_positive_rate",
    "predicted_mg_rate",
)
_LEGACY_METADATA_COLUMNS = (
    "artifact_version", "cohort_seed", "created_at", "feature_set", "model_family",
    "artifact_run_id", "scorer_llm", "stride", "train_hg_rows", "train_mg_rows",
    "train_parent_count", "train_rows", "training_regime", "training_window",
    "validation_hg_rows", "validation_mg_rows", "validation_parent_count", "validation_rows",
    "evaluation_window", "evaluation_regime",
)
_LEGACY_EVALUATION_METRICS_COLUMNS = _LEGACY_METADATA_COLUMNS + ("dataset", "partition") + _METRIC_COLUMNS
_SEMEVAL_DIMENSION_COLUMNS = (
    "run_id", "scorer_llm", "feature_set", "training_regime", "training_window",
    "model_family", "artifact_path", "artifact_run_id", "artifact_created_at",
    "training_datasets", "cohort_seed", "stride", "train_row_count", "validation_row_count",
    "train_parent_count", "validation_parent_count", "train_hg_row_count", "train_mg_row_count",
    "validation_hg_row_count", "validation_mg_row_count", "evaluation_regime",
    "evaluation_dataset", "semeval_split",
)
_LEGACY_COLUMNS = {
    "evaluation_metrics": _LEGACY_EVALUATION_METRICS_COLUMNS,
    "evaluation_coverage": _LEGACY_METADATA_COLUMNS + (
        "dataset", "partition", "split", "class", "parent_count", "row_count",
        "availability_status", "rows", "hg_rows", "mg_rows",
    ),
    "semeval_summary": (
        "model", "window_length", "texts", "accuracy", "mg_precision", "mg_recall", "mg_f1",
        "boundary_error", "exact_region_match", "spurious_mg_region_count", "missed_mg_region_count",
    ) + _SEMEVAL_DIMENSION_COLUMNS,
    "semeval_evaluations": (
        "model", "window_length", "text_id", "token_count", "accuracy", "mg_precision",
        "mg_recall", "mg_f1", "predicted_mg_region_count", "spurious_mg_region_count",
        "missed_mg_region_count", "exact_region_match", "gold_token_boundary",
        "nearest_predicted_hg_to_mg_boundary", "boundary_error",
    ) + _SEMEVAL_DIMENSION_COLUMNS,
    "semeval_boundary_exclusions": ("text_id", "reason") + _SEMEVAL_DIMENSION_COLUMNS,
    "semeval_boundary_summary": (
        "model", "window_length", "texts", "mean_absolute_token_error",
        "median_absolute_token_error", "exact_accuracy", "within_1_accuracy",
        "within_3_accuracy", "within_5_accuracy", "early_rate", "late_rate",
        "no_crossing_rate",
    ) + _SEMEVAL_DIMENSION_COLUMNS,
}
_BINARY_DATASETS = {"kaggle_train", "mage_train", "outfox_chatgpt_train"}
_SINGLE_SOURCE_REGIMES = {
    "Kaggle": "kaggle_train",
    "MAGE": "mage_train",
    "Outfox": "outfox_chatgpt_train",
}
_CONFIGURATION_COLUMNS = (
    "scorer_llm", "model_family", "feature_set", "training_regime",
    "training_window", "evaluation_window",
)
_MATCH_COLUMNS = _CONFIGURATION_COLUMNS[:3] + ("training_window", "evaluation_window", "dataset", "partition")


@dataclass(frozen=True)
class ClassificationRunResults:
    run_id: str
    root: Path
    tables: dict[str, pd.DataFrame]
    provenance: pd.DataFrame


def _result_root(data_root: Path | str, run_id: str) -> Path:
    if not run_id:
        raise ValueError("run_id must not be empty")
    root = Path(data_root) / "experiment-results" / "classification" / "evaluations" / run_id
    if not root.is_dir():
        raise ValueError(f"result directory does not exist: {root}")
    return root


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle) if row]


def _read_result_table(path: Path, name: str) -> tuple[pd.DataFrame, str]:
    rows = _read_rows(path)
    if not rows:
        return pd.DataFrame(), "headered"
    header = rows[0]
    if "run_id" in header or "artifact_run_id" in header:
        return pd.read_csv(path), "headered"
    columns = _LEGACY_COLUMNS.get(name)
    data_rows = rows[1:] if columns is not None and len(header) != len(columns) else rows
    if columns is None or any(len(row) != len(columns) for row in data_rows):
        raise ValueError(f"unsupported legacy row width for {path}")
    return pd.DataFrame(data_rows, columns=columns), "legacy positional recovery"


def _filter_run(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    key = "artifact_run_id" if "artifact_run_id" in frame.columns else "run_id"
    if key not in frame.columns:
        raise ValueError("result table has neither artifact_run_id nor run_id")
    return frame.loc[frame[key].astype(str).eq(run_id)].reset_index(drop=True)


def _normalize_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = set(_METRIC_COLUMNS) | {
        "window_length", "texts", "boundary_error", "exact_region_match",
        "spurious_mg_region_count", "missed_mg_region_count",
        "mean_absolute_token_error", "median_absolute_token_error", "exact_accuracy",
        "within_1_accuracy", "within_3_accuracy", "within_5_accuracy", "early_rate",
        "late_rate", "no_crossing_rate",
    }
    result = frame.copy()
    for column in numeric_columns.intersection(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_classification_run(data_root: Path | str, *, run_id: str) -> ClassificationRunResults:
    root = _result_root(data_root, run_id)
    tables: dict[str, pd.DataFrame] = {}
    provenance_rows: list[dict[str, object]] = []
    for name in _RESULT_TABLES:
        path = root / f"{name}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame, source = _read_result_table(path, name)
        if frame.empty:
            continue
        filtered = _normalize_numeric_columns(_filter_run(frame, run_id))
        tables[name] = filtered
        provenance_rows.append({"table": name, "source": source, "rows": len(filtered)})
    if not tables:
        raise ValueError(f"no supported result exports found: {root}")
    return ClassificationRunResults(run_id, root, tables, pd.DataFrame(provenance_rows))


def available_table(results: ClassificationRunResults, name: str) -> pd.DataFrame:
    return results.tables.get(name, pd.DataFrame()).copy()


def document_metrics(results: ClassificationRunResults) -> pd.DataFrame:
    metrics = available_table(results, "evaluation_metrics")
    required = {"dataset", "partition", "macro_f1"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"evaluation_metrics is missing required columns: {missing}")
    return metrics.loc[
        metrics["partition"].astype(str).eq("test")
        & metrics["dataset"].astype(str).isin(_BINARY_DATASETS)
    ].copy()


def _macro_f1_summary(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return metrics.groupby(columns, dropna=False)["macro_f1"].agg(
        mean="mean", minimum="min", maximum="max", standard_deviation="std", count="count",
    ).reset_index().sort_values(["mean", "minimum"], ascending=False, ignore_index=True)


def configuration_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    return _macro_f1_summary(metrics, list(_CONFIGURATION_COLUMNS))


def aggregate_by_llm(metrics: pd.DataFrame) -> pd.DataFrame:
    return _macro_f1_summary(metrics, ["scorer_llm"])


def aggregate_by_dataset(metrics: pd.DataFrame) -> pd.DataFrame:
    return _macro_f1_summary(metrics, ["dataset"])


def transfer_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    expected = result["training_regime"].map(_SINGLE_SOURCE_REGIMES)
    result["is_in_distribution"] = expected.notna() & result["dataset"].eq(expected)
    result["is_eligible_transfer"] = ~result["is_in_distribution"]
    return result


def universal_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = transfer_scores(metrics).loc[lambda frame: frame["is_eligible_transfer"]]
    return eligible.groupby(list(_CONFIGURATION_COLUMNS), dropna=False)["macro_f1"].agg(
        worst_source_macro_f1="min",
        mean_source_macro_f1="mean",
        standard_deviation="std",
        eligible_source_count="count",
    ).reset_index().sort_values(
        ["worst_source_macro_f1", "mean_source_macro_f1"], ascending=False, ignore_index=True,
    )


def best_universal_by_llm(metrics: pd.DataFrame) -> pd.DataFrame:
    ranking = universal_ranking(metrics)
    return ranking.groupby("scorer_llm", sort=False, group_keys=False).head(1).reset_index(drop=True)


def _categorical_colors(values: pd.Series) -> dict[str, object]:
    labels = sorted({
        label
        for value in values.dropna()
        for label in (str(value),)
        if label.strip()
    })
    palette = plt.get_cmap("tab10")
    return {label: palette(index % palette.N) for index, label in enumerate(labels)}


def _scorer_llm_colors(values: pd.Series) -> dict[str, object]:
    return _categorical_colors(values)


def transfer_matrix(metrics: pd.DataFrame, configuration: Mapping[str, object]) -> pd.DataFrame:
    selected = metrics.copy()
    for column, value in configuration.items():
        selected = selected.loc[selected[column].eq(value)]
    return selected.pivot_table(
        index="training_regime", columns="dataset", values="macro_f1", aggfunc="mean",
    ).reindex(columns=sorted(_BINARY_DATASETS))


def pareto_universal_configurations(ranking: pd.DataFrame) -> pd.DataFrame:
    values = ranking[["worst_source_macro_f1", "mean_source_macro_f1"]].to_numpy()
    keep = [not any(
        (other[0] >= value[0] and other[1] >= value[1])
        and (other[0] > value[0] or other[1] > value[1])
        for other in values
    ) for value in values]
    return ranking.loc[keep].reset_index(drop=True)


def plot_transfer_heatmap(matrix: pd.DataFrame, *, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, max(3, len(matrix) * .7)), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row, index in enumerate(matrix.index):
        for column, name in enumerate(matrix.columns):
            value = matrix.loc[index, name]
            if pd.notna(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Test macro F1")
    axis.set(title=title, xlabel="Test dataset", ylabel="Training regime")
    return figure


def plot_universal_frontier(ranking: pd.DataFrame, *, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(ranking["mean_source_macro_f1"], ranking["worst_source_macro_f1"], color="0.65", label="All configurations")
    pareto = pareto_universal_configurations(ranking)
    axis.scatter(pareto["mean_source_macro_f1"], pareto["worst_source_macro_f1"], color="C1", label="Pareto frontier")
    for _, row in pareto.iterrows():
        axis.annotate(
            str(row["scorer_llm"]),
            (row["mean_source_macro_f1"], row["worst_source_macro_f1"]),
            xytext=(4, 4), textcoords="offset points",
        )
    axis.set(title=title, xlabel="Mean eligible test macro F1", ylabel="Worst eligible test macro F1")
    axis.legend()
    return figure


def plot_stability_profiles(scores: pd.DataFrame, best_by_llm: pd.DataFrame, *, title: str) -> plt.Figure:
    configuration_columns = [column for column in _CONFIGURATION_COLUMNS if column in scores and column in best_by_llm]
    selected = scores.merge(best_by_llm[configuration_columns].drop_duplicates(), on=configuration_columns, how="inner")
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = _scorer_llm_colors(selected["scorer_llm"]) if "scorer_llm" in selected else {}
    for _, configuration_frame in selected.groupby(configuration_columns, dropna=False):
        configuration = configuration_frame.iloc[0]
        label = " | ".join(str(configuration[column]) for column in ("scorer_llm", "training_regime") if column in configuration)
        profile = configuration_frame.groupby("dataset", dropna=False)["macro_f1"].mean().reindex(sorted(_BINARY_DATASETS))
        axis.plot(profile.index, profile.values, marker="o", label=label, color=colors.get(str(configuration["scorer_llm"])) if "scorer_llm" in configuration else None)
    axis.set(title=title, xlabel="Test dataset", ylabel="Test macro F1", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    if not selected.empty:
        axis.legend(title="Best universal by LLM")
    return figure


def plot_dimension_stability_profiles(
    scores: pd.DataFrame,
    best_by_dimension: pd.DataFrame,
    *,
    dimension: str,
    title: str,
) -> plt.Figure:
    if dimension not in scores.columns or dimension not in best_by_dimension.columns:
        raise ValueError(f"dimension {dimension!r} must exist in both input frames")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    required = {"dataset", "macro_f1"}
    if required.issubset(scores.columns):
        configuration_columns = [
            column for column in _CONFIGURATION_COLUMNS
            if column in scores.columns and column in best_by_dimension.columns
        ]
        usable_scores = scores.loc[scores[dimension].notna()].copy()
        usable_best = best_by_dimension.loc[best_by_dimension[dimension].notna()].copy()
        usable_scores = usable_scores.loc[usable_scores[dimension].astype(str).str.strip().ne("")]
        usable_best = usable_best.loc[usable_best[dimension].astype(str).str.strip().ne("")]
        if configuration_columns and not usable_scores.empty and not usable_best.empty:
            selected = usable_scores.merge(
                usable_best[configuration_columns].drop_duplicates(),
                on=configuration_columns,
                how="inner",
            )
            labels = sorted(selected[dimension].astype(str).unique())
            colors = _categorical_colors(selected[dimension])
            for label in labels:
                points = selected.loc[selected[dimension].astype(str).eq(label)]
                profile = points.groupby("dataset", dropna=False)["macro_f1"].mean().reindex(sorted(_BINARY_DATASETS))
                profile = profile.dropna()
                axis.plot(
                    profile.index,
                    profile.values,
                    marker="o",
                    label=label,
                    color=colors[label],
                )
            if labels:
                axis.legend(title=dimension)
    axis.set(title=title, xlabel="Test dataset", ylabel="Test macro F1", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    return figure


def plot_transfer_asymmetry(matrix: pd.DataFrame, *, title: str) -> plt.Figure:
    sources = (
        ("Kaggle", "kaggle_train"),
        ("MAGE", "mage_train"),
        ("Outfox", "outfox_chatgpt_train"),
    )
    positions = {name: (0.0, 1.0) if index == 0 else (-.87, -.5) if index == 1 else (.87, -.5) for index, (name, _) in enumerate(sources)}
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=0, vmax=1)
    for name, _ in sources:
        x, y = positions[name]
        axis.scatter(x, y, s=900, color="0.9", edgecolor="black", zorder=2)
        axis.text(x, y, name, ha="center", va="center", zorder=3)
    for source, _ in sources:
        for target, dataset in sources:
            if source == target or source not in matrix.index or dataset not in matrix.columns:
                continue
            score = matrix.loc[source, dataset]
            if pd.isna(score):
                continue
            arrow = FancyArrowPatch(
                positions[source], positions[target], arrowstyle="->", mutation_scale=14,
                linewidth=1 + 5 * score, color=color_map(normalizer(score)),
                connectionstyle="arc3,rad=0.16", zorder=1,
            )
            axis.add_patch(arrow)
    axis.set(title=title, xlim=(-1.3, 1.3), ylim=(-.9, 1.3))
    axis.set_axis_off()
    return figure


def plot_semeval_tradeoff(region_ranking: pd.DataFrame, boundary_ranking: pd.DataFrame, *, title: str) -> plt.Figure:
    match_columns = [
        column for column in ("run_id", *_CONFIGURATION_COLUMNS)
        if column in region_ranking and column in boundary_ranking
    ]
    region = region_ranking[[*match_columns, "mg_f1"]]
    boundary = boundary_ranking[[*match_columns, "mean_absolute_token_error"]]
    joined = region.merge(boundary, on=match_columns, how="inner")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = _scorer_llm_colors(joined["scorer_llm"]) if "scorer_llm" in joined else {}
    for scorer_llm, points in joined.groupby("scorer_llm", sort=False, dropna=False):
        label = str(scorer_llm)
        axis.scatter(
            points["mean_absolute_token_error"], points["mg_f1"],
            color=colors.get(label), label=label,
        )
    axis.set(title=title, xlabel="Mean absolute token error", ylabel="MG F1", ylim=(0, 1))
    if colors:
        axis.legend(title="Scorer LLM")
    return figure


def plot_dimension_semeval_tradeoff(
    region_ranking: pd.DataFrame,
    boundary_ranking: pd.DataFrame,
    *,
    dimension: str,
    title: str,
) -> plt.Figure:
    if dimension not in region_ranking.columns or dimension not in boundary_ranking.columns:
        raise ValueError(f"dimension {dimension!r} must exist in both input frames")

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    required_region = {"mg_f1"}
    required_boundary = {"mean_absolute_token_error"}
    if required_region.issubset(region_ranking.columns) and required_boundary.issubset(boundary_ranking.columns):
        match_columns = [
            column for column in _CONFIGURATION_COLUMNS
            if column in region_ranking.columns and column in boundary_ranking.columns
        ]
        usable_region = region_ranking.loc[region_ranking[dimension].notna()].copy()
        usable_boundary = boundary_ranking.loc[boundary_ranking[dimension].notna()].copy()
        usable_region = usable_region.loc[usable_region[dimension].astype(str).str.strip().ne("")]
        usable_boundary = usable_boundary.loc[usable_boundary[dimension].astype(str).str.strip().ne("")]
        if match_columns and not usable_region.empty and not usable_boundary.empty:
            joined = usable_region[match_columns + ["mg_f1"]].merge(
                usable_boundary[match_columns + ["mean_absolute_token_error"]],
                on=match_columns,
                how="inner",
            )
            labels = sorted(joined[dimension].astype(str).unique())
            colors = _categorical_colors(joined[dimension])
            for label in labels:
                points = joined.loc[joined[dimension].astype(str).eq(label)]
                axis.scatter(
                    points["mean_absolute_token_error"],
                    points["mg_f1"],
                    color=colors[label],
                    label=label,
                )
            if labels:
                axis.legend(title=dimension)
    axis.set(title=title, xlabel="Mean absolute token error", ylabel="MG F1", ylim=(0, 1))
    return figure


def mixed_bbc_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["training_regime"].isin(["Mixed", "Mixed+BBC"])].copy()
    mixed = selected.loc[selected["training_regime"].eq("Mixed"), [*_MATCH_COLUMNS, "macro_f1"]]
    mixed_bbc = selected.loc[selected["training_regime"].eq("Mixed+BBC"), [*_MATCH_COLUMNS, "macro_f1"]]
    merged = mixed.merge(mixed_bbc, on=list(_MATCH_COLUMNS), how="inner", suffixes=("_mixed", "_mixed_bbc"))
    merged = merged.rename(columns={"macro_f1_mixed": "mixed_macro_f1", "macro_f1_mixed_bbc": "mixed_bbc_macro_f1"})
    return merged.assign(macro_f1_delta=merged["mixed_bbc_macro_f1"] - merged["mixed_macro_f1"])


def bbc_safety_metrics(results: ClassificationRunResults) -> pd.DataFrame:
    metrics = available_table(results, "evaluation_metrics")
    if not {"dataset", "partition"}.issubset(metrics.columns):
        return pd.DataFrame()
    return metrics.loc[
        metrics["partition"].astype(str).eq("test")
        & metrics["dataset"].astype(str).eq("bbc_news_content")
    ].copy()


def semeval_region_ranking(results: ClassificationRunResults) -> pd.DataFrame:
    summary = available_table(results, "semeval_summary")
    if "mg_f1" not in summary:
        return pd.DataFrame(columns=["mg_f1"])
    return summary.sort_values("mg_f1", ascending=False, ignore_index=True)


def semeval_boundary_ranking(results: ClassificationRunResults) -> pd.DataFrame:
    summary = available_table(results, "semeval_boundary_summary")
    if "mean_absolute_token_error" not in summary:
        return pd.DataFrame(columns=["mean_absolute_token_error"])
    return summary.sort_values("mean_absolute_token_error", ascending=True, ignore_index=True)


def semeval_exclusions(results: ClassificationRunResults) -> pd.DataFrame:
    return available_table(results, "semeval_boundary_exclusions")


def plot_aggregate_bars(summary: pd.DataFrame, *, category: str, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(max(6, len(summary) * 1.2), 4), constrained_layout=True)
    sns.barplot(data=summary, x=category, y="mean", ax=axis)
    axis.set(title=title, xlabel=category.replace("_", " "), ylabel="Mean test macro F1", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    return figure


def plot_feature_set_heatmap(metrics: pd.DataFrame, *, title: str) -> plt.Figure:
    pivot = metrics.pivot_table(index="feature_set", columns="scorer_llm", values="macro_f1", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(3, len(pivot) * 0.7)), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis", ax=axis)
    axis.set(title=title, xlabel="Scorer LLM", ylabel="Feature set")
    return figure


def plot_window_heatmap(metrics: pd.DataFrame, *, title: str) -> plt.Figure:
    pivot = metrics.pivot_table(index="training_window", columns="evaluation_window", values="macro_f1", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(3, len(pivot) * 0.7)), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis", ax=axis)
    axis.set(title=title, xlabel="Evaluation window", ylabel="Training window")
    return figure


def plot_regime_deltas(comparison: pd.DataFrame, *, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(max(7, len(comparison) * 0.25), 4), constrained_layout=True)
    sns.barplot(data=comparison, x=comparison.index, y="macro_f1_delta", ax=axis)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(title=title, xlabel="Matched configuration", ylabel="Macro F1 delta (Mixed+BBC - Mixed)")
    return figure


def plot_semeval_ranking(ranking: pd.DataFrame, *, metric: str, title: str, ascending: bool = False) -> plt.Figure:
    plotted = ranking.sort_values(metric, ascending=ascending).head(20).copy()
    plotted["configuration"] = plotted[[column for column in ("scorer_llm", "feature_set", "training_window") if column in plotted]].astype(str).agg(" | ".join, axis=1)
    figure, axis = plt.subplots(figsize=(10, max(3, len(plotted) * 0.35)), constrained_layout=True)
    sns.barplot(data=plotted, y="configuration", x=metric, ax=axis)
    axis.set(title=title, xlabel=metric.replace("_", " "), ylabel="Configuration")
    return figure
