from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MATCH_COLUMNS = (
    "scorer_llm",
    "model_family",
    "training_regime",
    "training_window",
    "evaluation_window",
    "dataset",
    "partition",
)

DEFAULT_MODEL_SIZES_BILLIONS = {
    "gemma_3_270m_it": 0.27,
    "smollm2_360m": 0.36,
    "littlelamb": 0.3,
    "opt_350m": 0.35,
    "gpt2": 0.124,
    "gemma_3_1b_it": 1.0,
    "llama_3_2_1b": 1.0,
    "qwen3_5_0_8b": 0.8,
    "gemma_2_2b_it": 2.0,
    "llama_3_2_3b": 3.0,
    "ministral_3_3b": 3.0,
    "ministral_3_3b_reasoning": 3.0,
    "hymba_1_5b_instruct": 1.5,
    "phi_4_mini_instruct": 3.8,
    "phi_4_mini_reasoning": 3.8,
    "smollm2_1_7b": 1.7,
}

GPT_WHO_EXPERIMENT_BY_TRAINING_REGIME = {
    "Kaggle+MAGE+Outfox": "smollm_mixed",
    "Kaggle+MAGE+Outfox+BBC": "smollm_mixed_bbc",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"metrics are missing required columns: {missing}")


def matched_feature_set_deltas(
    metrics: pd.DataFrame,
    *,
    baseline_feature_set: str,
    comparison_feature_sets: Sequence[str],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    required = set(MATCH_COLUMNS) | {"feature_set", metric}
    _require_columns(metrics, required)
    comparisons = [name for name in dict.fromkeys(comparison_feature_sets) if name != baseline_feature_set]
    if not comparisons:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, "feature_set", "baseline_score", "comparison_score", "delta"])

    selected = metrics.loc[metrics["feature_set"].isin([baseline_feature_set, *comparisons])].copy()
    baseline = (
        selected.loc[selected["feature_set"].eq(baseline_feature_set)]
        .groupby(list(MATCH_COLUMNS), as_index=False)[metric]
        .mean()
        .rename(columns={metric: "baseline_score"})
    )
    frames: list[pd.DataFrame] = []
    for feature_set in comparisons:
        comparison = (
            selected.loc[selected["feature_set"].eq(feature_set)]
            .groupby(list(MATCH_COLUMNS), as_index=False)[metric]
            .mean()
            .rename(columns={metric: "comparison_score"})
        )
        if comparison.empty:
            continue
        matched = baseline.merge(comparison, on=list(MATCH_COLUMNS), how="inner")
        if matched.empty:
            continue
        matched["feature_set"] = feature_set
        matched["delta"] = matched["comparison_score"] - matched["baseline_score"]
        frames.append(matched)
    if not frames:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, "feature_set", "baseline_score", "comparison_score", "delta"])
    return pd.concat(frames, ignore_index=True)[
        [*MATCH_COLUMNS, "feature_set", "baseline_score", "comparison_score", "delta"]
    ]


def aggregate_feature_set_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    required = {"scorer_llm", "feature_set", "delta"}
    _require_columns(deltas, required)
    return (
        deltas.groupby(["scorer_llm", "feature_set"], as_index=False)["delta"]
        .agg(mean_delta="mean", std_delta="std", match_count="count")
        .sort_values(["feature_set", "mean_delta", "scorer_llm"], ascending=[True, False, True], ignore_index=True)
    )


def gpt_who_experiment_for_training_regime(training_regime: str) -> str | None:
    return GPT_WHO_EXPERIMENT_BY_TRAINING_REGIME.get(training_regime)


def append_external_feature_set_comparison(
    metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
    *,
    baseline_feature_set: str,
    external_feature_set: str,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Represent an external scorer as a feature-set comparison per scorer LLM."""
    _require_columns(metrics, {"scorer_llm", "feature_set", metric})
    _require_columns(external_metrics, {"scorer_llm", metric})
    baseline = (
        metrics.loc[metrics["feature_set"].eq(baseline_feature_set)]
        .groupby("scorer_llm", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "baseline_score"})
    )
    external = (
        external_metrics.groupby("scorer_llm", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "comparison_score"})
    )
    result = baseline.merge(external, on="scorer_llm", how="inner")
    result["feature_set"] = external_feature_set
    result["model_family"] = external_feature_set
    result["delta"] = result["comparison_score"] - result["baseline_score"]
    return result


def attach_model_sizes(
    summary: pd.DataFrame,
    *,
    model_sizes_billions: Mapping[str, float] = DEFAULT_MODEL_SIZES_BILLIONS,
) -> pd.DataFrame:
    _require_columns(summary, {"scorer_llm"})
    result = summary.copy()
    result["parameter_billion"] = result["scorer_llm"].map(model_sizes_billions)
    result["size_known"] = result["parameter_billion"].notna()
    return result


def plot_feature_set_comparison(
    deltas: pd.DataFrame, *, baseline_feature_set: str, title: str,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if deltas.empty:
        axis.text(0.5, 0.5, "No matched feature-set comparisons", ha="center", va="center")
        axis.set_axis_off()
        return figure
    group_columns = ["scorer_llm"]
    if "model_family" in deltas.columns:
        group_columns.append("model_family")
    summary = deltas.groupby([*group_columns, "feature_set"], as_index=False)[
        ["baseline_score", "comparison_score"]
    ].mean()
    configurations = summary[group_columns].drop_duplicates().sort_values(group_columns).reset_index(drop=True)
    positions = {tuple(row[column] for column in group_columns): index for index, row in configurations.iterrows()}
    feature_sets = sorted(summary["feature_set"].unique())
    feature_colors = dict(zip(feature_sets, sns.color_palette("colorblind", n_colors=len(feature_sets))))
    labeled_feature_sets: set[str] = set()
    for feature_set, frame in summary.groupby("feature_set", sort=True):
        for _, row in frame.iterrows():
            configuration = tuple(row[column] for column in group_columns)
            x = positions[configuration]
            axis.plot([x - 0.12, x + 0.12], [row["baseline_score"], row["comparison_score"]], color="0.7", zorder=1)
            axis.scatter(x - 0.12, row["baseline_score"], color="0.25", s=30, zorder=2)
            label = feature_set if feature_set not in labeled_feature_sets else "_nolegend_"
            axis.scatter(
                x + 0.12,
                row["comparison_score"],
                color=feature_colors[feature_set],
                s=35,
                label=label,
                zorder=2,
            )
            labeled_feature_sets.add(feature_set)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, title="Comparison feature set")
    if len(group_columns) == 1:
        tick_labels = configurations["scorer_llm"].tolist()
    else:
        tick_labels = [f"{row['scorer_llm']}\n{row['model_family']}" for _, row in configurations.iterrows()]
    axis.set_xticks(range(len(configurations)), tick_labels, rotation=30, ha="right")
    axis.set_ylabel("Mean score")
    axis.set_title(f"{title} ({baseline_feature_set} baseline)")
    return figure


def plot_feature_set_delta_heatmap(summary: pd.DataFrame, *, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if summary.empty:
        axis.text(0.5, 0.5, "No feature-set deltas", ha="center", va="center")
        axis.set_axis_off()
        return figure
    table = summary.pivot(index="scorer_llm", columns="feature_set", values="mean_delta")
    sns.heatmap(table, annot=True, fmt="+.3f", center=0, cmap="RdBu_r", ax=axis)
    axis.set_xlabel("Comparison feature set")
    axis.set_ylabel("Scorer LLM")
    axis.set_title(title)
    return figure


def plot_delta_by_model_size(summary: pd.DataFrame, *, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    known = summary.loc[summary["size_known"]].copy() if not summary.empty else summary
    if known.empty:
        axis.text(0.5, 0.5, "No known model sizes", ha="center", va="center")
        axis.set_axis_off()
        return figure
    sns.scatterplot(data=known, x="parameter_billion", y="mean_delta", hue="feature_set", style="feature_set", s=90, ax=axis)
    for _, row in known.iterrows():
        axis.annotate(str(row["scorer_llm"]), (row["parameter_billion"], row["mean_delta"]), xytext=(5, 5), textcoords="offset points")
    if len(known) >= 3:
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("Scorer model size (billion parameters)")
    axis.set_ylabel("Mean enriched-feature-set gain")
    axis.set_title(title)
    return figure
