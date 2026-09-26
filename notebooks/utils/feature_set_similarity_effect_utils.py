from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
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

FEATURE_SET_COLORS = {
    "scalar_compact": "#6B7280",
    "extended": "#0072B2",
    "gpt_who": "#D55E00",
    "scalar_all": "#009E73",
    "simple_similarity": "#7C3AED",
}

MODEL_FAMILY_MARKERS = {
    "logistic_regression": "o",
    "random_forest": "s",
    "gpt_who": "D",
}

FEATURE_SET_LABELS = {
    "scalar_compact": "simple features",
    "simple_similarity": "similarity",
    "scalar_all": "all scalar features",
    "extended": "extended features",
    "gpt_who": "GPT-who",
}


def _ordered_scorer_llms(values: Sequence[str]) -> list[str]:
    unique_values = list(dict.fromkeys(values))
    known = [value for value in unique_values if value in DEFAULT_MODEL_SIZES_BILLIONS]
    unknown = [value for value in unique_values if value not in DEFAULT_MODEL_SIZES_BILLIONS]
    return sorted(known, key=DEFAULT_MODEL_SIZES_BILLIONS.get) + unknown


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
    feature_colors = {
        feature_set: FEATURE_SET_COLORS.get(feature_set, color)
        for feature_set, color in zip(feature_sets, sns.color_palette("colorblind", n_colors=len(feature_sets)))
    }
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


def plot_feature_set_lines(
    metrics: pd.DataFrame,
    *,
    title: str,
    metric: str = "macro_f1",
) -> plt.Figure:
    """Plot one model-family/feature-set line across scorer LLMs."""
    _require_columns(metrics, {"scorer_llm", "model_family", "feature_set", metric})
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    scorer_llms = _ordered_scorer_llms(metrics["scorer_llm"])
    summary = (
        metrics.groupby(["scorer_llm", "model_family", "feature_set"], as_index=False)[metric]
        .mean()
    )
    if summary.empty:
        axis.text(0.5, 0.5, "No feature-set scores", ha="center", va="center")
        axis.set_axis_off()
        return figure

    positions = {scorer_llm: index for index, scorer_llm in enumerate(scorer_llms)}
    line_frames = summary.assign(
        line_label=summary.apply(
            lambda row: "gpt_who" if row["feature_set"] == "gpt_who" else f"{row['model_family']} / {row['feature_set']}",
            axis=1,
        )
    )
    line_labels = sorted(line_frames["line_label"].unique())
    colors = dict(zip(line_labels, sns.color_palette("colorblind", n_colors=len(line_labels))))
    for line_label, frame in line_frames.groupby("line_label", sort=True):
        frame = frame.sort_values("scorer_llm")
        model_family = frame["model_family"].iloc[0]
        axis.plot(
            [positions[scorer_llm] for scorer_llm in frame["scorer_llm"]],
            frame[metric],
            marker=MODEL_FAMILY_MARKERS.get(model_family, "o"),
            linestyle="None",
            label=line_label,
            color=colors[line_label],
        )
    axis.set_xticks(range(len(scorer_llms)), scorer_llms, rotation=30, ha="right")
    axis.set_xlabel("Scorer LLM")
    axis.set_ylabel(f"Mean {metric}")
    axis.set_title(title)
    axis.legend(title="Model / feature set")
    return figure


def plot_classifier_feature_set_comparison(
    metrics: pd.DataFrame,
    *,
    title: str,
    metric: str = "macro_f1",
) -> plt.Figure:
    """Compare classifier feature sets in separate model-family panels."""
    _require_columns(metrics, {"scorer_llm", "model_family", "feature_set", metric})
    classifier_metrics = metrics.loc[
        ~metrics["model_family"].eq("gpt_who") & ~metrics["feature_set"].eq("gpt_who")
    ].copy()
    model_families = list(pd.unique(classifier_metrics["model_family"]))
    if not model_families:
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axis.text(0.5, 0.5, "No classifier feature-set scores", ha="center", va="center")
        axis.set_axis_off()
        return figure

    figure, axes = plt.subplots(
        1, len(model_families), figsize=(6 * len(model_families), 5),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]
    scorer_llms = _ordered_scorer_llms(classifier_metrics["scorer_llm"])
    positions = {scorer_llm: index for index, scorer_llm in enumerate(scorer_llms)}
    feature_sets = list(pd.unique(classifier_metrics["feature_set"]))
    offsets = {
        feature_set: offset
        for feature_set, offset in zip(
            feature_sets,
            np.linspace(-0.18, 0.18, num=len(feature_sets)),
        )
    }
    colors = {
        feature_set: FEATURE_SET_COLORS.get(feature_set, color)
        for feature_set, color in zip(
            feature_sets, sns.color_palette("colorblind", n_colors=len(feature_sets))
        )
    }
    summary = (
        classifier_metrics.groupby(["model_family", "scorer_llm", "feature_set"], as_index=False)[metric]
        .mean()
    )
    gpt_who_summary = (
        metrics.loc[metrics["feature_set"].eq("gpt_who")]
        .groupby("scorer_llm", as_index=False)[metric]
        .mean()
    )
    for axis, model_family in zip(axes, model_families):
        family_summary = summary.loc[summary["model_family"].eq(model_family)]
        for feature_set, frame in family_summary.groupby("feature_set", sort=False):
            axis.scatter(
                [positions[scorer_llm] + offsets[feature_set] for scorer_llm in frame["scorer_llm"]],
                frame[metric],
                color=colors[feature_set],
                marker="o",
                s=55,
                label=FEATURE_SET_LABELS.get(feature_set, feature_set),
            )
        if not gpt_who_summary.empty:
            axis.scatter(
                [positions[scorer_llm] + 0.24 for scorer_llm in gpt_who_summary["scorer_llm"]],
                gpt_who_summary[metric],
                color=FEATURE_SET_COLORS["gpt_who"],
                marker=MODEL_FAMILY_MARKERS["gpt_who"],
                s=70,
                label=FEATURE_SET_LABELS["gpt_who"],
            )
        axis.set_title(model_family.replace("_", " ").title())
        axis.set_xticks(range(len(scorer_llms)), scorer_llms, rotation=30, ha="right")
        axis.set_xlabel("Scorer LLM")
        axis.legend(title="Feature set")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"Mean {metric}")
    figure.suptitle(title)
    return figure


def plot_new_method_vs_gpt_who(
    metrics: pd.DataFrame,
    *,
    title: str,
    new_method_feature_set: str = "simple_similarity",
    metric: str = "macro_f1",
) -> plt.Figure:
    """Compare one new method directly against GPT-who across scorer LLMs."""
    _require_columns(metrics, {"scorer_llm", "feature_set", metric})
    selected = metrics.loc[
        metrics["feature_set"].isin([new_method_feature_set, "gpt_who"])
    ].copy()
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if selected.empty:
        axis.text(0.5, 0.5, "No new-method/GPT-who scores", ha="center", va="center")
        axis.set_axis_off()
        return figure

    scorer_llms = _ordered_scorer_llms(selected["scorer_llm"])
    positions = {scorer_llm: index for index, scorer_llm in enumerate(scorer_llms)}
    methods = [new_method_feature_set, "gpt_who"]
    offsets = {method: offset for method, offset in zip(methods, (-0.12, 0.12))}
    summary = selected.groupby(["scorer_llm", "feature_set"], as_index=False)[metric].mean()
    for feature_set in methods:
        frame = summary.loc[summary["feature_set"].eq(feature_set)]
        if frame.empty:
            continue
        axis.scatter(
            [positions[scorer_llm] + offsets[feature_set] for scorer_llm in frame["scorer_llm"]],
            frame[metric],
            color=FEATURE_SET_COLORS.get(feature_set, "#666666"),
            marker="o" if feature_set != "gpt_who" else "D",
            s=65,
            label=feature_set,
        )
    axis.set_xticks(range(len(scorer_llms)), scorer_llms, rotation=30, ha="right")
    axis.set_xlabel("Scorer LLM")
    axis.set_ylabel(f"Mean {metric}")
    axis.set_title(title)
    axis.legend(title="Method")
    axis.grid(axis="y", alpha=0.25)
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
    sns.scatterplot(
        data=known,
        x="parameter_billion",
        y="mean_delta",
        hue="feature_set",
        style="feature_set",
        palette=FEATURE_SET_COLORS,
        s=90,
        ax=axis,
    )
    for _, row in known.iterrows():
        axis.annotate(str(row["scorer_llm"]), (row["parameter_billion"], row["mean_delta"]), xytext=(5, 5), textcoords="offset points")
    if len(known) >= 3:
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("Scorer model size (billion parameters)")
    axis.set_ylabel("Mean enriched-feature-set gain")
    axis.set_title(title)
    return figure
