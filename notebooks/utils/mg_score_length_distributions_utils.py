from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from notebooks.utils.classification_results_analysis_utils import load_prediction_artifacts
from notebooks.utils.document_window_evaluation_utils import aggregate_window_predictions


PREDICTION_COLUMNS = (
    "dataset",
    "split",
    "text_id",
    "label",
    "mg_score",
    "start_token",
    "stop_token",
    "scorer_llm",
    "feature_set",
    "model_family",
    "training_regime",
    "training_window",
    "evaluation_window",
    "source_dataset",
)
BOUNDED_PREFIX_DIMENSION_COLUMNS = (
    "run_id",
    "model_family",
    "scorer_llm",
    "feature_set",
    "training_regime",
    "training_window",
    "evaluation_regime",
    "partition",
)


def _matches(value: object, expected: object) -> bool:
    return expected is None or str(value) == str(expected)


def _prediction_artifact_filter(
    predictions_root: Path,
    *,
    partition: str,
    scorer_llm: str,
    feature_sets: Sequence[str],
    training_window: int | str,
    evaluation_window: int | str,
    model_families: Sequence[str],
    training_composition: str | None,
    evaluation_compositions: Sequence[str] | None,
):
    feature_set_values = set(feature_sets)
    model_family_values = set(model_families)
    def matches(artifact: Path) -> bool:
        try:
            relative_parts = artifact.relative_to(predictions_root).parts
        except ValueError:
            return False
        if len(relative_parts) < 5:
            return False
        artifact_id, artifact_training, artifact_evaluation_window, _, artifact_partition = relative_parts[:5]
        artifact_id_parts = artifact_id.split("__")
        if len(artifact_id_parts) < 5:
            return False
        artifact_model, artifact_scorer, artifact_feature_set, _, artifact_training_window = artifact_id_parts[:5]
        return (
            artifact_model in model_family_values
            and artifact_scorer == scorer_llm
            and artifact_feature_set in feature_set_values
            and artifact_training_window == str(training_window)
            and artifact_evaluation_window == str(evaluation_window)
            and artifact_partition == partition
            and (
                training_composition is None
                or artifact_training.replace("+", "_") == training_composition.replace("+", "_")
            )
        )

    return matches


def filter_mg_score_predictions(
    predictions: pd.DataFrame,
    *,
    scorer_llm: str,
    feature_sets: Sequence[str],
    training_window: int | str,
    evaluation_window: int | str = "whole_text",
    model_families: Sequence[str],
    training_composition: str | None = None,
    evaluation_compositions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Filter prediction rows to one explicit experiment configuration."""
    if predictions.empty:
        raise ValueError("no complete prediction artifacts were loaded for the requested run")
    required = {"dataset", "split", "text_id", "label", "mg_score", "stop_token", "model_family"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")

    selected = predictions.copy()
    if not feature_sets:
        raise ValueError("at least one feature set must be requested")
    filters = {
        "scorer_llm": scorer_llm,
        "training_window": training_window,
        "evaluation_window": evaluation_window,
    }
    if training_composition is not None:
        training_column = "training_composition" if "training_composition" in selected else "training_regime"
        filters[training_column] = training_composition
    for column, expected in filters.items():
        if column not in selected:
            raise ValueError(f"prediction frame is missing configuration column: {column}")
        selected = selected.loc[selected[column].map(lambda value: _matches(value, expected))]
    if "feature_set" not in selected:
        raise ValueError("prediction frame is missing configuration column: feature_set")
    selected = selected.loc[selected["feature_set"].isin(feature_sets)]
    selected = selected.loc[selected["model_family"].isin(model_families)] if "model_family" in selected else selected
    if evaluation_compositions is not None:
        evaluation_column = "source_dataset" if "source_dataset" in selected else "dataset"
        selected = selected.loc[selected[evaluation_column].isin(evaluation_compositions)]
        if evaluation_column == "source_dataset":
            selected = selected.assign(dataset=selected[evaluation_column])

    selected = selected.assign(
        label=selected["label"].map({0: "HG", 1: "MG", "0": "HG", "1": "MG", "HG": "HG", "MG": "MG"}),
        mg_score=pd.to_numeric(selected["mg_score"], errors="coerce"),
        token_count=pd.to_numeric(selected["stop_token"], errors="coerce"),
    )
    selected = selected.dropna(subset=["label", "mg_score", "token_count"])
    selected = selected.loc[selected["label"].isin(("HG", "MG")) & selected["token_count"].gt(0)]
    selected = selected.drop_duplicates()
    if selected.empty:
        raise ValueError("no prediction rows match the requested configuration")

    identity = ["model_family", "dataset", "split", "text_id", "feature_set"]
    if str(evaluation_window) != "whole_text":
        if "start_token" not in selected:
            raise ValueError("window prediction frame is missing required column: start_token")
        identity.append("start_token")
    duplicate_mask = selected.duplicated(identity, keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, identity].drop_duplicates().head(3).to_dict("records")
        raise ValueError(f"prediction configuration is ambiguous; duplicate rows include: {duplicates}")
    return selected.reset_index(drop=True)


def aggregate_mg_score_window_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one mean-window MG score and full document length per text."""
    if predictions.empty:
        raise ValueError("window prediction frame cannot be empty")
    required = {
        "dataset", "split", "text_id", "label", "mg_score", "start_token",
        "stop_token", "model_family", "feature_set",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"window prediction frame is missing required columns: {missing}")

    selected = predictions.copy()
    if "source_dataset" not in selected:
        selected["source_dataset"] = selected["dataset"]
    selected["start_token"] = pd.to_numeric(selected["start_token"], errors="raise")
    selected["stop_token"] = pd.to_numeric(selected["stop_token"], errors="raise")
    document_frames = []
    for (model_family, feature_set), windows in selected.groupby(
        ["model_family", "feature_set"], sort=False, dropna=False,
    ):
        aggregation_rows = windows[
            ["source_dataset", "split", "text_id", "label", "mg_score", "start_token"]
        ].rename(columns={"start_token": "window_id"})
        documents = aggregate_window_predictions(aggregation_rows)
        lengths = (
            windows.groupby(["source_dataset", "split", "text_id"], as_index=False, sort=False)
            .agg(token_count=("stop_token", "max"))
        )
        documents = documents.merge(
            lengths, on=["source_dataset", "split", "text_id"], validate="one_to_one",
        )
        documents = documents.assign(
            dataset=documents["source_dataset"],
            model_family=model_family,
            feature_set=feature_set,
            mg_score=documents["document_score"],
        )
        document_frames.append(documents)
    return pd.concat(document_frames, ignore_index=True)


def load_mg_score_length_frame(
    data_root: Path | str,
    *,
    run_id: str,
    partition: str = "test",
    scorer_llm: str,
    feature_sets: Sequence[str],
    training_window: int | str,
    evaluation_window: int | str = "whole_text",
    model_families: Sequence[str],
    training_composition: str | None = None,
    evaluation_compositions: Sequence[str] | None = None,
) -> pd.DataFrame:
    predictions_root = Path(data_root) / "experiment-results" / "classification" / "evaluations" / run_id / "predictions"
    predictions = load_prediction_artifacts(
        data_root,
        run_id=run_id,
        progress_description="Loading classification prediction artifacts",
        columns=list(PREDICTION_COLUMNS),
        artifact_filter=_prediction_artifact_filter(
            predictions_root,
            partition=partition,
            scorer_llm=scorer_llm,
            feature_sets=feature_sets,
            training_window=training_window,
            evaluation_window=evaluation_window,
            model_families=model_families,
            training_composition=training_composition,
            evaluation_compositions=evaluation_compositions,
        ),
    )
    return filter_mg_score_predictions(
        predictions,
        scorer_llm=scorer_llm,
        feature_sets=feature_sets,
        training_window=training_window,
        evaluation_window=evaluation_window,
        model_families=model_families,
        training_composition=training_composition,
        evaluation_compositions=evaluation_compositions,
    )


def _filter_bounded_prefix_results(
    frame: pd.DataFrame,
    *,
    run_id: str,
    model_family: str,
    scorer_llm: str,
    feature_set: str,
    training_regime: str,
    training_window: int | str,
    evaluation_regime: str,
    partition: str,
) -> pd.DataFrame:
    filters = {
        "run_id": run_id,
        "model_family": model_family,
        "scorer_llm": scorer_llm,
        "feature_set": feature_set,
        "training_regime": training_regime,
        "training_window": training_window,
        "evaluation_regime": evaluation_regime,
        "partition": partition,
    }
    missing = sorted(set(BOUNDED_PREFIX_DIMENSION_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"bounded-prefix result frame is missing configuration columns: {missing}")
    selected = frame.copy()
    for column, expected in filters.items():
        selected = selected.loc[selected[column].map(lambda value: _matches(value, expected))]
    if selected.empty:
        raise ValueError("no bounded-prefix result rows match the requested configuration")
    return selected.reset_index(drop=True)


def load_bounded_prefix_result_frames(
    data_root: Path | str,
    *,
    run_id: str,
    model_family: str,
    scorer_llm: str,
    feature_set: str,
    training_regime: str,
    training_window: int | str,
    evaluation_regime: str,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics and coverage for one bounded-prefix experiment configuration."""
    result_root = Path(data_root) / "experiment-results" / "classification" / "evaluations" / run_id
    filters = {
        "run_id": run_id,
        "model_family": model_family,
        "scorer_llm": scorer_llm,
        "feature_set": feature_set,
        "training_regime": training_regime,
        "training_window": training_window,
        "evaluation_regime": evaluation_regime,
        "partition": partition,
    }
    metrics = _filter_bounded_prefix_results(
        pd.read_csv(result_root / "prefix_evaluation_metrics.csv"), **filters,
    )
    coverage = _filter_bounded_prefix_results(
        pd.read_csv(result_root / "prefix_evaluation_coverage.csv"), **filters,
    )
    return metrics, coverage


def pooled_bounded_prefix_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Derive pooled class-balanced metrics from per-dataset confusion counts."""
    required = {"min_tokens", "max_tokens", "tn", "fp", "fn", "tp"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"bounded-prefix metrics are missing required columns: {missing}")
    counts = metrics.loc[:, list(required)].copy()
    for column in required:
        counts[column] = pd.to_numeric(counts[column], errors="coerce")
    counts = counts.dropna()
    if counts.empty:
        raise ValueError("bounded-prefix metrics contain no complete confusion counts")
    pooled = counts.groupby(["min_tokens", "max_tokens"], as_index=False)[["tn", "fp", "fn", "tp"]].sum()
    sensitivity = pooled["tp"] / (pooled["tp"] + pooled["fn"])
    specificity = pooled["tn"] / (pooled["tn"] + pooled["fp"])
    mg_f1 = (2 * pooled["tp"]) / (2 * pooled["tp"] + pooled["fp"] + pooled["fn"])
    hg_f1 = (2 * pooled["tn"]) / (2 * pooled["tn"] + pooled["fp"] + pooled["fn"])
    return pooled.assign(
        balanced_accuracy=(sensitivity + specificity) / 2,
        macro_f1=(mg_f1 + hg_f1) / 2,
    ).sort_values(["min_tokens", "max_tokens"]).reset_index(drop=True)


def pooled_bounded_prefix_feature_set_metrics(
    metrics_by_feature_set: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Pool bounded-prefix confusion counts independently for each feature set."""
    if not metrics_by_feature_set:
        raise ValueError("at least one feature set is required")
    frames = []
    for feature_set, metrics in metrics_by_feature_set.items():
        pooled = pooled_bounded_prefix_metrics(metrics).assign(feature_set=str(feature_set))
        frames.append(pooled)
    return pd.concat(frames, ignore_index=True)[
        [
            "feature_set",
            "min_tokens",
            "max_tokens",
            "tn",
            "fp",
            "fn",
            "tp",
            "balanced_accuracy",
            "macro_f1",
        ]
    ]


def dataset_balanced_bounded_prefix_feature_set_metrics(
    metrics_by_feature_set: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average per-dataset macro F1 equally for each feature set and bound."""
    if not metrics_by_feature_set:
        raise ValueError("at least one feature set is required")
    frames = []
    for feature_set, metrics in metrics_by_feature_set.items():
        required = {"dataset", "min_tokens", "max_tokens", "tn", "fp", "fn", "tp"}
        missing = sorted(required.difference(metrics.columns))
        if missing:
            raise ValueError(f"bounded-prefix metrics are missing required columns: {missing}")
        counts = metrics.loc[:, list(required)].copy()
        for column in required.difference({"dataset"}):
            counts[column] = pd.to_numeric(counts[column], errors="coerce")
        counts = counts.dropna()
        if counts.empty:
            raise ValueError("bounded-prefix metrics contain no complete confusion counts")
        mg_f1 = (2 * counts["tp"]) / (2 * counts["tp"] + counts["fp"] + counts["fn"])
        hg_f1 = (2 * counts["tn"]) / (2 * counts["tn"] + counts["fp"] + counts["fn"])
        counts = counts.assign(dataset_macro_f1=(mg_f1 + hg_f1) / 2)
        balanced = counts.groupby(["min_tokens", "max_tokens"], as_index=False).agg(
            macro_f1=("dataset_macro_f1", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        frames.append(balanced.assign(feature_set=str(feature_set)))
    return pd.concat(frames, ignore_index=True)[
        ["feature_set", "min_tokens", "max_tokens", "macro_f1", "dataset_count"]
    ]


def plot_bounded_prefix_macro_f1_comparison(
    metrics: pd.DataFrame,
    *,
    feature_set_order: Sequence[str] | None = None,
    bound_labels: Sequence[str] | None = None,
) -> plt.Figure:
    """Plot dataset-balanced bounded-prefix macro F1 for multiple feature sets."""
    required = {"feature_set", "min_tokens", "max_tokens", "macro_f1"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"bounded-prefix comparison is missing required columns: {missing}")
    selected = metrics.loc[:, list(required)].copy()
    for column in ("min_tokens", "max_tokens", "macro_f1"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=["macro_f1", "min_tokens", "max_tokens"])
    if selected.empty:
        raise ValueError("bounded-prefix comparison contains no complete macro F1 rows")
    selected["feature_set"] = selected["feature_set"].astype(str)
    selected["token_bound"] = (
        selected["min_tokens"].astype(int).astype(str)
        + "-"
        + selected["max_tokens"].astype(int).astype(str)
    )
    order = list(bound_labels) if bound_labels is not None else list(dict.fromkeys(selected["token_bound"]))
    feature_order = (
        list(feature_set_order)
        if feature_set_order is not None
        else list(dict.fromkeys(selected["feature_set"]))
    )
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(
        data=selected,
        x="token_bound",
        y="macro_f1",
        hue="feature_set",
        order=order,
        hue_order=feature_order,
        errorbar=None,
        ax=axis,
    )
    axis.set_title("Dataset-balanced bounded-prefix macro F1")
    axis.set_xlabel("Leading prefix length (tokens)")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0, 1)
    axis.legend(title="Feature set")
    return figure


def plot_bounded_prefix_evaluation(metrics: pd.DataFrame, coverage: pd.DataFrame) -> plt.Figure:
    """Plot pooled bounded-prefix performance and source-dataset class coverage."""
    pooled = pooled_bounded_prefix_metrics(metrics)
    required_coverage = {"dataset", "class", "min_tokens", "max_tokens", "scored_row_count"}
    missing = sorted(required_coverage.difference(coverage.columns))
    if missing:
        raise ValueError(f"bounded-prefix coverage is missing required columns: {missing}")
    coverage_frame = coverage.loc[:, list(required_coverage)].copy()
    coverage_frame["scored_row_count"] = pd.to_numeric(coverage_frame["scored_row_count"], errors="coerce")
    coverage_frame = coverage_frame.dropna(subset=["scored_row_count"])
    if coverage_frame.empty:
        raise ValueError("bounded-prefix coverage contains no scored rows")

    def bound_label(frame: pd.DataFrame) -> pd.Series:
        return frame["min_tokens"].astype(int).astype(str) + "-" + frame["max_tokens"].astype(int).astype(str)

    pooled = pooled.assign(token_bound=bound_label(pooled))
    coverage_frame = coverage_frame.assign(
        token_bound=bound_label(coverage_frame),
        source_class=coverage_frame["dataset"].astype(str) + " " + coverage_frame["class"].astype(str),
    )
    bound_order = pooled["token_bound"].tolist()
    performance = pooled.melt(
        id_vars="token_bound",
        value_vars=["balanced_accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    sns.barplot(
        data=performance,
        x="token_bound",
        y="score",
        hue="metric",
        order=bound_order,
        palette={"balanced_accuracy": "#2878b5", "macro_f1": "#c83e4d"},
        ax=axes[0],
    )
    axes[0].set_title("Pooled Mixed performance")
    axes[0].set_xlabel("Leading prefix length (tokens)")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="Metric", labels=["Balanced accuracy", "Macro F1"])
    sns.barplot(
        data=coverage_frame,
        x="token_bound",
        y="scored_row_count",
        hue="source_class",
        order=bound_order,
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("Scored texts by source dataset")
    axes[1].set_xlabel("Leading prefix length (tokens)")
    axes[1].set_ylabel("Scored texts")
    axes[1].legend(title="Dataset and class")
    return figure


def plot_mg_score_by_length(
    frame: pd.DataFrame,
    *,
    dataset: str,
    model_family: str,
    length_bin_size: int = 100,
    axes: Sequence[plt.Axes] | None = None,
) -> plt.Figure:
    """Plot HG/MG score distributions across token-length bins."""
    if length_bin_size <= 0:
        raise ValueError("length_bin_size must be positive")
    required = {"dataset", "model_family", "feature_set", "label", "mg_score", "token_count"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    selected = frame.loc[
        frame["dataset"].eq(dataset) & frame["model_family"].eq(model_family)
    ].copy()
    if selected.empty:
        raise ValueError(f"no rows available for dataset={dataset!r}, model_family={model_family!r}")
    selected["length_bin"] = (
        (selected["token_count"].astype("int64") // length_bin_size) * length_bin_size
    ).astype("int64")
    selected["length_label"] = selected["length_bin"].map(lambda value: f"{value}-{value + length_bin_size - 1}")
    order = [
        f"{value}-{value + length_bin_size - 1}"
        for value in sorted(selected["length_bin"].unique())
    ]

    feature_set_order = list(dict.fromkeys(selected["feature_set"].astype(str)))
    if axes is None:
        column_count = 2
        row_count = (len(feature_set_order) + column_count - 1) // column_count
        figure, axes_grid = plt.subplots(
            row_count,
            column_count,
            figsize=(max(6, len(order) * 0.7) * column_count, 6 * row_count),
            squeeze=False,
            constrained_layout=True,
        )
        plot_axes = list(axes_grid.flat)
        for unused_axis in plot_axes[len(feature_set_order):]:
            unused_axis.set_visible(False)
    else:
        plot_axes = list(axes)
        if len(plot_axes) != len(feature_set_order):
            raise ValueError("axes must contain one axis per feature set")
        figure = plot_axes[0].figure
    for axis, feature_set in zip(plot_axes, feature_set_order):
        feature_frame = selected.loc[selected["feature_set"].eq(feature_set)]
        sns.boxenplot(
            data=feature_frame,
            x="length_label",
            y="mg_score",
            hue="label",
            hue_order=("HG", "MG"),
            order=order,
            showfliers=False,
            palette={"HG": "#2878b5", "MG": "#c83e4d"},
            ax=axis,
        )
        axis.set_title(feature_set)
        axis.set_xlabel("Text length (tokens)")
        axis.set_ylabel("MG score")
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylim(0, 1)
        axis.legend(title="Class")
    if axes is None:
        figure.suptitle(f"{dataset}: {model_family} MG score by text length")
    else:
        plot_axes[0].text(
            -0.28,
            0.5,
            f"{dataset}\n{model_family}",
            transform=plot_axes[0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )
    return figure
