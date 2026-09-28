from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from notebooks.utils.classification_results_analysis_utils import load_prediction_artifacts


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


def _matches(value: object, expected: object) -> bool:
    return expected is None or str(value) == str(expected)


def _prediction_artifact_filter(
    predictions_root: Path,
    *,
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
        artifact_id, artifact_training, artifact_evaluation_window, _, _ = relative_parts[:5]
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
    duplicate_mask = selected.duplicated(identity, keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, identity].drop_duplicates().head(3).to_dict("records")
        raise ValueError(f"prediction configuration is ambiguous; duplicate rows include: {duplicates}")
    return selected.reset_index(drop=True)


def load_mg_score_length_frame(
    data_root: Path | str,
    *,
    run_id: str,
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
