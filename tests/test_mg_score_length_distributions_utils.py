import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from notebooks.utils.mg_score_length_distributions_utils import (
    filter_mg_score_predictions,
    plot_mg_score_by_length,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["Kaggle", "Kaggle", "MAGE", "MAGE"],
            "split": ["HG", "MG", "HG", "MG"],
            "text_id": ["h1", "m1", "h2", "m2"],
            "label": [0, 1, 0, 1],
            "mg_score": [0.2, 0.8, 0.3, 0.7],
            "start_token": [0, 0, 0, 0],
            "stop_token": [80, 180, 220, 260],
            "scorer_llm": ["gemma_3_270m_it"] * 4,
            "feature_set": ["simple_similarity", "simple_similarity", "scalar_all", "scalar_all"],
            "model_family": ["logistic_regression"] * 4,
            "training_composition": ["Mix"] * 4,
            "training_window": [32] * 4,
            "evaluation_window": ["whole_text"] * 4,
        }
    )


def test_filter_selects_requested_configuration_and_normalizes_labels():
    result = filter_mg_score_predictions(
        _predictions(),
        scorer_llm="gemma_3_270m_it",
        feature_sets=("simple_similarity",),
         training_window=32,
        model_families=("logistic_regression",),
        training_composition="Mix",
        evaluation_compositions=("Kaggle",),
    )

    assert result["label"].tolist() == ["HG", "MG"]
    assert result["token_count"].tolist() == [80, 180]


def test_filter_reports_empty_prediction_artifacts_clearly():
    with pytest.raises(ValueError, match="no complete prediction artifacts"):
        filter_mg_score_predictions(
            pd.DataFrame(),
            scorer_llm="gemma_3_270m_it",
            feature_sets=("simple_similarity",),
             training_window=32,
            model_families=("logistic_regression",),
        )


def test_filter_rejects_duplicate_document_rows():
    frame = pd.concat([_predictions(), _predictions().iloc[[0]].assign(mg_score=0.25)], ignore_index=True)

    with pytest.raises(ValueError, match="ambiguous"):
        filter_mg_score_predictions(
            frame,
            scorer_llm="gemma_3_270m_it",
            feature_sets=("simple_similarity",),
             training_window=32,
            model_families=("logistic_regression",),
            training_composition="Mix",
        )


def test_plot_mg_score_by_length_returns_figure():
    frame = filter_mg_score_predictions(
        _predictions(),
        scorer_llm="gemma_3_270m_it",
        feature_sets=("simple_similarity", "scalar_all"),
         training_window=32,
        model_families=("logistic_regression",),
        training_composition="Mix",
    )
    comparison_frame = pd.concat(
        [frame.loc[frame["dataset"].eq("Kaggle")], frame.loc[frame["dataset"].eq("Kaggle")].assign(feature_set="scalar_all")],
        ignore_index=True,
    )

    figure = plot_mg_score_by_length(comparison_frame, dataset="Kaggle", model_family="logistic_regression")

    assert len(figure.axes) == 2
    assert all(axis.get_ylabel() == "MG score" for axis in figure.axes)


def test_plot_mg_score_by_length_can_draw_into_existing_axes():
    frame = filter_mg_score_predictions(
        _predictions(),
        scorer_llm="gemma_3_270m_it",
        feature_sets=("simple_similarity", "scalar_all"),
        training_window=32,
        model_families=("logistic_regression",),
        training_composition="Mix",
    )
    frame = pd.concat(
        [
            frame.loc[frame["dataset"].eq("Kaggle")],
            frame.loc[frame["dataset"].eq("Kaggle")].assign(feature_set="scalar_all"),
        ],
        ignore_index=True,
    )
    figure, axes = plt.subplots(1, 2)

    result = plot_mg_score_by_length(
        frame,
        dataset="Kaggle",
        model_family="logistic_regression",
        axes=axes,
    )

    assert result is figure
    assert len(figure.axes) == 2
