import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from notebooks.utils.mg_score_length_distributions_utils import (
    _prediction_artifact_filter,
    aggregate_mg_score_window_predictions,
    filter_mg_score_predictions,
    load_bounded_prefix_result_frames,
    plot_bounded_prefix_macro_f1_comparison,
    plot_bounded_prefix_evaluation,
    pooled_bounded_prefix_metrics,
    pooled_bounded_prefix_feature_set_metrics,
    dataset_balanced_bounded_prefix_feature_set_metrics,
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


def test_prediction_artifact_filter_accepts_test_and_rejects_validation(tmp_path):
    predictions_root = tmp_path / "predictions"
    artifact_filter = _prediction_artifact_filter(
        predictions_root,
        partition="test",
        scorer_llm="gemma_3_270m_it",
        feature_sets=("simple_similarity",),
        training_window=32,
        evaluation_window="whole_text",
        model_families=("random_forest",),
        training_composition="Kaggle+MAGE+Outfox+BBC",
        evaluation_compositions=("kaggle_train", "mage_train", "outfox_chatgpt_train"),
    )
    base = (
        predictions_root
        / "random_forest__gemma_3_270m_it__simple_similarity__Kaggle_MAGE_Outfox_BBC__32"
        / "Kaggle_MAGE_Outfox_BBC"
        / "whole_text"
        / "mixed"
    )

    assert artifact_filter(base / "test")
    assert not artifact_filter(base / "validation")


def test_aggregate_mg_score_window_predictions_averages_per_document_and_keeps_length():
    windows = pd.DataFrame(
        {
            "dataset": ["Kaggle", "Kaggle", "Kaggle"],
            "source_dataset": ["Kaggle"] * 3,
            "split": ["HG", "HG", "MG"],
            "text_id": ["human-1", "human-1", "machine-1"],
            "label": ["HG", "HG", "MG"],
            "mg_score": [0.2, 0.4, 0.9],
            "start_token": [0, 32, 0],
            "stop_token": [32, 64, 48],
            "model_family": ["random_forest"] * 3,
            "feature_set": ["simple_similarity"] * 3,
        }
    )

    documents = aggregate_mg_score_window_predictions(windows)

    assert documents[["text_id", "label", "mg_score", "token_count", "window_count"]].to_dict("records") == [
        {"text_id": "human-1", "label": "HG", "mg_score": pytest.approx(0.3), "token_count": 64, "window_count": 2},
        {"text_id": "machine-1", "label": "MG", "mg_score": pytest.approx(0.9), "token_count": 48, "window_count": 1},
    ]


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


def test_filter_keeps_distinct_windows_for_window_evaluation():
    windows = pd.DataFrame(
        {
            "dataset": ["Kaggle", "Kaggle"],
            "source_dataset": ["Kaggle", "Kaggle"],
            "split": ["HG", "HG"],
            "text_id": ["h1", "h1"],
            "label": [0, 0],
            "mg_score": [0.2, 0.4],
            "start_token": [0, 32],
            "stop_token": [32, 64],
            "scorer_llm": ["gemma_3_270m_it"] * 2,
            "feature_set": ["simple_similarity"] * 2,
            "model_family": ["random_forest"] * 2,
            "training_composition": ["Mix"] * 2,
            "training_window": [32] * 2,
            "evaluation_window": [32] * 2,
        }
    )

    result = filter_mg_score_predictions(
        windows,
        scorer_llm="gemma_3_270m_it",
        feature_sets=("simple_similarity",),
        training_window=32,
        evaluation_window=32,
        model_families=("random_forest",),
        training_composition="Mix",
    )

    assert result["start_token"].tolist() == [0, 32]


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


def test_load_bounded_prefix_result_frames_filters_the_exact_configuration(tmp_path):
    result_root = tmp_path / "experiment-results" / "classification" / "evaluations" / "run"
    result_root.mkdir(parents=True)
    dimensions = {
        "run_id": "run",
        "artifact_id": "artifact",
        "model_family": "random_forest",
        "scorer_llm": "gemma_3_270m_it",
        "feature_set": "simple_similarity",
        "training_regime": "Kaggle+MAGE+Outfox+BBC",
        "training_window": 32,
        "evaluation_regime": "Mixed",
        "partition": "validation",
        "min_tokens": 100,
        "max_tokens": 300,
    }
    pd.DataFrame([
        dimensions | {"dataset": "Kaggle", "tn": 8, "fp": 2, "fn": 1, "tp": 9},
        dimensions | {"dataset": "MAGE", "feature_set": "scalar_all", "tn": 8, "fp": 2, "fn": 1, "tp": 9},
    ]).to_csv(result_root / "prefix_evaluation_metrics.csv", index=False)
    pd.DataFrame([
        dimensions | {"dataset": "Kaggle", "class": "HG", "scored_row_count": 10},
        dimensions | {"dataset": "MAGE", "class": "MG", "scored_row_count": 10},
        dimensions | {"dataset": "Outfox", "partition": "test", "class": "HG", "scored_row_count": 10},
    ]).to_csv(result_root / "prefix_evaluation_coverage.csv", index=False)

    metrics, coverage = load_bounded_prefix_result_frames(
        tmp_path,
        run_id="run",
        model_family="random_forest",
        scorer_llm="gemma_3_270m_it",
        feature_set="simple_similarity",
        training_regime="Kaggle+MAGE+Outfox+BBC",
        training_window=32,
        evaluation_regime="Mixed",
        partition="validation",
    )

    assert metrics["dataset"].tolist() == ["Kaggle"]
    assert coverage["dataset"].tolist() == ["Kaggle", "MAGE"]


def test_pooled_bounded_prefix_metrics_derives_scores_from_confusion_counts():
    metrics = pd.DataFrame([
        {"min_tokens": 100, "max_tokens": 300, "tn": 80, "fp": 20, "fn": 10, "tp": 90},
        {"min_tokens": 100, "max_tokens": 300, "tn": 40, "fp": 10, "fn": 20, "tp": 30},
    ])

    result = pooled_bounded_prefix_metrics(metrics)

    assert result["balanced_accuracy"].tolist() == [pytest.approx(0.8)]
    assert result["macro_f1"].tolist() == [pytest.approx(0.8)]


def test_plot_bounded_prefix_evaluation_draws_performance_and_coverage_panels():
    metrics = pd.DataFrame([
        {"dataset": "Kaggle", "min_tokens": 100, "max_tokens": 300, "tn": 80, "fp": 20, "fn": 10, "tp": 90},
        {"dataset": "MAGE", "min_tokens": 100, "max_tokens": 300, "tn": 40, "fp": 10, "fn": 20, "tp": 30},
    ])
    coverage = pd.DataFrame([
        {"dataset": "Kaggle", "class": "HG", "min_tokens": 100, "max_tokens": 300, "scored_row_count": 100},
        {"dataset": "Kaggle", "class": "MG", "min_tokens": 100, "max_tokens": 300, "scored_row_count": 100},
    ])

    figure = plot_bounded_prefix_evaluation(metrics, coverage)

    assert [axis.get_title() for axis in figure.axes] == [
        "Pooled Mixed performance", "Scored texts by source dataset",
    ]
    assert figure.axes[0].get_ylabel() == "Score"
    assert figure.axes[1].get_ylabel() == "Scored texts"
    plt.close(figure)


def test_pooled_bounded_prefix_feature_set_metrics_keeps_feature_sets_separate():
    metrics_by_feature_set = {
        "simple_similarity": pd.DataFrame([
            {"min_tokens": 100, "max_tokens": 300, "tn": 80, "fp": 20, "fn": 20, "tp": 80},
            {"min_tokens": 300, "max_tokens": 600, "tn": 70, "fp": 30, "fn": 20, "tp": 80},
        ]),
        "scalar_all": pd.DataFrame([
            {"min_tokens": 100, "max_tokens": 300, "tn": 60, "fp": 40, "fn": 30, "tp": 70},
            {"min_tokens": 300, "max_tokens": 600, "tn": 50, "fp": 50, "fn": 40, "tp": 60},
        ]),
    }

    result = pooled_bounded_prefix_feature_set_metrics(metrics_by_feature_set)

    assert result[["feature_set", "min_tokens", "max_tokens"]].to_dict("records") == [
        {"feature_set": "simple_similarity", "min_tokens": 100, "max_tokens": 300},
        {"feature_set": "simple_similarity", "min_tokens": 300, "max_tokens": 600},
        {"feature_set": "scalar_all", "min_tokens": 100, "max_tokens": 300},
        {"feature_set": "scalar_all", "min_tokens": 300, "max_tokens": 600},
    ]
    assert result.loc[result["feature_set"].eq("simple_similarity"), "macro_f1"].tolist() == [pytest.approx(0.8), pytest.approx(0.7493734336)]
    assert result.loc[result["feature_set"].eq("scalar_all"), "macro_f1"].tolist() == [pytest.approx(0.6491228070), pytest.approx(0.5488721805)]


def test_dataset_balanced_bounded_prefix_feature_set_metrics_averages_datasets_equally():
    metrics_by_feature_set = {
        "simple_similarity": pd.DataFrame([
            {"dataset": "large", "min_tokens": 100, "max_tokens": 300, "tn": 90, "fp": 10, "fn": 10, "tp": 90},
            {"dataset": "small", "min_tokens": 100, "max_tokens": 300, "tn": 8, "fp": 2, "fn": 2, "tp": 8},
        ]),
    }

    result = dataset_balanced_bounded_prefix_feature_set_metrics(metrics_by_feature_set)

    assert result[["feature_set", "min_tokens", "max_tokens"]].to_dict("records") == [
        {"feature_set": "simple_similarity", "min_tokens": 100, "max_tokens": 300},
    ]
    assert result.loc[0, "macro_f1"] == pytest.approx(0.85)


def test_plot_bounded_prefix_macro_f1_comparison_draws_grouped_bars():
    metrics = pd.DataFrame([
        {"feature_set": "simple_similarity", "min_tokens": 100, "max_tokens": 300, "macro_f1": 0.80},
        {"feature_set": "simple_similarity", "min_tokens": 300, "max_tokens": 600, "macro_f1": 0.75},
        {"feature_set": "scalar_all", "min_tokens": 100, "max_tokens": 300, "macro_f1": 0.65},
        {"feature_set": "scalar_all", "min_tokens": 300, "max_tokens": 600, "macro_f1": 0.55},
    ])

    figure = plot_bounded_prefix_macro_f1_comparison(metrics)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "Dataset-balanced bounded-prefix macro F1"
    assert figure.axes[0].get_xlabel() == "Leading prefix length (tokens)"
    assert figure.axes[0].get_ylabel() == "Macro F1"
    assert figure.axes[0].get_ylim() == (0.0, 1.0)
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == ["100-300", "300-600"]
    plt.close(figure)


def test_pooled_bounded_prefix_feature_set_metrics_rejects_empty_mapping():
    with pytest.raises(ValueError, match="at least one feature set"):
        pooled_bounded_prefix_feature_set_metrics({})


def test_plot_bounded_prefix_macro_f1_comparison_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        plot_bounded_prefix_macro_f1_comparison(pd.DataFrame({"feature_set": ["scalar_all"]}))
