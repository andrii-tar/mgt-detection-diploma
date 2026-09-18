from pathlib import Path

import pandas as pd
import pytest

from notebooks.utils.classifier_generalization_comparison_utils import (
    comparison_matrix,
    configuration_index,
    filter_prediction_cohort,
    held_out_binary_metrics,
    rank_generalization_candidates,
    select_comparison_metrics,
)


def test_filter_prediction_cohort_keeps_all_windows_for_long_parents() -> None:
    rows = pd.DataFrame(
        {
            "split": ["HG", "HG", "MG"],
            "text_id": [1, 1, 2],
            "token_count": [220, 220, 190],
        }
    )

    result = filter_prediction_cohort(rows, min_text_tokens=200)

    assert result["text_id"].tolist() == [1, 1]


def test_held_out_binary_metrics_excludes_hg_only_bbc_and_in_domain_rows() -> None:
    metrics = pd.DataFrame(
        [
            {
                "training_regime": "Mixed",
                "dataset": "kaggle_train",
                "partition": "test",
                "hg_support": 5,
                "mg_support": 5,
                "macro_f1": 0.9,
            },
            {
                "training_regime": "Mixed",
                "dataset": "bbc_news_content",
                "partition": "test",
                "hg_support": 5,
                "mg_support": 0,
                "macro_f1": 0.8,
            },
            {
                "training_regime": "Mixed",
                "dataset": "external_binary",
                "partition": "test",
                "hg_support": 5,
                "mg_support": 5,
                "macro_f1": 0.7,
            },
        ]
    )

    result = held_out_binary_metrics(metrics)

    assert result["dataset"].tolist() == ["external_binary"]


def test_rank_generalization_candidates_uses_mean_then_worst() -> None:
    metrics = pd.DataFrame(
        [
            {"candidate": "a", "dataset": "d1", "macro_f1": 0.8},
            {"candidate": "a", "dataset": "d2", "macro_f1": 0.6},
            {"candidate": "b", "dataset": "d1", "macro_f1": 0.7},
            {"candidate": "b", "dataset": "d2", "macro_f1": 0.7},
        ]
    )
    semeval = pd.DataFrame(
        [
            {"candidate": "a", "mean_absolute_token_error": 10},
            {"candidate": "b", "mean_absolute_token_error": 20},
        ]
    )

    result = rank_generalization_candidates(metrics, semeval, top_tier_gap=0.2)

    assert result.iloc[0]["candidate"] == "b"


def test_rank_generalization_candidates_reports_empty_evidence() -> None:
    result = rank_generalization_candidates(
        pd.DataFrame(columns=["candidate", "dataset", "macro_f1"]),
        pd.DataFrame(),
        top_tier_gap=0.02,
    )

    assert result.empty
    assert result.attrs["evidence_status"] == "insufficient_binary_holdout"


def test_select_comparison_metrics_preserves_non_holdout_binary_rows() -> None:
    metrics = pd.DataFrame(
        [{"training_regime": "Mixed", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.8}]
    )

    result = select_comparison_metrics(metrics, min_text_tokens=None)

    assert len(result) == 1


def test_add_candidate_column_uses_configuration_dimensions() -> None:
    from notebooks.utils.classifier_generalization_comparison_utils import add_candidate_column

    metrics = pd.DataFrame(
        [{"scorer_llm": "model", "feature_set": "features", "training_window": 64, "macro_f1": 0.8}]
    )

    result = add_candidate_column(metrics)

    assert result.loc[0, "candidate"] == "model | features | 64"


def test_plot_dataset_heatmap_accepts_raw_configuration_metrics() -> None:
    from notebooks.utils.classifier_generalization_comparison_utils import plot_dataset_heatmap

    figure = plot_dataset_heatmap(
        pd.DataFrame(
            [{"scorer_llm": "model", "feature_set": "features", "dataset": "d", "macro_f1": 0.8}]
        ),
        title="heatmap",
    )

    assert figure.axes


def test_plot_comparison_matrix_uses_numeric_annotations_and_no_long_candidate_labels() -> None:
    from notebooks.utils.classifier_generalization_comparison_utils import plot_comparison_matrix

    figure = plot_comparison_matrix(
        pd.DataFrame([[0.8, 0.6]], index=["llm"], columns=["f1", "f2"]),
        title="mean macro-F1",
        metric="mean_macro_f1",
    )

    labels = [text.get_text() for text in figure.axes[0].texts]
    assert "0.800" in labels
    assert all(" | " not in label for label in labels)


def test_plot_matrix_facets_returns_empty_state_for_no_matrices() -> None:
    from notebooks.utils.classifier_generalization_comparison_utils import plot_matrix_facets

    figure = plot_matrix_facets({}, title="empty", metric="mean_macro_f1", facet_columns=("model_family",))

    assert any(text.get_text() == "No data" for text in figure.axes[0].texts)


def test_comparison_matrix_uses_llm_rows_and_feature_columns_without_long_labels() -> None:
    metrics = pd.DataFrame(
        [
            {"scorer_llm": "llm-b", "feature_set": "extended", "model_family": "rf", "training_window": 64, "evaluation_window": "whole_text", "dataset": "d1", "macro_f1": 0.8},
            {"scorer_llm": "llm-a", "feature_set": "basic", "model_family": "rf", "training_window": 64, "evaluation_window": "whole_text", "dataset": "d1", "macro_f1": 0.7},
        ]
    )

    matrices = comparison_matrix(metrics, value="macro_f1", aggregation="mean")

    matrix = matrices[("rf", 64, "whole_text")]
    assert matrix.index.tolist() == ["llm-a", "llm-b"]
    assert matrix.columns.tolist() == ["basic", "extended"]
    assert pd.isna(matrix.loc["llm-b", "basic"])


def test_comparison_matrix_worst_aggregation_is_grouped_by_dataset_before_worst() -> None:
    metrics = pd.DataFrame(
        [
            {"scorer_llm": "llm", "feature_set": "f", "model_family": "rf", "training_window": 64, "evaluation_window": "whole_text", "dataset": "d1", "macro_f1": 0.9},
            {"scorer_llm": "llm", "feature_set": "f", "model_family": "rf", "training_window": 64, "evaluation_window": "whole_text", "dataset": "d2", "macro_f1": 0.6},
        ]
    )

    result = comparison_matrix(metrics, value="macro_f1", aggregation="worst")[("rf", 64, "whole_text")]

    assert result.loc["llm", "f"] == 0.6


def test_configuration_index_is_deterministic() -> None:
    metrics = pd.DataFrame(
        [
            {"scorer_llm": "b", "feature_set": "f2", "model_family": "rf", "training_window": 64, "evaluation_window": "whole_text", "macro_f1": 0.8},
            {"scorer_llm": "a", "feature_set": "f1", "model_family": "lr", "training_window": 128, "evaluation_window": 64, "macro_f1": 0.7},
        ]
    )

    result = configuration_index(metrics, facet_columns=("model_family", "training_window", "evaluation_window"))

    assert result[["model_family", "training_window", "evaluation_window"]].to_records(index=False).tolist() == [
        ("lr", 128, 64),
        ("rf", 64, "whole_text"),
    ]
