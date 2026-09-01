from __future__ import annotations

import pandas as pd
import pytest

from notebooks.semeval_boundary_utils import (
    add_gold_boundaries,
    predict_boundary_positions,
    summarize_boundary_predictions,
)


def _scored_token_windows(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "window_length": 1,
                "text_id": 7,
                "start_token": token_position,
                "stop_token": token_position + 1,
                "mg_score": score,
            }
            for token_position, score in enumerate(scores)
        ]
    )


def test_predict_boundary_positions_accepts_smoothing_and_sustained_length() -> None:
    predictions = predict_boundary_positions(
        _scored_token_windows([0.8, 0.8, 0.8, 0.8, 0.8]),
        {7: 5},
        threshold=0.5,
        smoothing_window=5,
        sustained_length=3,
    )

    assert predictions["predicted_token_boundary"].tolist() == [0]


def test_ignores_isolated_one_token_spike() -> None:
    predictions = predict_boundary_positions(
        _scored_token_windows([0.1, 0.8, 0.1, 0.1]),
        {7: 4},
        threshold=0.5,
        smoothing_window=1,
        sustained_length=3,
    )

    assert predictions["predicted_token_boundary"].tolist() == [4]


def test_selects_first_run_of_three_consecutive_above_threshold_scores() -> None:
    predictions = predict_boundary_positions(
        _scored_token_windows([0.1, 0.8, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8]),
        {7: 8},
        threshold=0.5,
        smoothing_window=1,
        sustained_length=3,
    )

    assert predictions["predicted_token_boundary"].tolist() == [1]


def test_falls_back_to_token_count_when_no_sustained_run_exists() -> None:
    predictions = predict_boundary_positions(
        _scored_token_windows([0.1, 0.8, 0.8, 0.1]),
        {7: 4},
        threshold=0.5,
        smoothing_window=1,
        sustained_length=3,
    )

    assert predictions["predicted_token_boundary"].tolist() == [4]
    assert predictions["no_crossing"].tolist() == [True]


@pytest.mark.parametrize(
    ("smoothing_window", "sustained_length"),
    [(0, 3), (-1, 3), (2, 3), (1, 0), (1, -1)],
)
def test_rejects_invalid_smoothing_window_or_sustained_length(
    smoothing_window: int,
    sustained_length: int,
) -> None:
    with pytest.raises(ValueError):
        predict_boundary_positions(
            _scored_token_windows([0.1]),
            {7: 1},
            smoothing_window=smoothing_window,
            sustained_length=sustained_length,
        )


def test_predicts_first_mg_scored_token_without_accessing_gold_labels() -> None:
    scored_windows = pd.DataFrame(
        [
            {"model": "logistic_regression", "window_length": 2, "text_id": 7, "start_token": 0, "stop_token": 2, "mg_score": 0.2},
            {"model": "logistic_regression", "window_length": 2, "text_id": 7, "start_token": 2, "stop_token": 4, "mg_score": 0.8},
        ]
    )

    predictions = predict_boundary_positions(
        scored_windows,
        {7: 4},
        threshold=0.5,
        smoothing_window=1,
        sustained_length=1,
    )

    assert predictions.to_dict("records") == [
        {
            "model": "logistic_regression",
            "window_length": 2,
            "text_id": 7,
            "predicted_token_boundary": 2,
            "token_count": 4,
            "no_crossing": False,
        }
    ]


def test_aligns_semantic_word_boundary_to_token_and_summarizes_errors() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "window_length": 2,
                "text_id": 7,
                "predicted_token_boundary": 2,
                "token_count": 4,
            },
            {
                "model": "logistic_regression",
                "window_length": 2,
                "text_id": 8,
                "predicted_token_boundary": 1,
                "token_count": 3,
            },
        ]
    )
    annotations = pd.DataFrame(
        [
            {"id": 7, "text": "human words machine words", "boundary_word_index": 2},
            {"id": 8, "text": "machine only now", "boundary_word_index": 0},
        ]
    )
    token_labels = {7: ["human", " words", " machine", " words"], 8: ["machine", " only", " now"]}

    labeled = add_gold_boundaries(predictions, annotations, token_labels)
    summary = summarize_boundary_predictions(labeled)

    assert labeled["gold_token_boundary"].tolist() == [2, 0]
    assert labeled["absolute_token_error"].tolist() == [0, 1]
    assert summary.to_dict("records") == [
        {
            "model": "logistic_regression",
            "window_length": 2,
            "texts": 2,
            "mean_absolute_token_error": 0.5,
            "median_absolute_token_error": 0.5,
            "exact_accuracy": 0.5,
            "within_1_accuracy": 1.0,
            "within_3_accuracy": 1.0,
            "within_5_accuracy": 1.0,
            "early_rate": 0.0,
            "late_rate": 0.5,
            "no_crossing_rate": 0.0,
        }
    ]


def test_summarizes_early_late_and_no_crossing_rates() -> None:
    predictions = pd.DataFrame(
        [
            {"model": "lr", "window_length": 1, "predicted_token_boundary": 2, "gold_token_boundary": 3, "absolute_token_error": 1, "no_crossing": False},
            {"model": "lr", "window_length": 1, "predicted_token_boundary": 4, "gold_token_boundary": 3, "absolute_token_error": 1, "no_crossing": False},
            {"model": "lr", "window_length": 1, "predicted_token_boundary": 5, "gold_token_boundary": 3, "absolute_token_error": 2, "no_crossing": True},
        ]
    )

    assert summarize_boundary_predictions(predictions).to_dict("records") == [
        {
            "model": "lr",
            "window_length": 1,
            "texts": 3,
            "mean_absolute_token_error": 4 / 3,
            "median_absolute_token_error": 1.0,
            "exact_accuracy": 0.0,
            "within_1_accuracy": 2 / 3,
            "within_3_accuracy": 1.0,
            "within_5_accuracy": 1.0,
            "early_rate": 1 / 3,
            "late_rate": 2 / 3,
            "no_crossing_rate": 1 / 3,
        }
    ]


def test_summary_preserves_legacy_input_and_reports_unavailable_diagnostics_as_nan() -> None:
    summary = summarize_boundary_predictions(
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "window_length": 1,
                    "absolute_token_error": 2,
                }
            ]
        )
    )

    assert summary.loc[0, "mean_absolute_token_error"] == 2.0
    assert pd.isna(summary.loc[0, "early_rate"])
    assert pd.isna(summary.loc[0, "late_rate"])
    assert pd.isna(summary.loc[0, "no_crossing_rate"])


def test_summary_uses_token_count_for_no_crossing_when_flag_is_absent() -> None:
    summary = summarize_boundary_predictions(
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "window_length": 1,
                    "absolute_token_error": 1,
                    "predicted_token_boundary": 4,
                    "token_count": 4,
                }
            ]
        )
    )

    assert summary.loc[0, "no_crossing_rate"] == 1.0


def test_summary_ignores_rows_with_partial_boundary_pairs() -> None:
    summary = summarize_boundary_predictions(
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "window_length": 1,
                    "absolute_token_error": 1,
                    "predicted_token_boundary": 2,
                    "gold_token_boundary": 3,
                    "no_crossing": False,
                },
                {
                    "model": "lr",
                    "window_length": 1,
                    "absolute_token_error": 2,
                    "predicted_token_boundary": None,
                    "gold_token_boundary": 3,
                    "no_crossing": True,
                },
                {
                    "model": "lr",
                    "window_length": 1,
                    "absolute_token_error": 3,
                    "predicted_token_boundary": 4,
                    "gold_token_boundary": None,
                    "no_crossing": False,
                },
            ]
        )
    )

    assert summary.loc[0, "early_rate"] == 1.0
    assert summary.loc[0, "late_rate"] == 0.0


def test_aligns_boundary_when_decoded_tokens_normalize_whitespace() -> None:
    predictions = pd.DataFrame(
        [{"text_id": 7, "predicted_token_boundary": 2, "token_count": 4}]
    )
    annotations = pd.DataFrame(
        [{"id": 7, "text": "human words\nmachine only", "boundary_word_index": 2}]
    )

    labeled = add_gold_boundaries(
        predictions,
        annotations,
        {7: ["human", " words", " machine", " only"]},
    )

    assert labeled["gold_token_boundary"].tolist() == [2]


def test_aligns_boundary_when_unicode_artifacts_are_ascii_compatible() -> None:
    predictions = pd.DataFrame(
        [{"text_id": 7, "predicted_token_boundary": 2, "token_count": 4}]
    )
    annotations = pd.DataFrame(
        [{"id": 7, "text": "human’s words machine only", "boundary_word_index": 2}]
    )

    labeled = add_gold_boundaries(
        predictions,
        annotations,
        {7: ["human�s", " words", " machine", " only"]},
    )

    assert labeled["gold_token_boundary"].tolist() == [2]


def test_omits_repeated_replacement_artifacts_during_reconstruction() -> None:
    predictions = pd.DataFrame(
        [{"text_id": 7, "predicted_token_boundary": 1, "token_count": 2}]
    )
    annotations = pd.DataFrame(
        [{"id": 7, "text": "human’s—words machine", "boundary_word_index": 1}]
    )

    labeled = add_gold_boundaries(
        predictions,
        annotations,
        {7: ["human��s���words", " machine"]},
    )

    assert labeled["gold_token_boundary"].tolist() == [1]


def test_aligns_boundary_when_token_reconstruction_loses_one_character() -> None:
    predictions = pd.DataFrame(
        [{"text_id": 7, "predicted_token_boundary": 2, "token_count": 4}]
    )
    annotations = pd.DataFrame(
        [{"id": 7, "text": "human words machine only", "boundary_word_index": 2}]
    )

    labeled = add_gold_boundaries(
        predictions,
        annotations,
        {7: ["human", " words", " machine", " onl"]},
    )

    assert labeled["gold_token_boundary"].tolist() == [2]


def test_aligns_boundary_when_long_token_reconstruction_loses_two_characters() -> None:
    predictions = pd.DataFrame(
        [{"text_id": 7, "predicted_token_boundary": 5, "token_count": 6}]
    )
    annotations = pd.DataFrame(
        [{
            "id": 7,
            "text": (
                "human written material contains a carefully constructed machine generated continuation"
            ),
            "boundary_word_index": 5,
        }]
    )

    labeled = add_gold_boundaries(
        predictions,
        annotations,
        {
            7: [
                "human",
                " written",
                " material",
                " contains",
                " a",
                " carefully constructed machine generated continuati",
            ]
        },
    )

    assert labeled["gold_token_boundary"].tolist() == [5]


def test_rejects_token_labels_that_do_not_match_annotation() -> None:
    with pytest.raises(ValueError, match="do not reconstruct"):
        add_gold_boundaries(
            pd.DataFrame([{"text_id": 7, "predicted_token_boundary": 2, "token_count": 4}]),
            pd.DataFrame([{"id": 7, "text": "human words machine only", "boundary_word_index": 2}]),
            {7: ["human", " words", " different", " only"]},
        )
