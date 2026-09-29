from __future__ import annotations

import pandas as pd
import pytest

from notebooks.utils.semeval_local_evidence_export_utils import (
    SCORED_WINDOW_KEY_COLUMNS,
    merge_scored_window_exports,
    validate_scored_window_columns,
)


def _window_key(text_id: int) -> dict[str, object]:
    return {
        "dataset": "semeval",
        "split": "MG",
        "text_id": text_id,
        "model": "logistic_regression",
        "window_length": 32,
        "start_token": 0,
        "stop_token": 32,
    }


def test_merge_replaces_matching_rows_and_preserves_unrelated_rows() -> None:
    existing = pd.DataFrame(
        [
            {
                "dataset": "kaggle_train",
                "split": "HG",
                "text_id": 1,
                "model": "logistic_regression",
                "window_length": 32,
                "start_token": 0,
                "stop_token": 32,
                "mg_score": 0.1,
            },
            {**_window_key(7), "mg_score": 0.2},
        ]
    )
    generated = existing.iloc[[1]].assign(mg_score=0.9)

    merged = merge_scored_window_exports(existing, generated)

    assert len(merged) == 2
    assert merged.loc[merged["dataset"].eq("semeval"), "mg_score"].item() == 0.9
    assert merged.loc[merged["dataset"].eq("kaggle_train"), "mg_score"].item() == 0.1


def test_merge_is_idempotent() -> None:
    existing = pd.DataFrame([{**_window_key(1), "mg_score": 0.1}])
    generated = pd.DataFrame([{**_window_key(1), "mg_score": 0.9}])

    once = merge_scored_window_exports(existing, generated)
    twice = merge_scored_window_exports(once, generated)

    pd.testing.assert_frame_equal(once, twice)


def test_validation_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="stop_token"):
        validate_scored_window_columns(
            pd.DataFrame({"text_id": [1]}), SCORED_WINDOW_KEY_COLUMNS
        )
