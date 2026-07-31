from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.logprob_metrics import compute_logprob_metrics
from notebooks.token_candidate_similarity_heatmaps_utils import (
    build_text_heatmap_matrices,
    compute_token_coverage,
    text_d_score_summary,
)


def test_text_d_score_summary_matches_production_metrics():
    row = pd.Series(
        {
            "tokens": [
                {
                    "logprob": math.log(0.5),
                    "top_logprobs": [{"logprob": math.log(0.5)}],
                }
            ]
        }
    )

    summary = text_d_score_summary(row)

    assert math.isnan(summary["d_score_top20"])
    assert summary["d_score_top20_coverage"] == pytest.approx(0.5)
    assert summary["d_score_top20_num_tokens"] == 1


def test_token_coverage_matrix_contains_candidate_probability_mass():
    logprob_row = pd.Series(
        {
            "tokens": [
                {
                    "token": " hello",
                    "logprob": math.log(0.25),
                    "top_logprobs": [{"logprob": math.log(0.5)}],
                }
            ]
        }
    )
    similarity_row = pd.Series({"token_similarities": [{"top_logprobs": []}]})

    matrices = build_text_heatmap_matrices(logprob_row, similarity_row, top_k=1)

    assert matrices.coverage_matrix[0, 0] == pytest.approx(0.5)
    assert matrices.coverage_scale == pytest.approx((0.5, 0.5))


def test_invalid_candidate_logprobs_are_omitted_from_coverage_matrix():
    logprob_row = pd.Series(
        {
            "tokens": [
                {
                    "token": " hello",
                    "logprob": math.log(0.25),
                    "top_logprobs": [
                        {"logprob": None},
                        {"logprob": float("nan")},
                        {"logprob": float("inf")},
                        {"logprob": math.log(0.5)},
                    ],
                }
            ]
        }
    )
    similarity_row = pd.Series(
        {
            "token_similarities": [
                {
                    "top_logprobs": [
                        {"similarity_to_gt": 0.1},
                        {"similarity_to_gt": 0.2},
                        {"similarity_to_gt": 0.3},
                        {"similarity_to_gt": 0.4},
                    ]
                }
            ]
        }
    )

    matrices = build_text_heatmap_matrices(logprob_row, similarity_row, top_k=4)

    assert math.isnan(matrices.coverage_matrix[0, 0])
    assert math.isnan(matrices.coverage_matrix[1, 0])
    assert math.isnan(matrices.coverage_matrix[2, 0])
    assert matrices.coverage_matrix[3, 0] == pytest.approx(0.5)
    assert math.isnan(matrices.logprob_matrix[1, 0])
    assert math.isnan(matrices.logprob_matrix[2, 0])
    assert math.isnan(matrices.logprob_matrix[3, 0])
    assert matrices.similarity_matrix[1:, 0] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert matrices.coverage_scale == pytest.approx((0.5, 0.5))


def test_compute_token_coverage_returns_cumulative_top_k_mass_per_position():
    row = pd.Series(
        {
            "tokens": [
                {
                    "logprob": math.log(0.25),
                    "top_logprobs": [
                        {"logprob": math.log(0.5)},
                        {"logprob": math.log(0.25)},
                    ],
                },
                {
                    "logprob": math.log(0.5),
                    "top_logprobs": [],
                },
            ]
        }
    )

    coverage = compute_token_coverage(row, top_k=2)

    assert coverage == pytest.approx([0.75, math.nan], nan_ok=True)


def test_summary_candidate_count_matches_production_non_none_semantics():
    row = {
        "id": 7,
        "tokens": [
            {
                "logprob": math.log(0.25),
                "top_logprobs": [
                    {"logprob": None},
                    {"logprob": float("nan")},
                ],
            }
        ],
    }

    summary = text_d_score_summary(pd.Series(row))
    production = compute_logprob_metrics(row)

    assert summary["d_score_top20_num_tokens"] == production["d_score_top20_num_tokens"]


def test_heatmap_notebook_imports_updated_orchestration_before_calls():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "token_candidate_similarity_heatmaps.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    import_index = next(
        index
        for index, cell in enumerate(cells)
        if "from notebooks.token_candidate_similarity_heatmaps_utils import" in "".join(cell.get("source", []))
    )
    call_indices = [
        index
        for index, cell in enumerate(cells)
        if "visualize_text_similarities(" in "".join(cell.get("source", []))
    ]

    assert import_index < min(call_indices)
    assert all(
        f"def {name}(" not in "".join(cell.get("source", []))
        for cell in cells
        for name in ("path_slug", "experiment_name", "load_text_row", "load_similarity_row", "build_text_heatmap_matrices")
    )
