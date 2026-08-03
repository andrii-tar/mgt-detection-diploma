from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.logprob_metrics import compute_logprob_metrics
from notebooks.logprob_metrics_distributions_utils import (
    TOKEN_DERIVED_METRICS,
    compute_length_metric_table,
    compute_prefix_metrics,
    default_length_metrics_path,
    load_length_metric_table,
    load_logprob_rows,
    plot_length_metric_distributions,
    save_length_metric_table,
)
from notebooks.token_candidate_similarity_heatmaps_utils import (
    build_text_heatmap_matrices,
    compute_token_coverage,
    text_d_score_summary,
)


def _prefix_test_row(split="HG"):
    return {
        "id": 3,
        "split": split,
        "text": "one two three four",
        "tokens": [
            {"logprob": math.log(0.5), "top_logprobs": [{"logprob": math.log(0.5)}]},
            {"logprob": math.log(0.25), "top_logprobs": [{"logprob": math.log(0.25)}]},
            {"logprob": math.log(0.125), "top_logprobs": [{"logprob": math.log(0.125)}]},
            {"logprob": math.log(0.0625), "top_logprobs": [{"logprob": math.log(0.0625)}]},
        ],
    }


def test_compute_prefix_metrics_uses_only_complete_prefix_lengths():
    result = compute_prefix_metrics(_prefix_test_row(), step=2)

    assert [record["token_length"] for record in result] == [2, 4]
    assert [record["num_tokens"] for record in result] == [2, 4]


def test_compute_prefix_metrics_matches_production_for_each_prefix():
    row = _prefix_test_row()

    result = compute_prefix_metrics(row, step=2)

    for record, length in zip(result, (2, 4)):
        expected = compute_logprob_metrics({**row, "tokens": row["tokens"][:length]})
        assert record["prompt_mu_score"] == pytest.approx(expected["prompt_mu_score"])
        assert record["d_score_top20"] == pytest.approx(expected["d_score_top20"], nan_ok=True)
        assert record["d_score_top20_coverage"] == pytest.approx(
            expected["d_score_top20_coverage"], nan_ok=True
        )


def test_load_logprob_rows_adds_experiment_metadata(tmp_path):
    data_dir = tmp_path / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([_prefix_test_row()]).to_parquet(data_dir / "part.parquet")

    rows = list(load_logprob_rows(tmp_path))

    assert len(rows) == 1
    assert rows[0]["dataset"] == "demo"
    assert rows[0]["model"] == "model"
    assert rows[0]["split"] == "HG"


def test_compute_length_metric_table_keeps_token_metrics_only(tmp_path):
    data_dir = tmp_path / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([_prefix_test_row()]).to_parquet(data_dir / "part.parquet")

    result = compute_length_metric_table(tmp_path, step=2, show_progress=False)

    assert list(result["token_length"]) == [2, 4]
    assert {"dataset", "model", "split", "token_length", *TOKEN_DERIVED_METRICS} <= set(result.columns)
    assert not {"text_len_symbols", "text_len_words", "text_len_sentences"} & set(result.columns)


def test_default_length_metrics_path_uses_processed_root(tmp_path):
    assert default_length_metrics_path(tmp_path) == tmp_path / "length_metric_distributions" / "metrics.parquet"


def test_length_metric_table_parquet_round_trip(tmp_path):
    data = pd.DataFrame(
        {
            "dataset": ["demo"],
            "token_length": [50],
            "perplexity": [1.5],
        }
    )
    output_path = tmp_path / "nested" / "metrics.parquet"

    saved_path = save_length_metric_table(data, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    pd.testing.assert_frame_equal(load_length_metric_table(output_path), data)


def test_plot_length_metric_distributions_draws_one_3d_figure_per_experiment(monkeypatch):
    data = pd.DataFrame(
        [
            {"dataset": "demo", "model": "model", "split": "HG", "token_length": 2, "perplexity": 2.0},
            {"dataset": "demo", "model": "model", "split": "MG", "token_length": 2, "perplexity": 3.0},
            {"dataset": "demo", "model": "model", "split": "HG", "token_length": 4, "perplexity": 2.5},
        ]
    )
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_length_metric_distributions(data, "perplexity", bins=2)

    figures = plt.get_fignums()
    assert len(figures) == 1
    assert plt.figure(figures[0]).axes[0].name == "3d"
    plt.close("all")


def test_length_distribution_notebook_imports_helpers_before_calls():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "logprob_metric_length_distributions.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    import_index = next(
        index
        for index, cell in enumerate(cells)
        if "from logprob_metrics_distributions_utils import" in "".join(cell.get("source", []))
    )
    call_indices = [
        index
        for index, cell in enumerate(cells)
        if "compute_length_metric_table(" in "".join(cell.get("source", []))
    ]

    assert import_index < min(call_indices)
    assert all(
        not any(line.lstrip().startswith(f"def {name}(") for line in cell.get("source", []))
        for cell in cells
        for name in ("compute_prefix_metrics", "compute_length_metric_table", "plot_length_metric_distributions")
    )
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in cells
        if cell["cell_type"] == "code"
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
