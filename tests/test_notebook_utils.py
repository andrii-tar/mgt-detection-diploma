from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import notebooks.logprob_metrics_distributions_utils as metric_utils
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
    TextHeatmapMatrices,
    build_text_heatmap_matrices,
    compute_token_coverage,
    plot_smoothed_token_summaries,
    rolling_mean,
    token_summary_series,
    text_d_score_summary,
    visualize_text_similarities,
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


def test_compute_length_metric_table_supports_source_batches(tmp_path):
    data_dir = tmp_path / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([_prefix_test_row()]).to_parquet(data_dir / "part.parquet")

    result = compute_length_metric_table(tmp_path, step=2, batch_size=1, show_progress=False)

    assert list(result["token_length"]) == [2, 4]


def test_compute_length_metric_table_reports_batch_progress(tmp_path, monkeypatch):
    data_dir = tmp_path / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([_prefix_test_row()]).to_parquet(data_dir / "part.parquet")
    progress = {}

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            progress.update(kwargs)

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(metric_utils, "tqdm", Progress)

    compute_length_metric_table(tmp_path, step=2, batch_size=1, show_progress=True)

    assert progress["desc"] == "Computing prefix metric batches"
    assert progress["unit"] == "batch"


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


def test_plot_length_metric_distributions_uses_token_length_for_y_axis(monkeypatch):
    data = pd.DataFrame(
        [
            {"dataset": "demo", "model": "model", "split": "HG", "token_length": 2, "perplexity": 2.0},
            {"dataset": "demo", "model": "model", "split": "HG", "token_length": 4, "perplexity": 2.5},
        ]
    )
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_length_metric_distributions(data, "perplexity", bins=2)

    axis = plt.figure(plt.get_fignums()[0]).axes[0]
    assert axis.get_ylabel() == "token length"
    assert axis.get_zlabel() == "density"
    plt.close("all")


# def test_plot_length_metric_distributions_draws_3d_bars(monkeypatch):
#     data = pd.DataFrame(
#         [
#             {"dataset": "demo", "model": "model", "split": "HG", "token_length": 2, "perplexity": 2.0},
#             {"dataset": "demo", "model": "model", "split": "HG", "token_length": 4, "perplexity": 2.5},
#         ]
#     )
#     plt.close("all")
#     monkeypatch.setattr(plt, "show", lambda: None)

#     plot_length_metric_distributions(data, "perplexity", bins=2)

#     axis = plt.figure(plt.get_fignums()[0]).axes[0]
#     assert axis.collections
#     plt.close("all")


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
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert "default_length_metrics_path" in notebook_source
    assert "load_length_metric_table" in notebook_source
    assert "save_length_metric_table" in notebook_source
    assert "RECOMPUTE = False" in notebook_source
    assert "LENGTH_METRICS_PATH = default_length_metrics_path(PROCESSED_ROOT)" in notebook_source
    assert "if RECOMPUTE or not LENGTH_METRICS_PATH.exists():" in notebook_source
    assert 'print(f"Stored at: {LENGTH_METRICS_PATH}")' in notebook_source
    assert "show_progress=True" in notebook_source
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
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert "SMOOTHING_WINDOW = 10" in notebook_source
    assert "SHOW_SMOOTHED = True" in notebook_source
    assert "smoothing_window=SMOOTHING_WINDOW" in notebook_source
    assert "show_smoothed=SHOW_SMOOTHED" in notebook_source
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in cells
        if cell["cell_type"] == "code"
    )


def test_rolling_mean_is_centered_and_preserves_edges() -> None:
    result = rolling_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)

    assert result == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])


def test_rolling_mean_even_window_uses_exactly_window_positions() -> None:
    result = rolling_mean(np.arange(1.0, 13.0), window=10)

    assert result[5] == pytest.approx(np.mean(np.arange(1.0, 11.0)))


def test_rolling_mean_ignores_nan_values_and_honors_min_periods() -> None:
    result = rolling_mean(np.array([1.0, np.nan, 3.0, np.nan, 5.0]), window=3, min_periods=2)

    assert result == pytest.approx([np.nan, 2.0, np.nan, 4.0, np.nan], nan_ok=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"window": -1},
        {"window": 3.0},
        {"window": np.int64(3), "min_periods": np.float64(1.0)},
        {"window": 3, "min_periods": 0},
        {"window": 3, "min_periods": 4},
    ],
)
def test_rolling_mean_rejects_invalid_parameters(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="window|min_periods"):
        rolling_mean([1.0, 2.0, 3.0], **kwargs)


def test_rolling_mean_returns_all_nan_for_empty_or_all_nan_input() -> None:
    assert rolling_mean([], window=3).size == 0
    assert np.isnan(rolling_mean([np.nan, np.nan], window=3)).all()


def test_token_summary_series_extracts_aligned_raw_signals() -> None:
    matrices = TextHeatmapMatrices(
        logprob_matrix=np.array(
            [
                [-0.1, -0.2, -0.3],
                [-0.3, np.nan, -0.5],
                [-0.5, -0.6, np.nan],
            ]
        ),
        similarity_matrix=np.array(
            [
                [1.0, 1.0, 1.0],
                [0.2, np.nan, 0.4],
                [0.6, 0.8, np.nan],
            ]
        ),
        token_labels=["a", "b", "c"],
        logprob_scale=(-0.6, -0.1),
        similarity_scale=(0.2, 1.0),
        coverage_matrix=np.array([[0.2, 0.3, 0.4], [0.1, np.nan, 0.2]]),
        coverage_scale=(0.1, 0.4),
    )

    summary = token_summary_series(matrices, np.array([0.3, 0.5, np.nan]))

    assert set(summary) == {
        "observed_logprob",
        "mean_candidate_logprob_delta",
        "mean_candidate_similarity",
        "top20_coverage",
    }
    assert summary["observed_logprob"] == pytest.approx([-0.1, -0.2, -0.3])
    assert summary["mean_candidate_logprob_delta"] == pytest.approx([-0.4, -0.6, -0.5], nan_ok=True)
    assert summary["mean_candidate_similarity"] == pytest.approx([0.4, 0.8, 0.4], nan_ok=True)
    assert summary["top20_coverage"] == pytest.approx([0.3, 0.5, np.nan], nan_ok=True)


def test_token_summary_series_ignores_non_finite_candidates() -> None:
    matrices = TextHeatmapMatrices(
        logprob_matrix=np.array([[0.0], [1.0], [np.inf], [np.nan]]),
        similarity_matrix=np.array([[1.0], [0.2], [np.inf], [np.nan]]),
        token_labels=["a"],
        logprob_scale=(0.0, 1.0),
        similarity_scale=(0.2, 1.0),
        coverage_matrix=np.zeros((1, 1)),
        coverage_scale=(0.0, 0.0),
    )

    summary = token_summary_series(matrices, np.array([0.3]))

    assert summary["mean_candidate_logprob_delta"] == pytest.approx([1.0])
    assert summary["mean_candidate_similarity"] == pytest.approx([0.2])


def test_token_summary_series_uses_supplied_observed_logprobs() -> None:
    matrices = TextHeatmapMatrices(
        logprob_matrix=np.array([[-0.1, -0.2], [-0.3, -0.4]]),
        similarity_matrix=np.zeros((2, 2)),
        token_labels=["a", "b"],
        logprob_scale=(-0.4, -0.1),
        similarity_scale=(0.0, 0.0),
        coverage_matrix=np.zeros((1, 2)),
        coverage_scale=(0.0, 0.0),
    )

    summary = token_summary_series(
        matrices,
        np.array([0.3, 0.4]),
        observed_logprobs=np.array([-1.5, np.nan]),
    )

    assert summary["observed_logprob"] == pytest.approx([-1.5, np.nan], nan_ok=True)


def test_token_summary_series_rejects_mismatched_coverage_length() -> None:
    matrices = TextHeatmapMatrices(
        logprob_matrix=np.zeros((2, 3)),
        similarity_matrix=np.zeros((2, 3)),
        token_labels=["a", "b", "c"],
        logprob_scale=(0.0, 0.0),
        similarity_scale=(0.0, 0.0),
        coverage_matrix=np.zeros((1, 3)),
        coverage_scale=(0.0, 0.0),
    )

    with pytest.raises(ValueError, match="coverage length.*heatmap token width"):
        token_summary_series(matrices, np.array([0.1, 0.2]))


def test_plot_smoothed_token_summaries_draws_one_panel_per_metric(monkeypatch) -> None:
    summary = {
        "observed_logprob": np.array([0.1, 0.2, 0.3]),
        "mean_candidate_logprob_delta": np.array([-0.1, -0.2, -0.3]),
        "mean_candidate_similarity": np.array([0.2, 0.4, 0.6]),
        "top20_coverage": np.array([0.5, 0.6, 0.7]),
    }
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_smoothed_token_summaries(summary, window=10)

    assert len(plt.get_fignums()) == 1
    assert len(plt.figure(plt.get_fignums()[0]).axes) == 4
    plt.close("all")


def test_visualize_text_similarities_can_disable_smoothed_plot(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "notebooks.token_candidate_similarity_heatmaps_utils.load_text_row",
        lambda *args: pd.Series({"id": 1, "tokens": [], "text": ""}),
    )
    monkeypatch.setattr(
        "notebooks.token_candidate_similarity_heatmaps_utils.load_similarity_row",
        lambda *args: pd.Series({"token_similarities": []}),
    )
    monkeypatch.setattr(
        "notebooks.token_candidate_similarity_heatmaps_utils.build_text_heatmap_matrices",
        lambda *args, **kwargs: TextHeatmapMatrices(
            np.empty((2, 0)), np.empty((2, 0)), [], (0.0, 0.0), (0.0, 0.0), np.empty((1, 0)), (0.0, 0.0)
        ),
    )
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.draw_square_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.draw_token_coverage_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.display", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: None)

    visualize_text_similarities(tmp_path, "demo", "model", "HG", 0, show_smoothed=False)

    assert plt.get_fignums() == []


def test_visualize_text_similarities_truncates_summary_inputs_to_matrix_width(monkeypatch, tmp_path) -> None:
    logprob_row = pd.Series(
        {
            "id": 1,
            "tokens": [
                {"logprob": -1.0, "token": "a"},
                {"logprob": -2.0, "token": "b"},
                {"logprob": -3.0, "token": "c"},
            ],
            "text": "abc",
        }
    )
    matrices = TextHeatmapMatrices(
        np.zeros((2, 2)), np.zeros((2, 2)), ["a", "b"], (0.0, 0.0), (0.0, 0.0), np.zeros((1, 2)), (0.0, 0.0)
    )
    captured = {}
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.load_text_row", lambda *args: logprob_row)
    monkeypatch.setattr(
        "notebooks.token_candidate_similarity_heatmaps_utils.load_similarity_row",
        lambda *args: pd.Series({"token_similarities": []}),
    )
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.build_text_heatmap_matrices", lambda *args, **kwargs: matrices)
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.draw_square_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.draw_token_coverage_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.display", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "notebooks.token_candidate_similarity_heatmaps_utils.token_summary_series",
        lambda matrices, coverage, observed_logprobs=None: captured.update(
            coverage=coverage, observed_logprobs=observed_logprobs
        ) or {},
    )
    monkeypatch.setattr("notebooks.token_candidate_similarity_heatmaps_utils.plot_smoothed_token_summaries", lambda *args, **kwargs: None)

    visualize_text_similarities(tmp_path, "demo", "model", "HG", 0)

    assert captured["coverage"].size == len(matrices.token_labels)
    assert captured["observed_logprobs"] == pytest.approx([-1.0, -2.0])
