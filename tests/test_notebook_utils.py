from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

import notebooks.utils.logprob_metrics_distributions_utils as metric_utils
import notebooks.utils.local_evidence_utils as local_evidence_utils
import notebooks.utils.token_candidate_similarity_heatmaps_utils as heatmap_utils
from scripts.logprob_metrics import compute_logprob_metrics
from notebooks.utils.logprob_metrics_distributions_utils import (
    TOKEN_DERIVED_METRICS,
    compute_length_metric_table,
    compute_prefix_metrics,
    default_length_metrics_path,
    iter_logprob_row_batches,
    load_length_metric_table,
    load_logprob_rows,
    plot_length_metric_distributions,
    save_length_metric_table,
    token_prefix_length_for_words,
    truncate_logprob_row_by_words,
    validate_word_limits,
)
from notebooks.utils.token_candidate_similarity_heatmaps_utils import (
    TextHeatmapMatrices,
    build_text_heatmap_matrices,
    calculate_rolling_statistics,
    compute_token_coverage,
    _expand_window_scores_to_tokens,
    _normalize_series,
    plot_rolling_statistics_histograms,
    plot_smoothed_token_summaries,
    rolling_mean,
    token_summary_series,
    text_d_score_summary,
    validated_text_summary,
    visualize_text_similarities,
)
from notebooks.utils.local_evidence_utils import (
    LOCAL_FEATURE_COLUMNS,
    build_local_evidence_windows,
    limit_local_evidence_windows_by_words,
    load_local_evidence_windows,
    save_local_evidence_windows,
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


def test_token_prefix_length_for_words_stops_at_requested_word():
    tokens = [{"token": token} for token in ("One", " two", " three", " four")]

    assert token_prefix_length_for_words("One two three four", tokens, 3) == 3


def test_token_prefix_treats_replacement_tokens_for_nonbreaking_space_as_whitespace():
    tokens = [{"token": token} for token in ("One", "\ufffd", "\ufffd", " two", " three")]

    assert token_prefix_length_for_words("One\u00a0 two three", tokens, 2) == 4


def test_token_prefix_keeps_short_text_unchanged():
    tokens = [{"token": token} for token in ("One", " two")]

    assert token_prefix_length_for_words("One two", tokens, 10) == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_validate_word_limits_rejects_invalid_values(limit):
    with pytest.raises(ValueError):
        validate_word_limits((limit,))


def test_truncate_logprob_row_by_words_updates_text_and_tokens():
    row = _prefix_test_row()
    row["tokens"] = [
        {**token, "token": text}
        for token, text in zip(row["tokens"], ("one", " two", " three", " four"), strict=True)
    ]

    result = truncate_logprob_row_by_words(row, 2)

    assert result["text"] == "one two"
    assert len(result["tokens"]) == 2
    assert row["text"] == "one two three four"


def test_truncate_logprob_row_keeps_short_text_exactly_unchanged():
    row = _prefix_test_row()
    row["text"] = "one  two three four"
    row["tokens"] = [
        {**token, "token": text}
        for token, text in zip(row["tokens"], ("one", "  two", " three", " four"), strict=True)
    ]

    result = truncate_logprob_row_by_words(row, 10)

    assert result["text"] == row["text"]
    assert result["tokens"] == row["tokens"]


def test_limit_local_evidence_windows_by_words_excludes_crossing_windows():
    windows = pd.DataFrame([
        {"split": "HG", "text_id": 1, "start_token": start, "stop_token": start + 2}
        for start in range(4)
    ])
    stage1_rows = {
        ("HG", "1"): {
            "text": "one two three four",
            "tokens": [{"token": token} for token in ("one", " two", " three", " four")],
        }
    }

    result = limit_local_evidence_windows_by_words(windows, stage1_rows, 3)

    assert result["stop_token"].tolist() == [2, 3]


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


def test_iter_logprob_row_batches_limits_rows_per_batch(tmp_path):
    data_dir = tmp_path / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([
        {**_prefix_test_row(), "id": 1},
        {**_prefix_test_row(), "id": 2},
        {**_prefix_test_row(), "id": 3},
    ]).to_parquet(data_dir / "part.parquet", index=False)

    batches = list(iter_logprob_row_batches(tmp_path, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 1]
    assert [row["id"] for batch in batches for row in batch] == [1, 2, 3]


def test_iter_logprob_row_batches_samples_ids_per_model_and_split(tmp_path):
    for model, row_ids in (("model_a", range(10)), ("model_b", range(100, 110))):
        data_dir = tmp_path / f"demo__{model}" / "HG" / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame([
            {**_prefix_test_row(), "id": row_id}
            for row_id in row_ids
        ]).to_parquet(data_dir / "part.parquet", index=False)

    rows = list(
        metric_utils.iter_logprob_rows(
            tmp_path,
            selected_models=["model_a", "model_b"],
            max_rows_per_dataset_split=2,
        )
        )

    sampled = {}
    for row in rows:
        sampled.setdefault(row["model"], []).append(row["id"])

    assert set(sampled) == {"model_a", "model_b"}
    assert all(len(ids) == 2 for ids in sampled.values())
    assert all(set(ids).issubset(set(range(start, start + 10))) for ids, start in ((sampled["model_a"], 0), (sampled["model_b"], 100)))

    repeated_rows = list(
        metric_utils.iter_logprob_rows(
            tmp_path,
            selected_models=["model_a", "model_b"],
            max_rows_per_dataset_split=2,
        )
    )
    assert [(row["model"], row["id"]) for row in rows] == [
        (row["model"], row["id"]) for row in repeated_rows
    ]


def test_iter_logprob_row_batches_rejects_cap_above_five_hundred(tmp_path):
    with pytest.raises(ValueError, match="at most 500"):
        list(
            metric_utils.iter_logprob_row_batches(
                tmp_path,
                max_rows_per_dataset_split=501,
            )
        )


def test_load_logprob_rows_filters_requested_split_parent_ids(tmp_path: Path) -> None:
    for split, row_id in (("HG", 1), ("HG", 2), ("MG", 1)):
        data_dir = tmp_path / "demo__model" / split / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**_prefix_test_row(split), "id": row_id}]).to_parquet(
            data_dir / f"{row_id}.parquet", index=False
        )

    rows = metric_utils.load_logprob_rows(
        tmp_path,
        selected_datasets=["demo"],
        selected_models=["model"],
        selected_parent_ids={"HG": {2}},
    )

    assert [(row["split"], row["id"]) for row in rows] == [("HG", 2)]


def test_load_metrics_uses_directory_split_when_file_has_no_split_column(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "demo__model" / "HG" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame([{"id": 1, "perplexity": 2.0}]).to_parquet(metrics_dir / "metrics.parquet", index=False)

    metrics = metric_utils.load_metrics(
        tmp_path,
        selected_datasets=["demo"],
        selected_models=["model"],
        columns=["id", "perplexity"],
    )

    assert metrics["split"].tolist() == ["HG"]


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
        if "from notebooks.utils.logprob_metrics_distributions_utils import" in "".join(cell.get("source", []))
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
    assert "RECOMPUTE = True" in notebook_source
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
        if "from notebooks.utils.token_candidate_similarity_heatmaps_utils import" in "".join(cell.get("source", []))
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


def test_plot_smoothed_token_summaries_displays_smoothed_statistics(monkeypatch) -> None:
    summary = {"metric": np.array([1.0, 2.0, 3.0])}
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_smoothed_token_summaries(summary, window=5)

    title = plt.figure(plt.get_fignums()[0]).axes[0].get_title()
    assert "mean=2.0000" in title
    assert "std=0.0000" in title
    assert "variance=0.0000" in title
    plt.close("all")


def test_expand_window_scores_to_tokens_averages_overlapping_windows() -> None:
    scored_windows = pd.DataFrame(
        {
            "start_token": [0, 2],
            "stop_token": [4, 5],
            "mg_score": [0.2, 0.8],
        }
    )

    scores = _expand_window_scores_to_tokens(scored_windows, token_count=5)

    assert scores == pytest.approx([0.2, 0.2, 0.5, 0.5, 0.8])


def test_expand_window_scores_to_tokens_ignores_missing_and_non_numeric_scores() -> None:
    scored_windows = pd.DataFrame(
        {
            "start_token": [0, 0, 0],
            "stop_token": [2, 2, 2],
            "mg_score": [0.2, pd.NA, "missing"],
        }
    )

    scores = _expand_window_scores_to_tokens(scored_windows, token_count=2)

    assert scores == pytest.approx([0.2, 0.2])


def test_expand_window_scores_to_tokens_returns_nan_for_uncovered_positions() -> None:
    scored_windows = pd.DataFrame(
        {
            "start_token": [1],
            "stop_token": [3],
            "mg_score": [0.4],
        }
    )

    scores = _expand_window_scores_to_tokens(scored_windows, token_count=4)

    assert scores == pytest.approx([np.nan, 0.4, 0.4, np.nan], nan_ok=True)


def test_expand_window_scores_to_tokens_returns_all_nan_for_all_invalid_scores() -> None:
    scored_windows = pd.DataFrame(
        {
            "start_token": [0, 1],
            "stop_token": [2, 4],
            "mg_score": [np.nan, "invalid"],
        }
    )

    scores = _expand_window_scores_to_tokens(scored_windows, token_count=4)

    assert np.isnan(scores).all()


@pytest.mark.parametrize(
    "scored_windows, token_count, match",
    [
        (pd.DataFrame({"start_token": [0], "stop_token": [1]}), 1, "mg_score"),
        (pd.DataFrame({"stop_token": [1], "mg_score": [0.2]}), 1, "start_token"),
        (pd.DataFrame({"start_token": [0], "stop_token": [1], "mg_score": [0.2]}), 0, "token_count"),
        (pd.DataFrame({"start_token": [0], "stop_token": [1], "mg_score": [0.2]}), -1, "token_count"),
        (pd.DataFrame({"start_token": [2], "stop_token": [2], "mg_score": [0.2]}), 3, "span"),
        (pd.DataFrame({"start_token": [3], "stop_token": [2], "mg_score": [0.2]}), 3, "span"),
        (pd.DataFrame({"start_token": [-1], "stop_token": [2], "mg_score": [0.2]}), 3, "range"),
        (pd.DataFrame({"start_token": [1], "stop_token": [4], "mg_score": [0.2]}), 3, "range"),
    ],
)
def test_expand_window_scores_to_tokens_rejects_invalid_input(
    scored_windows: pd.DataFrame,
    token_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _expand_window_scores_to_tokens(scored_windows, token_count)


def test_normalize_series_scales_finite_values_and_preserves_nan() -> None:
    values = _normalize_series(np.array([2.0, 4.0, np.nan]))

    assert values == pytest.approx([0.0, 1.0, np.nan], nan_ok=True)


def test_normalize_series_maps_constant_finite_values_to_zero() -> None:
    values = _normalize_series(np.array([3.0, 3.0]))

    assert values == pytest.approx([0.0, 0.0])


def test_normalize_series_preserves_all_nan_series() -> None:
    values = _normalize_series(np.array([np.nan]))

    assert np.isnan(values).all()


def test_plot_normalized_token_metrics_with_local_evidence_uses_one_shared_axis(monkeypatch) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.lines = []
            self.legend_kwargs = None

        def plot(self, _positions, values, **kwargs):
            self.lines.append((np.asarray(values), kwargs))

        def set_title(self, _title):
            pass

        def set_xlabel(self, _label):
            pass

        def set_ylabel(self, _label):
            pass

        def set_ylim(self, _lower, _upper):
            pass

        def grid(self, **_kwargs):
            pass

        def legend(self, **_kwargs):
            self.legend_kwargs = _kwargs

    class FakeFigure:
        def suptitle(self, _title):
            pass

    axis = FakeAxis()
    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (FakeFigure(), axis))
    monkeypatch.setattr(plt, "show", lambda: None)
    scored_windows = pd.DataFrame(
        {
            "start_token": [0, 1],
            "stop_token": [2, 3],
            "mg_score": [0.2, 0.8],
        }
    )

    heatmap_utils.plot_normalized_token_metrics_with_local_evidence(
        {"metric_a": np.array([1.0, 2.0, 3.0]), "metric_b": np.array([3.0, 2.0, 1.0])},
        scored_windows,
        token_count=3,
        window=2,
    )

    assert len(axis.lines) == 3
    assert [line[1]["label"] for line in axis.lines] == [
        "metric_a rolling mean (2)",
        "metric_b rolling mean (2)",
        "local mg_score",
    ]
    assert [line[1]["alpha"] for line in axis.lines[:2]] == [0.5, 0.5]
    assert "alpha" not in axis.lines[2][1]
    assert axis.legend_kwargs == {"loc": "upper center", "bbox_to_anchor": (0.5, -0.15), "ncol": 2}
    assert all(
        np.all((values[np.isfinite(values)] >= 0.0) & (values[np.isfinite(values)] <= 1.0))
        for values, _kwargs in axis.lines
    )


def test_plot_normalized_token_metrics_with_local_evidence_rejects_summary_length_mismatch(monkeypatch) -> None:
    with pytest.raises(ValueError, match="metric length.*token_count"):
        heatmap_utils.plot_normalized_token_metrics_with_local_evidence(
            {"metric": np.array([1.0, 2.0])},
            pd.DataFrame({"start_token": [], "stop_token": [], "mg_score": []}),
            token_count=3,
        )


def test_plot_normalized_token_metrics_with_local_evidence_propagates_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        heatmap_utils.plot_normalized_token_metrics_with_local_evidence(
            {"metric": np.array([1.0, 2.0])},
            pd.DataFrame({"start_token": [], "stop_token": [], "mg_score": []}),
            token_count=2,
            window=0,
        )


def test_plot_normalized_token_metrics_with_local_evidence_handles_all_nan_series(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    heatmap_utils.plot_normalized_token_metrics_with_local_evidence(
        {"metric": np.array([np.nan, np.nan])},
        pd.DataFrame({"start_token": [], "stop_token": [], "mg_score": []}),
        token_count=2,
    )
    plt.close("all")


def test_plot_rolling_statistics_histograms_separates_splits_and_metrics(monkeypatch) -> None:
    statistics = pd.DataFrame(
        [
            {"split": "HG", "text_id": 1, "metric": "metric_b", "mean": 1.0, "std": 0.1, "variance": 0.01},
            {"split": "MG", "text_id": 2, "metric": "metric_b", "mean": 2.0, "std": 0.2, "variance": 0.04},
            {"split": "HG", "text_id": 3, "metric": "metric_a", "mean": 3.0, "std": 0.3, "variance": 0.09},
            {"split": "HG", "text_id": 5, "metric": "metric_a", "mean": np.inf, "std": np.inf, "variance": np.nan},
            {"split": "MG", "text_id": 4, "metric": "metric_a", "mean": np.inf, "std": np.nan, "variance": np.nan},
        ]
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        figures = plot_rolling_statistics_histograms(statistics, bins=5)

        assert len(figures) == 3
        assert all(len(figure.axes) == 4 for figure in figures)
        assert [figure._suptitle.get_text() for figure in figures] == [
            "Per-text rolling mean distributions",
            "Per-text rolling standard deviation distributions",
            "Per-text rolling variance distributions",
        ]

        statistic_labels = ["mean", "standard deviation", "variance"]
        for figure, statistic_label in zip(figures, statistic_labels):
            axes = np.asarray(figure.axes).reshape(2, 2)
            assert [axis.get_title() for axis in axes[0]] == ["HG", "MG"]
            assert [axis.get_ylabel() for axis in axes[:, 0]] == ["metric_a", "metric_b"]
            assert [axis.get_xlabel() for axis in axes[1]] == [statistic_label, statistic_label]

            # The HG metric_a cell contains one finite and one non-finite value.
            assert sum(patch.get_height() for patch in axes[0, 0].patches) == 1

            # The MG metric_a cell contains no finite values.
            assert axes[0, 1].texts[0].get_text() == "No finite values"
    finally:
        plt.close("all")


def test_visualize_text_similarities_can_disable_smoothed_plot(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "notebooks.utils.token_candidate_similarity_heatmaps_utils.load_text_row",
        lambda *args: pd.Series({"id": 1, "tokens": [], "text": ""}),
    )
    monkeypatch.setattr(
        "notebooks.utils.token_candidate_similarity_heatmaps_utils.load_similarity_row",
        lambda *args: pd.Series({"token_similarities": []}),
    )
    monkeypatch.setattr(
        "notebooks.utils.token_candidate_similarity_heatmaps_utils.build_text_heatmap_matrices",
        lambda *args, **kwargs: TextHeatmapMatrices(
            np.empty((2, 0)), np.empty((2, 0)), [], (0.0, 0.0), (0.0, 0.0), np.empty((1, 0)), (0.0, 0.0)
        ),
    )
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.draw_square_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.draw_token_coverage_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.display", lambda *args, **kwargs: None)
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
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.load_text_row", lambda *args: logprob_row)
    monkeypatch.setattr(
        "notebooks.utils.token_candidate_similarity_heatmaps_utils.load_similarity_row",
        lambda *args: pd.Series({"token_similarities": []}),
    )
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.build_text_heatmap_matrices", lambda *args, **kwargs: matrices)
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.draw_square_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.draw_token_coverage_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.display", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "notebooks.utils.token_candidate_similarity_heatmaps_utils.token_summary_series",
        lambda matrices, coverage, observed_logprobs=None: captured.update(
            coverage=coverage, observed_logprobs=observed_logprobs
        ) or {},
    )
    monkeypatch.setattr("notebooks.utils.token_candidate_similarity_heatmaps_utils.plot_smoothed_token_summaries", lambda *args, **kwargs: None)

    visualize_text_similarities(tmp_path, "demo", "model", "HG", 0)

    assert captured["coverage"].size == len(matrices.token_labels)
    assert captured["observed_logprobs"] == pytest.approx([-1.0, -2.0])


def _rolling_statistics_rows(text_id: int) -> tuple[dict, dict]:
    coverage = [0.1, 0.2, 0.3, 0.4, 0.5]
    tokens = [
        {
            "token": f" token-{index}",
            "logprob": 0.0,
            "top_logprobs": [{"logprob": math.log(value)}],
        }
        for index, value in enumerate(coverage)
    ]
    similarities = [
        {"top_logprobs": [{"similarity_to_gt": value}]}
        for value in coverage
    ]
    return (
        {"id": text_id, "text": "one two three four five", "tokens": tokens},
        {"id": text_id, "token_similarities": similarities},
    )


def test_validated_text_summary_rejects_misaligned_positions() -> None:
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    similarity_row["token_similarities"][1]["position"] = 9

    with pytest.raises(ValueError, match="position 1"):
        validated_text_summary(pd.Series(logprob_row), pd.Series(similarity_row), top_k=1)


def test_validated_text_summary_rejects_mismatched_token_counts() -> None:
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    similarity_row["token_similarities"].pop()

    with pytest.raises(ValueError, match="token count mismatch"):
        validated_text_summary(pd.Series(logprob_row), pd.Series(similarity_row), top_k=1)


def test_validated_text_summary_rejects_mismatched_token_ids() -> None:
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    logprob_row["tokens"][2]["token_id"] = 10
    similarity_row["token_similarities"][2]["token_id"] = 20

    with pytest.raises(ValueError, match="token id mismatch at position 2"):
        validated_text_summary(pd.Series(logprob_row), pd.Series(similarity_row), top_k=1)


def test_validated_text_summary_rejects_mismatched_parent_row_ids() -> None:
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    similarity_row["id"] = 99

    with pytest.raises(ValueError, match="parent row id mismatch: stage-1=11, stage-2=99"):
        validated_text_summary(pd.Series(logprob_row), pd.Series(similarity_row), top_k=1)


def test_calculate_rolling_statistics_uses_ground_truth_logprobs(tmp_path: Path) -> None:
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    for index, token in enumerate(logprob_row["tokens"]):
        token["logprob"] = float(index - 5)
    data_dir = tmp_path / "processed" / "demo__model" / "HG" / "data"
    similarity_dir = tmp_path / "processed" / "demo__model" / "HG" / "token_similarities"
    data_dir.mkdir(parents=True)
    similarity_dir.mkdir(parents=True)
    pd.DataFrame([logprob_row]).to_parquet(data_dir / "part.parquet")
    pd.DataFrame([similarity_row]).to_parquet(similarity_dir / "part.parquet")
    result = calculate_rolling_statistics(tmp_path / "processed", "demo", "model", splits=("HG",), top_k=1, window=5)

    observed = result.loc[result["metric"] == "observed_logprob"].iloc[0]
    assert observed["mean"] == pytest.approx(-3.0)


def test_calculate_rolling_statistics_reports_progress_per_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    progress: list[dict[str, object]] = []

    class Progress:
        def __init__(self, iterable: object, **kwargs: object) -> None:
            self.iterable = iterable
            progress.append(kwargs)

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(heatmap_utils, "tqdm", Progress)

    calculate_rolling_statistics(processed_root, "demo", "model", top_k=1, window=3)

    assert progress == [
        {"total": 1, "desc": "HG: rolling statistics", "unit": "text"},
        {"total": 1, "desc": "MG: rolling statistics", "unit": "text"},
    ]


def test_calculate_rolling_statistics_can_disable_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    monkeypatch.setattr(heatmap_utils, "tqdm", lambda *_args, **_kwargs: pytest.fail("progress created"))

    calculate_rolling_statistics(processed_root, "demo", "model", top_k=1, window=3, show_progress=False)


def test_calculate_rolling_statistics_reads_both_splits_and_computes_window_five(tmp_path):
    processed_root = tmp_path / "processed"
    for split, text_id in (("HG", 11), ("MG", 22)):
        data_dir = processed_root / "demo__model" / split / "data"
        similarity_dir = processed_root / "demo__model" / split / "token_similarities"
        data_dir.mkdir(parents=True)
        similarity_dir.mkdir(parents=True)
        logprob_row, similarity_row = _rolling_statistics_rows(text_id)
        pd.DataFrame([logprob_row]).to_parquet(data_dir / "part.parquet")
        pd.DataFrame([similarity_row]).to_parquet(similarity_dir / "part.parquet")

    result = calculate_rolling_statistics(processed_root, "demo", "model", top_k=1, window=5)

    assert list(result.columns) == ["split", "text_id", "metric", "mean", "std", "variance"]
    assert set(result["split"]) == {"HG", "MG"}
    assert set(result["text_id"]) == {11, 22}
    assert set(result["metric"]) == {
        "observed_logprob",
        "mean_candidate_logprob_delta",
        "mean_candidate_similarity",
        "top20_coverage",
    }
    coverage = result[(result["split"] == "HG") & (result["metric"] == "top20_coverage")].iloc[0]
    smoothed_coverage = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    assert coverage["mean"] == pytest.approx(smoothed_coverage.mean())
    assert coverage["std"] == pytest.approx(smoothed_coverage.std())
    assert coverage["variance"] == pytest.approx(smoothed_coverage.var())


def test_calculate_rolling_statistics_raises_for_missing_similarity_row(tmp_path):
    data_dir = tmp_path / "processed" / "demo__model" / "HG" / "data"
    similarity_dir = tmp_path / "processed" / "demo__model" / "HG" / "token_similarities"
    data_dir.mkdir(parents=True)
    similarity_dir.mkdir(parents=True)
    logprob_row, similarity_row = _rolling_statistics_rows(11)
    pd.DataFrame([logprob_row]).to_parquet(data_dir / "part.parquet")
    pd.DataFrame([{**similarity_row, "id": 99}]).to_parquet(similarity_dir / "part.parquet")

    with pytest.raises(KeyError, match="No token similarity row for text id 11"):
        calculate_rolling_statistics(tmp_path / "processed", "demo", "model", splits=("HG",))


def _local_evidence_rows(text_id: int) -> tuple[dict, dict]:
    tokens = [
        {
            "token": f" token-{index}",
            "token_id": index,
            "logprob": float(index) - 1.5,
            "top_logprobs": [{"logprob": -2.0}],
        }
        for index in range(6)
    ]
    similarities = [
        {
            "position": index,
            "token_id": index,
            "top_logprobs": [{"similarity_to_gt": 0.1 * (index + 1)}],
        }
        for index in range(6)
    ]
    return (
        {"id": text_id, "tokens": tokens},
        {"id": text_id, "token_similarities": similarities},
    )


def _write_local_evidence_artifacts(processed_root: Path, dataset: str = "demo", model: str = "model") -> None:
    for split, text_id in (("HG", 11), ("MG", 22)):
        data_dir = processed_root / f"{dataset}__{model}" / split / "data"
        similarity_dir = processed_root / f"{dataset}__{model}" / split / "token_similarities"
        data_dir.mkdir(parents=True)
        similarity_dir.mkdir(parents=True)
        logprob_row, similarity_row = _local_evidence_rows(text_id)
        pd.DataFrame([logprob_row]).to_parquet(data_dir / "part.parquet")
        pd.DataFrame([similarity_row]).to_parquet(similarity_dir / "part.parquet")


def test_build_local_evidence_windows_extracts_complete_windows_from_both_splits(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)

    windows = build_local_evidence_windows(
        processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1
    )

    assert windows[["split", "text_id", "label", "window_length", "start_token", "stop_token"]].to_dict("records") == [
        {"split": "HG", "text_id": 11, "label": 0, "window_length": 4, "start_token": 0, "stop_token": 4},
        {"split": "HG", "text_id": 11, "label": 0, "window_length": 4, "start_token": 2, "stop_token": 6},
        {"split": "MG", "text_id": 22, "label": 1, "window_length": 4, "start_token": 0, "stop_token": 4},
        {"split": "MG", "text_id": 22, "label": 1, "window_length": 4, "start_token": 2, "stop_token": 6},
    ]
    assert list(windows.columns) == [
        "dataset",
        "model",
        "split",
        "label",
        "text_id",
        "window_length",
        "start_token",
        "stop_token",
        "num_tokens",
        "window_text",
        *LOCAL_FEATURE_COLUMNS,
    ]
    assert windows.loc[0, "observed_logprob__mean"] == pytest.approx(0.0)
    assert windows.loc[0, "observed_logprob__finite_fraction"] == pytest.approx(1.0)
    assert windows.loc[0, "window_text"] == " token-0 token-1 token-2 token-3"


def test_build_local_evidence_windows_reports_progress_per_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    progress: list[dict[str, object]] = []

    class Progress:
        def __init__(self, iterable: object, **kwargs: object) -> None:
            self.iterable = iterable
            progress.append(kwargs)

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(local_evidence_utils, "tqdm", Progress)

    build_local_evidence_windows(processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1)

    assert progress == [
        {"total": 1, "desc": "HG: local evidence", "unit": "text"},
        {"total": 1, "desc": "MG: local evidence", "unit": "text"},
    ]


def test_build_local_evidence_windows_can_disable_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    monkeypatch.setattr(local_evidence_utils, "tqdm", lambda *_args, **_kwargs: pytest.fail("progress created"))

    build_local_evidence_windows(
        processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1, show_progress=False
    )


@pytest.mark.parametrize("window_lengths, stride", [((0,), 1), ((4,), 0), ((4,), -1)])
def test_build_local_evidence_windows_rejects_invalid_window_parameters(
    tmp_path: Path, window_lengths: tuple[int, ...], stride: int
) -> None:
    with pytest.raises(ValueError):
        build_local_evidence_windows(tmp_path, "demo", "model", window_lengths=window_lengths, stride=stride)


def test_build_local_evidence_windows_skips_lengths_longer_than_text(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)

    windows = build_local_evidence_windows(processed_root, "demo", "model", window_lengths=(7,), stride=3, top_k=1)

    assert windows.empty


def test_build_local_evidence_windows_raises_for_missing_similarity_row(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    data_dir = processed_root / "demo__model" / "HG" / "data"
    similarity_dir = processed_root / "demo__model" / "HG" / "token_similarities"
    data_dir.mkdir(parents=True)
    similarity_dir.mkdir(parents=True)
    logprob_row, similarity_row = _local_evidence_rows(11)
    pd.DataFrame([logprob_row]).to_parquet(data_dir / "part.parquet")
    pd.DataFrame([{**similarity_row, "id": 99}]).to_parquet(similarity_dir / "part.parquet")

    with pytest.raises(KeyError, match="No token similarity row for text id 11"):
        build_local_evidence_windows(processed_root, "demo", "model", window_lengths=(4,), stride=3, splits=("HG",))


def test_build_local_evidence_windows_raises_for_token_id_mismatch(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    similarity_path = processed_root / "demo__model" / "HG" / "token_similarities" / "part.parquet"
    similarity_row = pd.read_parquet(similarity_path).iloc[0].to_dict()
    similarity_row["token_similarities"][2]["token_id"] = 99
    pd.DataFrame([similarity_row]).to_parquet(similarity_path)

    with pytest.raises(ValueError, match="token id mismatch at position 2"):
        build_local_evidence_windows(processed_root, "demo", "model", window_lengths=(4,), stride=3, splits=("HG",))


def test_build_local_evidence_windows_streams_multi_shard_similarity_rows_without_pandas_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_root = tmp_path / "processed"
    data_dir = processed_root / "demo__model" / "HG" / "data"
    similarity_dir = processed_root / "demo__model" / "HG" / "token_similarities"
    data_dir.mkdir(parents=True)
    similarity_dir.mkdir(parents=True)
    first_logprob_row, first_similarity_row = _local_evidence_rows(11)
    second_logprob_row, second_similarity_row = _local_evidence_rows(12)
    pd.DataFrame([first_logprob_row]).to_parquet(data_dir / "one.parquet")
    pd.DataFrame([second_logprob_row]).to_parquet(data_dir / "two.parquet")
    pd.DataFrame([first_similarity_row]).to_parquet(similarity_dir / "one.parquet")
    pd.DataFrame([second_similarity_row]).to_parquet(similarity_dir / "two.parquet")
    original_read_parquet = pd.read_parquet
    similarity_reads: list[Path] = []

    def read_parquet(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        path = Path(path)
        if path.parent == similarity_dir:
            similarity_reads.append(path)
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    windows = build_local_evidence_windows(
        processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1, splits=("HG",)
    )

    assert set(windows["text_id"]) == {11, 12}
    assert similarity_reads == []


def test_build_local_evidence_windows_streams_similarity_parquets_without_pandas_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    similarity_dir = processed_root / "demo__model" / "HG" / "token_similarities"
    original_read_parquet = pd.read_parquet

    def read_parquet(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path).parent == similarity_dir:
            pytest.fail("similarity parquet must not be converted through pandas")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    windows = build_local_evidence_windows(
        processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1, splits=("HG",)
    )

    assert windows["text_id"].tolist() == [11, 11]


def test_local_evidence_windows_parquet_round_trip(tmp_path: Path) -> None:
    windows = pd.DataFrame({"dataset": ["demo"], "observed_logprob__mean": [0.0]})
    path = tmp_path / "nested" / "windows.parquet"

    saved_path = save_local_evidence_windows(
        windows,
        path,
        dataset="demo",
        model="model",
        splits=("HG", "MG"),
    )

    assert saved_path == path
    pd.testing.assert_frame_equal(
        load_local_evidence_windows(path, dataset="demo", model="model", splits=("HG", "MG")), windows
    )


def test_local_evidence_cache_reads_requested_columns_and_parents(tmp_path: Path) -> None:
    path = tmp_path / "windows.parquet"
    windows = pd.DataFrame([
        {"split": "HG", "text_id": 1, "label": 0, "window_text": "keep", "feature": 1.0},
        {"split": "HG", "text_id": 2, "label": 0, "window_text": "drop", "feature": 2.0},
    ])
    save_local_evidence_windows(windows, path, dataset="demo", model="model", splits=("HG",))

    selected = load_local_evidence_windows(
        path,
        dataset="demo",
        model="model",
        splits=("HG",),
        text_ids=pd.DataFrame({"split": ["HG"], "text_id": [1]}),
        columns=["split", "text_id", "label", "feature"],
    )

    assert selected.to_dict("records") == [
        {"split": "HG", "text_id": 1, "label": 0, "feature": 1.0}
    ]


def test_local_evidence_cache_persists_and_validates_schema_configuration(tmp_path: Path) -> None:
    windows = pd.DataFrame({"dataset": ["demo"], "window_length": [4], "observed_logprob__mean": [0.0]})
    path = tmp_path / "windows.parquet"

    save_local_evidence_windows(
        windows,
        path,
        dataset="demo",
        model="model",
        splits=("MG", "HG"),
        window_lengths=(4,),
        stride=2,
        top_k=1,
    )

    metadata = json.loads(pq.read_schema(path).metadata[b"local_evidence_cache"].decode("utf-8"))
    assert metadata["dataset"] == "demo"
    assert metadata["model"] == "model"
    assert metadata["splits"] == ["HG", "MG"]
    assert metadata["window_lengths"] == [4]
    assert metadata["stride"] == 2
    assert metadata["top_k"] == 1
    assert metadata["feature_columns"] == list(LOCAL_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        load_local_evidence_windows(
            path,
            dataset="demo",
            model="model",
            splits=("HG", "MG"),
            window_lengths=(4,),
            stride=2,
            top_k=1,
        ),
        windows,
    )
    with pytest.raises(ValueError, match="local-evidence cache configuration is incompatible"):
        load_local_evidence_windows(
            path,
            dataset="demo",
            model="model",
            splits=("HG", "MG"),
            window_lengths=(5,),
            stride=2,
            top_k=1,
        )


@pytest.mark.parametrize(
    ("dataset", "model", "splits"),
    [
        ("other", "model", ("HG", "MG")),
        ("demo", "other", ("HG", "MG")),
        ("demo", "model", ("HG",)),
    ],
    ids=("dataset", "model", "splits"),
)
def test_local_evidence_cache_rebuilds_when_identity_configuration_is_incompatible(
    tmp_path: Path, dataset: str, model: str, splits: tuple[str, ...]
) -> None:
    from notebooks.utils.local_evidence_utils import load_or_build_local_evidence_windows

    processed_root = tmp_path / "processed"
    _write_local_evidence_artifacts(processed_root)
    path = tmp_path / "windows.parquet"
    stale_windows = build_local_evidence_windows(
        processed_root, "demo", "model", window_lengths=(4,), stride=3, top_k=1
    )
    save_local_evidence_windows(
        stale_windows,
        path,
        dataset="demo",
        model="model",
        splits=("HG", "MG"),
        window_lengths=(4,),
        stride=3,
        top_k=1,
    )
    if (dataset, model) != ("demo", "model"):
        _write_local_evidence_artifacts(processed_root, dataset=dataset, model=model)

    rebuilt = load_or_build_local_evidence_windows(
        processed_root,
        dataset,
        model,
        path=path,
        window_lengths=(4,),
        stride=3,
        top_k=1,
        splits=splits,
    )

    assert set(rebuilt["dataset"]) == {dataset}
    assert set(rebuilt["model"]) == {model}
    assert set(rebuilt["split"]) == set(splits)
    pd.testing.assert_frame_equal(
        load_local_evidence_windows(
            path,
            dataset=dataset,
            model=model,
            splits=splits,
            window_lengths=(4,),
            stride=3,
            top_k=1,
        ),
        rebuilt,
    )


def test_local_evidence_cache_rebuild_passes_requested_text_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from notebooks.utils.local_evidence_utils import load_or_build_local_evidence_windows

    requested = pd.DataFrame({"split": ["MG"], "text_id": [7]})
    built = pd.DataFrame({"split": ["MG"], "text_id": [7], "window_length": [4]})
    build_ids: list[tuple[int, ...] | None] = []
    load_calls = 0

    def load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise FileNotFoundError
        return built

    def build(*args, **kwargs):
        build_ids.append(kwargs.get("text_ids"))
        return built

    monkeypatch.setattr(local_evidence_utils, "load_local_evidence_windows", load)
    monkeypatch.setattr(local_evidence_utils, "build_local_evidence_windows", build)
    monkeypatch.setattr(local_evidence_utils, "save_local_evidence_windows", lambda *args, **kwargs: args[1])

    result = load_or_build_local_evidence_windows(
        tmp_path, "demo", "model", path=tmp_path / "windows.parquet",
        splits=("MG",), text_ids=requested,
    )

    assert build_ids == [(7,)]
    pd.testing.assert_frame_equal(result, built)
