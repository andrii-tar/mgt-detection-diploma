from __future__ import annotations

import unittest
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from notebooks.logprob_metrics_distributions_utils import (
    available_metric_columns,
    load_token_similarity_stats,
    plot_hg_metric_mean_std_kde_by_model,
    plot_token_similarity,
    split_metric_summary,
)


class LoadTokenSimilarityStatsTest(unittest.TestCase):
    def test_loads_exported_token_similarity_stats_files(self) -> None:
        with TemporaryDirectory() as directory:
            processed_root = Path(directory)
            stats_path = processed_root / "dataset_a__model_a" / "HG" / "token_similarity_stats" / "token_similarity_stats.parquet"
            stats_path.parent.mkdir(parents=True)
            pd.DataFrame([self._row(1, "dataset_a", "model_a", "HG")]).to_parquet(stats_path, index=False)

            loaded = load_token_similarity_stats(processed_root)

        self.assertEqual(loaded[["id", "dataset", "model", "split"]].to_dict("records"), [{"id": 1, "dataset": "dataset_a", "model": "model_a", "split": "HG"}])

    def test_raises_when_no_exported_stats_files_exist(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "token similarity stats"):
                load_token_similarity_stats(Path(directory))

    def _row(self, row_id: int, dataset: str, model: str, split: str) -> dict[str, object]:
        return {
            "id": row_id,
            "dataset": dataset,
            "model": model,
            "experiment_name": f"{dataset}__{model}",
            "split": split,
            "selected_similarity_count": 2,
            "selected_similarity_mean": 0.9,
            "selected_similarity_std": 0.1,
            "candidate_similarity_token_count": 2,
            "candidate_similarity_mean": 0.8,
            "candidate_similarity_std": 0.2,
        }


class PlotTokenSimilarityTest(unittest.TestCase):
    def test_filters_selected_datasets_and_models_before_dispatch(self) -> None:
        data = pd.DataFrame(
            [
                self._row("dataset_a", "model_a", "HG"),
                self._row("dataset_a", "model_b", "MG"),
                self._row("dataset_b", "model_a", "HG"),
            ]
        )

        with patch("notebooks.logprob_metrics_distributions_utils.plot_similarity_kde_pair") as plot_kde:
            plot_token_similarity(
                data,
                plot_type="kde",
                selected_datasets=["dataset_a"],
                selected_models=["model_a"],
            )

        dispatched = plot_kde.call_args.args[0]
        self.assertEqual(dispatched[["dataset", "model", "split"]].to_dict("records"), [{"dataset": "dataset_a", "model": "model_a", "split": "HG"}])

    def test_rejects_unknown_plot_type(self) -> None:
        data = pd.DataFrame([self._row("dataset_a", "model_a", "HG")])

        with self.assertRaisesRegex(ValueError, "plot_type"):
            plot_token_similarity(data, plot_type="unknown")

    def _row(self, dataset: str, model: str, split: str) -> dict[str, object]:
        return {
            "dataset": dataset,
            "model": model,
            "split": split,
            "selected_similarity_mean": 0.9,
            "selected_similarity_std": 0.1,
            "candidate_similarity_mean": 0.8,
            "candidate_similarity_std": 0.2,
        }


class SplitMetricSummaryTest(unittest.TestCase):
    def test_available_metric_columns_includes_d_score_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "d_score_top20": [0.1],
                "d_score_top20_coverage": [0.8],
            }
        )

        available = available_metric_columns(frame)

        self.assertIn("d_score_top20", available)
        self.assertIn("d_score_top20_coverage", available)

    def test_split_metric_summary_accepts_hg_and_mg_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["demo", "demo"],
                "model": ["model", "model"],
                "split": ["HG", "MG"],
                "d_score_top20": [1.0, 2.0],
                "d_score_top20_coverage": [0.8, 0.6],
            }
        )

        summary = split_metric_summary(frame, ["d_score_top20", "d_score_top20_coverage"])

        self.assertEqual(set(summary["split"]), {"HG", "MG"})
        self.assertEqual(summary.loc[summary["split"] == "MG", "d_score_top20_mean"].iat[0], 2.0)

    def test_d_score_summary_result_is_displayed_in_dedicated_cell(self) -> None:
        notebook_path = Path(__file__).parents[1] / "notebooks" / "logprob_metrics_distributions.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        summary_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("id") == "d-score-comparison-code"
        ]

        self.assertEqual(len(summary_cells), 1)
        self.assertRegex(summary_cells[0], r"\bdisplay\(\s*split_metric_summary\(")


class PlotHgMetricMeanStdKdeByModelTest(unittest.TestCase):
    def test_includes_d_score_relationship_kdes(self) -> None:
        metrics_df = pd.DataFrame([self._metric_row(0, "dataset_a", "model_a", "HG")])
        similarity_stats_df = pd.DataFrame([self._similarity_row(0, "dataset_a", "model_a", "HG")])

        with patch("notebooks.logprob_metrics_distributions_utils.sns.kdeplot") as kdeplot, patch("notebooks.logprob_metrics_distributions_utils.plt.show"):
            plot_hg_metric_mean_std_kde_by_model(metrics_df, similarity_stats_df)

        plotted_pairs = {
            (
                call.kwargs["data"]["mean"].iat[0],
                call.kwargs["data"]["std"].iat[0],
            )
            for call in kdeplot.call_args_list
        }
        self.assertTrue(
            {
                (-2.0, 0.5),
                (-2.0, 0.0),
                (0.5, 0.0),
            }.issubset(plotted_pairs)
        )

    def test_dispatches_selected_split_per_text_rows_to_kde(self) -> None:
        metrics_df = pd.DataFrame(
            [
                self._metric_row(index, "dataset_a", "model_a", "HG")
                for index in range(4)
            ]
            + [
                self._metric_row(index, "dataset_a", "model_a", "MG")
                for index in range(4, 8)
            ]
        )
        similarity_stats_df = pd.DataFrame(
            [
                self._similarity_row(index, "dataset_a", "model_a", "HG")
                for index in range(4)
            ]
            + [
                self._similarity_row(index, "dataset_a", "model_a", "MG")
                for index in range(4, 8)
            ]
        )

        with patch("notebooks.logprob_metrics_distributions_utils.sns.kdeplot") as kdeplot, patch("notebooks.logprob_metrics_distributions_utils.plt.show"):
            plot_hg_metric_mean_std_kde_by_model(metrics_df, similarity_stats_df, split="MG")

        first_call_data = kdeplot.call_args_list[0].kwargs["data"]
        self.assertEqual(len(first_call_data), 4)
        self.assertEqual(set(first_call_data["split"]), {"MG"})

    def _metric_row(self, row_id: int, dataset: str, model: str, split: str) -> dict[str, object]:
        return {
            "id": row_id,
            "dataset": dataset,
            "model": model,
            "split": split,
            "prompt_mu_score": float(row_id),
            "prompt_sigma_score": float(row_id) + 0.1,
            "perplexity": float(row_id) + 10.0,
            "d_score_top20": float(row_id) - 2.0,
            "d_score_top20_coverage": 0.5 + float(row_id) / 10,
        }

    def _similarity_row(self, row_id: int, dataset: str, model: str, split: str) -> dict[str, object]:
        return {
            "id": row_id,
            "dataset": dataset,
            "model": model,
            "split": split,
            "selected_similarity_mean": float(row_id) / 10,
            "selected_similarity_std": float(row_id) / 20,
        }


if __name__ == "__main__":
    unittest.main()
