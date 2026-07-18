from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from notebooks.logprob_metrics_distributions_utils import load_token_similarity_stats, plot_token_similarity


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


if __name__ == "__main__":
    unittest.main()
