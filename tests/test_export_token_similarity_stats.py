from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.export_token_similarity_stats import export_stale_token_similarity_stats, export_token_similarity_stats


class ExportTokenSimilarityStatsTest(unittest.TestCase):
    def test_writes_combined_stats_file_for_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            similarity_dir = processed_root / "demo__opt_350m" / "HG" / "token_similarities"
            similarity_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "id": 0,
                        "dataset": "demo",
                        "model": "opt_350m",
                        "experiment_name": "demo__opt_350m",
                        "split": "HG",
                        "token_similarities": [
                            {
                                "position": 0,
                                "token_id": 10,
                                "top_logprobs": [
                                    {"token_id": 10, "similarity_to_gt": 1.0},
                                    {"token_id": 11, "similarity_to_gt": 0.5},
                                ],
                            },
                            {
                                "position": 1,
                                "token_id": 20,
                                "top_logprobs": [
                                    {"token_id": 20, "similarity_to_gt": 0.8},
                                    {"token_id": 21, "similarity_to_gt": 0.4},
                                ],
                            },
                        ],
                    },
                    {
                        "id": 1,
                        "dataset": "demo",
                        "model": "opt_350m",
                        "experiment_name": "demo__opt_350m",
                        "split": "HG",
                        "token_similarities": [
                            {
                                "position": 0,
                                "token_id": 30,
                                "top_logprobs": [
                                    {"token_id": 30, "similarity_to_gt": None},
                                    {"token_id": 31, "similarity_to_gt": 0.25},
                                ],
                            }
                        ],
                    },
                ]
            ).to_parquet(similarity_dir / "batch_00000000_00000001.parquet", index=False)

            output_path = export_token_similarity_stats(processed_root, "demo", "opt_350m", "HG")

            self.assertEqual(output_path, processed_root / "demo__opt_350m" / "HG" / "token_similarity_stats" / "token_similarity_stats.parquet")
            stats = pd.read_parquet(output_path)
            self.assertEqual(stats["id"].tolist(), [0, 1])
            self.assertEqual(stats["selected_similarity_count"].tolist(), [2, 0])
            self.assertAlmostEqual(stats.loc[0, "selected_similarity_mean"], 0.9)
            self.assertAlmostEqual(stats.loc[0, "candidate_similarity_mean"], 0.675)
            self.assertEqual(stats.loc[1, "candidate_similarity_token_count"], 1)

    def test_auto_exports_missing_and_stale_stats_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            self._write_similarity_rows(processed_root, "dataset_a", "model_a", "HG", [0, 1])
            self._write_similarity_rows(processed_root, "dataset_a", "model_a", "MG", [0, 1])
            self._write_similarity_rows(processed_root, "dataset_b", "model_a", "HG", [0])

            stale_path = processed_root / "dataset_a__model_a" / "MG" / "token_similarity_stats" / "token_similarity_stats.parquet"
            stale_path.parent.mkdir(parents=True)
            pd.DataFrame([{"id": 0}]).to_parquet(stale_path, index=False)

            fresh_path = export_token_similarity_stats(processed_root, "dataset_b", "model_a", "HG")

            exported = export_stale_token_similarity_stats(
                processed_root,
                datasets={"dataset_a": {"splits": {"HG": "", "MG": ""}}, "dataset_b": {"splits": {"HG": ""}}},
                models={"model_a": "model-a", "model_b": "model-b"},
            )

            self.assertEqual(
                exported,
                [
                    processed_root / "dataset_a__model_a" / "HG" / "token_similarity_stats" / "token_similarity_stats.parquet",
                    processed_root / "dataset_a__model_a" / "MG" / "token_similarity_stats" / "token_similarity_stats.parquet",
                ],
            )
            self.assertEqual(pd.read_parquet(exported[0])["id"].tolist(), [0, 1])
            self.assertEqual(pd.read_parquet(stale_path)["id"].tolist(), [0, 1])
            self.assertTrue(fresh_path.exists())

    def _write_similarity_rows(self, processed_root: Path, dataset: str, model: str, split: str, ids: list[int]) -> None:
        similarity_dir = processed_root / f"{dataset}__{model}" / split / "token_similarities"
        similarity_dir.mkdir(parents=True)
        rows = [
            {
                "id": row_id,
                "dataset": dataset,
                "model": model,
                "experiment_name": f"{dataset}__{model}",
                "split": split,
                "token_similarities": [
                    {
                        "position": 0,
                        "token_id": 10,
                        "top_logprobs": [{"token_id": 10, "similarity_to_gt": 1.0}],
                    }
                ],
            }
            for row_id in ids
        ]
        pd.DataFrame(rows).to_parquet(similarity_dir / f"batch_{ids[0]:08d}_{ids[-1]:08d}.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
