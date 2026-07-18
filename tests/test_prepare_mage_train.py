from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.extract_data.prepare_mage_train import prepare_mage_train


class PrepareMageTrainTest(unittest.TestCase):
    def test_writes_hg_and_mg_corpuses_from_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            input_path = data_root / "raw" / "MAGE_train.csv"
            output_root = data_root / "raw" / "MAGE" / "train"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"text": "machine text", "label": 0, "src": "source-a"},
                    {"text": "human text", "label": 1, "src": "source-b"},
                    {"text": "another machine text", "label": 0, "src": "source-c"},
                ]
            ).to_csv(input_path, index=False)

            prepare_mage_train(input_path, output_root)

            hg = pd.read_csv(output_root / "MAGE_train_HG.csv")
            mg = pd.read_csv(output_root / "MAGE_train_MG.csv")

            self.assertEqual(list(hg.columns), ["id", "text", "src"])
            self.assertEqual(list(mg.columns), ["id", "text", "src"])
            self.assertEqual(hg.to_dict("records"), [{"id": 0, "text": "human text", "src": "source-b"}])
            self.assertEqual(
                mg.to_dict("records"),
                [
                    {"id": 0, "text": "machine text", "src": "source-a"},
                    {"id": 1, "text": "another machine text", "src": "source-c"},
                ],
            )

    def test_rejects_labels_outside_hg_and_mg_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            input_path = data_root / "raw" / "MAGE_train.csv"
            output_root = data_root / "raw" / "MAGE" / "train"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"text": "bad label", "label": 2, "src": "source-a"}]).to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Unexpected Label values"):
                prepare_mage_train(input_path, output_root)


if __name__ == "__main__":
    unittest.main()
