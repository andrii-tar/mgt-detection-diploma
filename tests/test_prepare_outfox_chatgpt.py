from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.extract_data.prepare_outfox_chatgpt import prepare_outfox_chatgpt


class PrepareOutfoxChatgptTest(unittest.TestCase):
    def test_writes_split_and_all_hg_mg_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            input_root = data_root / "raw" / "outfox_raw"
            output_root = data_root / "raw" / "outfox"
            fixtures = {
                "train": (["human train 1", "human train 2"], ["machine train 1", "machine train 2"]),
                "valid": (["human valid 1"], ["machine valid 1"]),
                "test": (["human test 1"], ["machine test 1"]),
            }

            for split, (humans, machines) in fixtures.items():
                common_dir = input_root / "common" / split
                chatgpt_dir = input_root / "chatgpt" / split
                common_dir.mkdir(parents=True, exist_ok=True)
                chatgpt_dir.mkdir(parents=True, exist_ok=True)
                with (common_dir / f"{split}_humans.pkl").open("wb") as file:
                    pickle.dump(humans, file)
                with (chatgpt_dir / f"{split}_lms.pkl").open("wb") as file:
                    pickle.dump(machines, file)

            prepare_outfox_chatgpt(input_root, output_root)

            expected_files = [
                "outfox_chatgpt_train_HG.csv",
                "outfox_chatgpt_train_MG.csv",
                "outfox_chatgpt_valid_HG.csv",
                "outfox_chatgpt_valid_MG.csv",
                "outfox_chatgpt_test_HG.csv",
                "outfox_chatgpt_test_MG.csv",
                "outfox_chatgpt_all_HG.csv",
                "outfox_chatgpt_all_MG.csv",
            ]
            for file_name in expected_files:
                self.assertTrue((output_root / file_name).exists(), file_name)

            train_hg = pd.read_csv(output_root / "outfox_chatgpt_train_HG.csv")
            self.assertEqual(
                list(train_hg.columns),
                ["id", "text", "source_llm", "split", "origin", "source_dataset"],
            )
            self.assertEqual(train_hg["id"].tolist(), [0, 1])
            self.assertEqual(train_hg["text"].tolist(), ["human train 1", "human train 2"])
            self.assertEqual(train_hg["origin"].tolist(), ["HG", "HG"])

            all_hg = pd.read_csv(output_root / "outfox_chatgpt_all_HG.csv")
            all_mg = pd.read_csv(output_root / "outfox_chatgpt_all_MG.csv")
            self.assertEqual(all_hg["id"].tolist(), [0, 1, 2, 3])
            self.assertEqual(all_mg["id"].tolist(), [0, 1, 2, 3])
            self.assertEqual(all_hg["split"].tolist(), ["train", "train", "valid", "test"])
            self.assertEqual(all_mg["text"].tolist(), ["machine train 1", "machine train 2", "machine valid 1", "machine test 1"])


if __name__ == "__main__":
    unittest.main()
