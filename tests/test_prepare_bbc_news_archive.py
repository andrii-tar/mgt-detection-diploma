from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.extract_data.prepare_bbc_news_archive import prepare_bbc_news_archive


class PrepareBbcNewsArchiveTest(unittest.TestCase):
    def test_writes_content_and_title_hg_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            input_path = data_root / "raw" / "bbc-news-data.csv"
            output_root = data_root / "raw" / "bbc_news"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "category": "business",
                        "filename": "001.txt",
                        "title": "Ad sales boost Time Warner profit",
                        "content": "Quarterly profits jumped.",
                    },
                    {
                        "category": "sport",
                        "filename": "002.txt",
                        "title": "Cup holders progress",
                        "content": "The holders won again.",
                    },
                ]
            ).to_csv(input_path, sep="\t", index=False)

            prepare_bbc_news_archive(input_path, output_root)

            content = pd.read_csv(output_root / "bbc_news_content_HG.csv")
            titles = pd.read_csv(output_root / "bbc_news_titles_HG.csv")

            expected_columns = [
                "id",
                "text",
                "category",
                "filename",
                "title",
                "source_llm",
                "split",
                "origin",
                "source_dataset",
            ]
            self.assertEqual(list(content.columns), expected_columns)
            self.assertEqual(list(titles.columns), expected_columns)
            self.assertEqual(content["id"].tolist(), [0, 1])
            self.assertEqual(content["text"].tolist(), ["Quarterly profits jumped.", "The holders won again."])
            self.assertEqual(titles["text"].tolist(), ["Ad sales boost Time Warner profit", "Cup holders progress"])
            self.assertEqual(content["origin"].tolist(), ["HG", "HG"])
            self.assertEqual(titles["source_dataset"].tolist(), ["bbc_news_titles", "bbc_news_titles"])


if __name__ == "__main__":
    unittest.main()
