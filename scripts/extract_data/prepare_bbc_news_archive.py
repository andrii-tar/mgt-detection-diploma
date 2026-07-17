from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


CSV_COLUMNS = ["id", "text", "category", "filename", "title", "source_llm", "split", "origin", "source_dataset"]
INPUT_COLUMNS = ["category", "filename", "title", "content"]


def _data_root() -> Path:
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value is None:
        raise RuntimeError("TEXT_DETECTION_DATA is not set")
    return Path(value)


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return frame.reset_index(drop=True)


def _build_frame(source: pd.DataFrame, text_column: str, source_dataset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in source.itertuples(index=True):
        text = getattr(row, text_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid {text_column} at index {row.Index}")
        rows.append(
            {
                "id": row.Index,
                "text": text.strip(),
                "category": row.category,
                "filename": row.filename,
                "title": row.title,
                "source_llm": "",
                "split": "all",
                "origin": "HG",
                "source_dataset": source_dataset,
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def prepare_bbc_news_archive(input_path: Path, output_root: Path) -> None:
    source = _read_input(input_path)
    content = _build_frame(source, "content", "bbc_news_content")
    titles = _build_frame(source, "title", "bbc_news_titles")

    _write_csv(content, output_root / "bbc_news_content_HG.csv")
    _write_csv(titles, output_root / "bbc_news_titles_HG.csv")
    print(f"bbc_news_content: HG={len(content)}")
    print(f"bbc_news_titles: HG={len(titles)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BBC News Archive HG CSV files for processing.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.input_path is None or args.output_root is None:
        data_root = _data_root()
        if args.input_path is None:
            args.input_path = data_root / "raw" / "bbc-news-data.csv"
        if args.output_root is None:
            args.output_root = data_root / "raw" / "bbc_news"
    return args


def main() -> None:
    args = _parse_args()
    prepare_bbc_news_archive(args.input_path, args.output_root)


if __name__ == "__main__":
    main()
