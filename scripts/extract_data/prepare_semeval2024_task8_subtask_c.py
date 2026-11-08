from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


INPUT_FILES = {"train": "subtaskC_train.jsonl", "dev": "subtaskC_dev.jsonl"}
OUTPUT_COLUMNS = [
    "id",
    "text",
    "boundary_word_index",
    "source_id",
    "split",
    "origin",
    "source_dataset",
]


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(record)
    return records


def _build_frame(records: list[dict[str, object]], split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_ids: set[str] = set()
    for index, record in enumerate(records):
        if set(record) != {"id", "text", "label"}:
            raise ValueError(f"Expected id, text, and label fields at {split} index {index}")

        source_id = record["id"]
        text = record["text"]
        boundary = record["label"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"id must be a non-empty string at {split} index {index}")
        if source_id in source_ids:
            raise ValueError(f"id values must be unique within {split}")
        source_ids.add(source_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"text must be non-blank at {split} index {index}")
        if not isinstance(boundary, int) or isinstance(boundary, bool):
            raise ValueError(f"boundary must be an integer at {split} index {index}")
        if not 0 <= boundary < len(text.split(" ")):
            raise ValueError(f"boundary is outside the text word range at {split} index {index}")

        rows.append(
            {
                "id": index,
                "text": text,
                "boundary_word_index": boundary,
                "source_id": source_id,
                "split": split,
                "origin": "MG",
                "source_dataset": "semeval_2024_task8_subtask_c",
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def prepare_semeval2024_task8_subtask_c(input_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split, filename in INPUT_FILES.items():
        frame = _build_frame(_load_records(input_root / filename), split)
        frame.to_csv(output_root / f"SemEval2024_task8_C_{split}_MG.csv", index=False)
        print(f"{split}: MG={len(frame)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SemEval 2024 Task 8 Subtask C MG CSV files.")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.input_root is None or args.output_root is None:
        from scripts.data_root import resolve_data_root

        subtask_root = resolve_data_root() / "raw" / "semevals_2024-task8" / "SemEval2024-task8" / "subtaskC"
        if args.input_root is None:
            args.input_root = subtask_root
        if args.output_root is None:
            args.output_root = subtask_root
    return args


def main() -> None:
    args = _parse_args()
    prepare_semeval2024_task8_subtask_c(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
