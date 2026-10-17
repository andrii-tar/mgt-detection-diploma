from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_COLUMNS = ["id", "prompt_id", "text", "model"]
OUTPUT_COLUMNS = ["id", "source_id", "text", "prompt_id", "source_llm", "origin", "source_dataset"]


def prepare_name_that_llm(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    frame = pd.read_csv(input_path, dtype={"id": str, "prompt_id": str, "model": str})
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError("source id values must be unique")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("text contains blank values")
    if frame["model"].isna().any() or frame["model"].astype(str).str.strip().eq("").any():
        raise ValueError("model contains blank values")

    result = pd.DataFrame(
        {
            "id": range(len(frame)),
            "source_id": frame["id"],
            "text": frame["text"],
            "prompt_id": frame["prompt_id"],
            "source_llm": frame["model"],
            "origin": "MG",
            "source_dataset": "name_that_llm",
        },
        columns=OUTPUT_COLUMNS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Name That LLM MG CSV data for processing.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    if args.input_path is None or args.output_path is None:
        from scripts.data_root import resolve_data_root

        data_root = resolve_data_root()
        if args.input_path is None:
            args.input_path = data_root / "raw" / "name-that-llm" / "train.csv"
        if args.output_path is None:
            args.output_path = data_root / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
    return args


def main() -> None:
    args = _parse_args()
    prepare_name_that_llm(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
