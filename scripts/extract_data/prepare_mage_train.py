from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"text", "label", "src"}
OUTPUT_COLUMNS = ["id", "text", "src"]


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    unexpected_labels = sorted(set(frame["label"].dropna()) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Unexpected Label values: {unexpected_labels}")

    if frame["label"].isna().any():
        raise ValueError("label contains null values")
    if frame["src"].isna().any() or frame["src"].astype(str).str.strip().eq("").any():
        raise ValueError("src contains blank values")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("text contains blank values")


def _split_frame(frame: pd.DataFrame, label: int) -> pd.DataFrame:
    split = frame.loc[frame["label"] == label, ["text", "src"]].copy()
    split = split.reset_index(drop=True)
    split.insert(0, "id", split.index)
    return split[OUTPUT_COLUMNS]


def prepare_mage_train(input_path: Path, output_root: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    frame = pd.read_csv(input_path)
    _validate_frame(frame)

    output_root.mkdir(parents=True, exist_ok=True)
    _split_frame(frame, 1).to_csv(output_root / "MAGE_train_HG.csv", index=False)
    _split_frame(frame, 0).to_csv(output_root / "MAGE_train_MG.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MAGE train HG/MG CSV files for processing.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.input_path is None or args.output_root is None:
        from scripts.data_root import resolve_data_root

        data_root = resolve_data_root()
        if args.input_path is None:
            args.input_path = data_root / "raw" / "MAGE_train.csv"
        if args.output_root is None:
            args.output_root = data_root / "raw" / "MAGE" / "train"
    return args


def main() -> None:
    args = _parse_args()
    prepare_mage_train(args.input_path, args.output_root)


if __name__ == "__main__":
    main()
