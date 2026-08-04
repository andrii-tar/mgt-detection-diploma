from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_COLUMNS = ["LLM_origin", "text"]
OUTPUT_COLUMNS = ["id", "text", "LLM_origin"]


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return frame


def _build_split(source: pd.DataFrame, is_human: bool) -> pd.DataFrame:
    split = source[source["LLM_origin"].eq("human") == is_human][INPUT_COLUMNS].copy()
    split.insert(0, "id", range(len(split)))
    return split[OUTPUT_COLUMNS]


def prepare_custom(input_path: Path, output_root: Path) -> None:
    source = _read_input(input_path)
    human = _build_split(source, is_human=True)
    machine = _build_split(source, is_human=False)

    output_root.mkdir(parents=True, exist_ok=True)
    human.to_csv(output_root / "custom_2025_HG.csv", index=False)
    machine.to_csv(output_root / "custom_2025_MG.csv", index=False)
    print(f"custom_2025: HG={len(human)}, MG={len(machine)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare custom 2025 HG and MG CSV files for processing.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    if args.input_path is None or args.output_root is None:
        from scripts.data_root import resolve_data_root

        data_root = resolve_data_root()
        if args.input_path is None:
            args.input_path = data_root / "raw" / "custom" / "samples_2025" / "custom_examples.tsv"
        if args.output_root is None:
            args.output_root = data_root / "raw" / "custom" / "samples_2025"
    return args


def main() -> None:
    args = _parse_args()
    prepare_custom(args.input_path, args.output_root)


if __name__ == "__main__":
    main()
