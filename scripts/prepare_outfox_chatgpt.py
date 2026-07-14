from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


SPLITS = ("train", "valid", "test")
CSV_COLUMNS = ["id", "text", "source_llm", "split", "origin", "source_dataset"]


def _data_root() -> Path:
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value is None:
        raise RuntimeError("TEXT_DETECTION_DATA is not set")
    return Path(value)


def _load_pickle_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("rb") as file:
        value = pickle.load(file)

    if not isinstance(value, list):
        raise TypeError(f"Expected list in {path}, got {type(value).__name__}")
    return value


def _build_frame(texts: list[str], split: str, origin: str, start_id: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text at {split}/{origin} index {offset}")
        rows.append(
            {
                "id": start_id + offset,
                "text": text,
                "source_llm": "chatgpt",
                "split": split,
                "origin": origin,
                "source_dataset": "outfox",
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def prepare_outfox_chatgpt(input_root: Path, output_root: Path) -> None:
    all_hg_frames: list[pd.DataFrame] = []
    all_mg_frames: list[pd.DataFrame] = []
    next_all_id = 0

    for split in SPLITS:
        humans_path = input_root / "common" / split / f"{split}_humans.pkl"
        machines_path = input_root / "chatgpt" / split / f"{split}_lms.pkl"
        humans = _load_pickle_list(humans_path)
        machines = _load_pickle_list(machines_path)

        if len(humans) != len(machines):
            raise ValueError(f"Mismatched {split} lengths: HG={len(humans)} MG={len(machines)}")

        hg_frame = _build_frame(humans, split, "HG")
        mg_frame = _build_frame(machines, split, "MG")
        _write_csv(hg_frame, output_root / f"outfox_chatgpt_{split}_HG.csv")
        _write_csv(mg_frame, output_root / f"outfox_chatgpt_{split}_MG.csv")

        all_hg_frames.append(_build_frame(humans, split, "HG", next_all_id))
        all_mg_frames.append(_build_frame(machines, split, "MG", next_all_id))
        next_all_id += len(humans)

        print(f"{split}: HG={len(humans)} MG={len(machines)}")

    all_hg = pd.concat(all_hg_frames, ignore_index=True)
    all_mg = pd.concat(all_mg_frames, ignore_index=True)
    _write_csv(all_hg, output_root / "outfox_chatgpt_all_HG.csv")
    _write_csv(all_mg, output_root / "outfox_chatgpt_all_MG.csv")
    print(f"all: HG={len(all_hg)} MG={len(all_mg)}")


def _parse_args() -> argparse.Namespace:
    data_root = _data_root()
    parser = argparse.ArgumentParser(description="Prepare OUTFOX ChatGPT HG/MG CSV files for processing.")
    parser.add_argument("--input-root", type=Path, default=data_root / "raw" / "outfox_raw")
    parser.add_argument("--output-root", type=Path, default=data_root / "raw" / "outfox")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_outfox_chatgpt(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
