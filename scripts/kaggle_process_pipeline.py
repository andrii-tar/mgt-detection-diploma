from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from openai_logprobs import DEFAULT_MODEL_ID, load_local_model, score_texts_with_model


DATASET_NAME = "kaggle_train"
RAW_SPLITS = {
    "HG": Path("data/raw/kaggle_train_hg.csv"),
    "MG": Path("data/raw/kaggle_train_mg.csv"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_dir(split: str) -> Path:
    return Path("data/processed") / f"{DATASET_NAME}__{split}"


def _data_path(split: str) -> Path:
    return _split_dir(split) / "data.jsonl"


def _state_path(split: str) -> Path:
    return _split_dir(split) / "state.json"


def _read_last_processed_id(split: str) -> int:
    state_path = _state_path(split)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("last_id") is not None:
            return int(state["last_id"])

    data_path = _data_path(split)
    if not data_path.exists():
        return -1

    last_id = -1
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_id = int(json.loads(line)["id"])

    return last_id


def _write_state(split: str, last_id: int, processed_count: int) -> None:
    state_path = _state_path(split)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    previous_count = 0
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_count = int(previous_state.get("processed_count", 0))

    payload = {
        "dataset_name": DATASET_NAME,
        "split": split,
        "last_id": last_id,
        "processed_count": previous_count + processed_count,
        "updated_at": _utc_now(),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_split_frame(split: str) -> pd.DataFrame:
    path = RAW_SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing input split file: {path}")

    frame = pd.read_csv(path)
    if "id" not in frame.columns or "text" not in frame.columns:
        raise ValueError(f"Expected columns 'id' and 'text' in {path}")

    return frame.sort_values("id").reset_index(drop=True)


def _select_next_rows(frame: pd.DataFrame, last_id: int, count: int) -> pd.DataFrame:
    next_rows = frame[frame["id"] > last_id].head(count)
    return next_rows.reset_index(drop=True)


def _process_split(
    split: str,
    count: int,
    tokenizer: Any,
    model: Any,
    device: Any,
    top_n: int,
) -> None:
    frame = _load_split_frame(split)
    last_id = _read_last_processed_id(split)
    batch = _select_next_rows(frame, last_id, count)

    if batch.empty:
        print(f"{split}: no unprocessed rows left after id {last_id}")
        return

    texts = batch["text"].tolist()
    scored = score_texts_with_model(texts, tokenizer, model, device, top_n=top_n)

    records: list[dict[str, Any]] = []
    for row, scored_row in zip(batch.to_dict(orient="records"), scored):
        records.append(
            {
                "id": int(row["id"]),
                "dataset_name": DATASET_NAME,
                "split": split,
                "text": row["text"],
                "model_id": DEFAULT_MODEL_ID,
                "tokens": scored_row["tokens"],
                "processed_at": _utc_now(),
            }
        )

    _append_jsonl(_data_path(split), records)
    _write_state(split, int(batch["id"].iloc[-1]), len(records))

    print(f"{split}: processed {len(records)} rows, last_id={int(batch['id'].iloc[-1])}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process next HG and MG texts into JSONL logprob files.")
    parser.add_argument("--count", type=int, default=None, help="Number of rows to process per split.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top alternatives per token.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = args.count if args.count is not None else int(input("How many texts per split should be processed? "))

    tokenizer, model, device = load_local_model(DEFAULT_MODEL_ID)

    for split in ("HG", "MG"):
        _process_split(split, count, tokenizer, model, device, args.top_n)


if __name__ == "__main__":
    main()
