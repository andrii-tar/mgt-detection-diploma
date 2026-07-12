from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_config import DATASETS, DATA_ROOT, MODEL_IDS, resolve_dataset_name, resolve_dataset_splits, resolve_model_id
from openai_logprobs import load_local_model, score_texts_with_model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _experiment_name(dataset: str, model: str) -> str:
    return f"{_path_slug(dataset)}__{_path_slug(model)}"


def _experiment_params(dataset: str, dataset_name: str, model: str, model_id: str) -> dict[str, str]:
    return {"dataset": dataset, "dataset_name": dataset_name, "model": model, "model_id": model_id}


def _split_dir(split: str, dataset: str, model: str) -> Path:
    return DATA_ROOT / "processed" / _experiment_name(dataset, model) / split


def _data_path(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "data.jsonl"


def _state_path(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "state.json"


def _read_last_processed_id(split: str, dataset: str, model: str) -> int:
    state_path = _state_path(split, dataset, model)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("last_id") is not None:
            return int(state["last_id"])

    data_path = _data_path(split, dataset, model)
    if not data_path.exists():
        return -1

    last_id = -1
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_id = int(json.loads(line)["id"])

    return last_id


def _write_state(
    split: str,
    dataset: str,
    dataset_name: str,
    model: str,
    model_id: str,
    last_id: int,
    processed_count: int,
) -> None:
    state_path = _state_path(split, dataset, model)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    previous_count = 0
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_count = int(previous_state.get("processed_count", 0))

    payload = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": model,
        "model_id": model_id,
        "experiment_name": _experiment_name(dataset, model),
        "experiment_params": _experiment_params(dataset, dataset_name, model, model_id),
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


def _load_split_frame(split: str, split_path: str) -> pd.DataFrame:
    path = Path(split_path)
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
    dataset: str,
    dataset_name: str,
    split_path: str,
    model_name: str,
    model_id: str,
    top_n: int,
) -> None:
    frame = _load_split_frame(split, split_path)
    last_id = _read_last_processed_id(split, dataset, model_name)
    batch = _select_next_rows(frame, last_id, count)

    if batch.empty:
        print(f"{split}: no unprocessed rows left after id {last_id}")
        return

    texts = batch["text"].tolist()
    scored = score_texts_with_model(texts, tokenizer, model, device, top_n=top_n, progress_desc=f"{split}: scoring")

    records: list[dict[str, Any]] = []
    rows = batch.to_dict(orient="records")
    for row, scored_row in zip(rows, scored):
        records.append(
            {
                "id": int(row["id"]),
                "dataset": dataset,
                "dataset_name": dataset_name,
                "model": model_name,
                "model_id": model_id,
                "experiment_name": _experiment_name(dataset, model_name),
                "experiment_params": _experiment_params(dataset, dataset_name, model_name, model_id),
                "split": split,
                "text": row["text"],
                "tokens": scored_row["tokens"],
                "processed_at": _utc_now(),
            }
        )

    _append_jsonl(_data_path(split, dataset, model_name), records)
    _write_state(split, dataset, dataset_name, model_name, model_id, int(batch["id"].iloc[-1]), len(records))

    print(
        f"{split}: processed {len(records)} rows, "
        f"experiment={_experiment_name(dataset, model_name)}, last_id={int(batch['id'].iloc[-1])}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process next HG and MG texts into JSONL logprob files.")
    parser.add_argument("--count", type=int, default=None, help="Number of rows to process per split.")
    parser.add_argument(
        "--dataset",
        default="kaggle_train",
        choices=sorted(DATASETS),
        help="Dataset id from experiment_config.py.",
    )
    parser.add_argument(
        "--model",
        default="smollm2_360m",
        choices=sorted(MODEL_IDS),
        help="Model id alias from experiment_config.py.",
    )
    parser.add_argument("--model-id", default=None, help="Raw Hugging Face model id. Overrides --model.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top alternatives per token.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = args.count if args.count is not None else int(input("How many texts per split should be processed? "))
    dataset_name = resolve_dataset_name(args.dataset)
    dataset_splits = resolve_dataset_splits(args.dataset)
    model_id = args.model_id if args.model_id is not None else resolve_model_id(args.model)
    model_name = args.model if args.model_id is None else args.model_id

    tokenizer, model, device = load_local_model(model_id)

    for split, split_path in dataset_splits.items():
        _process_split(
            split,
            count,
            tokenizer,
            model,
            device,
            args.dataset,
            dataset_name,
            split_path,
            model_name,
            model_id,
            args.top_n,
        )


if __name__ == "__main__":
    main()
