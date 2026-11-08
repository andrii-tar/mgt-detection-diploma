from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

if __package__:
    from .experiment_config import DATASETS, DATA_ROOT, MODEL_IDS, resolve_dataset_name, resolve_dataset_splits, resolve_model_id, resolve_model_name
    from .data_root import resolve_data_root
    from .interactive_cli import input_int, optional_choice, select_key
    from .openai_logprobs import load_local_model, score_texts_with_model
    from .token_similarity_stage2 import run_stage_2
else:
    from experiment_config import DATASETS, DATA_ROOT, MODEL_IDS, resolve_dataset_name, resolve_dataset_splits, resolve_model_id, resolve_model_name
    from data_root import resolve_data_root
    from interactive_cli import input_int, optional_choice, select_key
    from openai_logprobs import load_local_model, score_texts_with_model
    from token_similarity_stage2 import run_stage_2

BATCH_SIZE = 30
DEFAULT_COUNT = 100
DEFAULT_INFERENCE_BATCH_SIZE = 2
DEFAULT_WRITE_BATCH_SIZE = 20
DEFAULT_TOP_N = 20


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


def _data_dir(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "data"


def _data_batch_path(split: str, dataset: str, model: str, first_id: int, last_id: int) -> Path:
    return _data_dir(split, dataset, model) / f"batch_{first_id:08d}_{last_id:08d}.parquet"


def _state_path(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "state.json"


def _read_last_processed_id(split: str, dataset: str, model: str) -> int:
    state_path = _state_path(split, dataset, model)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("last_id") is not None:
            return int(state["last_id"])

    data_dir = _data_dir(split, dataset, model)
    if not data_dir.exists():
        return -1

    last_id = -1
    for data_path in data_dir.glob("*.parquet"):
        data_frame = pd.read_parquet(data_path, columns=["id"])
        if not data_frame.empty:
            last_id = max(last_id, int(data_frame["id"].max()))

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


def _write_parquet_batch(split: str, dataset: str, model: str, rows: list[dict[str, Any]]) -> None:
    first_id = int(rows[0]["id"])
    last_id = int(rows[-1]["id"])
    path = _data_batch_path(split, dataset, model, first_id, last_id)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(temp_path, compression="zstd", index=False)
    temp_path.replace(path)


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
    inference_batch_size: int,
    write_batch_size: int,
) -> None:
    frame = _load_split_frame(split, split_path)
    last_id = _read_last_processed_id(split, dataset, model_name)
    remaining_count = count
    total_processed = 0

    with tqdm(total=count, desc=f"{split}: processing", unit="text") as progress:
        while remaining_count > 0:
            batch = _select_next_rows(frame, last_id, min(write_batch_size, remaining_count))

            if batch.empty:
                if total_processed == 0:
                    print(f"{split}: no unprocessed rows left after id {last_id}")
                break

            texts = batch["text"].tolist()
            scored = score_texts_with_model(
                texts,
                tokenizer,
                model,
                device,
                top_n=top_n,
                inference_batch_size=inference_batch_size,
                progress_desc=None,
            )

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

            _write_parquet_batch(split, dataset, model_name, records)
            processed_count = len(records)
            last_id = int(batch["id"].iloc[-1])
            total_processed += processed_count
            remaining_count -= processed_count
            _write_state(split, dataset, dataset_name, model_name, model_id, last_id, processed_count)
            progress.update(processed_count)

    if total_processed:
        print(f"{split}: processed {total_processed} rows, experiment={_experiment_name(dataset, model_name)}, last_id={last_id}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process next HG and MG texts into Parquet logprob files.")
    parser.add_argument("--count", type=int, default=None, help="Number of rows to process per split.")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASETS),
        help="Dataset id from experiment_config.py.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(MODEL_IDS),
        help="Model id alias from experiment_config.py.",
    )
    parser.add_argument("--model-id", default=None, help="Raw Hugging Face model id. Overrides --model.")
    parser.add_argument("--split", choices=("HG", "MG"), default=None, help="Only process one split. Defaults to both splits.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top alternatives per token.")
    parser.add_argument("--inference-batch-size", type=int, default=None, help="Number of texts per model forward pass.")
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=None,
        help="Number of processed texts to score and write per Parquet file.",
    )
    parser.add_argument(
        "--no-stage-2",
        action="store_false",
        dest="run_stage_2",
        help="Only run stage 1 logprob processing; skip token similarity stage 2.",
    )
    parser.set_defaults(run_stage_2=True)
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    is_interactive = args.model is None and args.model_id is None and args.dataset is None

    if args.model is None and args.model_id is None:
        args.model = select_key("Select model", dict(sorted(MODEL_IDS.items())))

    if args.dataset is None:
        args.dataset = select_key("Select dataset", {key: DATASETS[key] for key in sorted(DATASETS)})

    if args.count is None:
        args.count = input_int("Count per split", DEFAULT_COUNT)

    if args.split is None and is_interactive:
        args.split = optional_choice("Split", ["HG", "MG"])

    if args.inference_batch_size is None:
        args.inference_batch_size = input_int("Inference batch size", DEFAULT_INFERENCE_BATCH_SIZE)

    if args.write_batch_size is None:
        args.write_batch_size = input_int("Write batch size", DEFAULT_WRITE_BATCH_SIZE)

    return args


def main(args: argparse.Namespace | None = None) -> None:
    global DATA_ROOT
    DATA_ROOT = resolve_data_root()
    args = _resolve_args(_parse_args() if args is None else args)
    if args.write_batch_size < 1:
        raise ValueError("--write-batch-size must be at least 1")

    count = args.count
    dataset_name = resolve_dataset_name(args.dataset)
    dataset_splits = resolve_dataset_splits(args.dataset, DATA_ROOT)
    selected_splits = [args.split] if args.split is not None else list(dataset_splits)
    model_id = args.model_id if args.model_id is not None else resolve_model_id(args.model)
    model_name = resolve_model_name(args.model if args.model_id is None else args.model_id)

    tokenizer, model, device = load_local_model(model_id)

    for split in selected_splits:
        split_path = dataset_splits[split]
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
            args.inference_batch_size,
            args.write_batch_size,
        )

    if args.run_stage_2:
        run_stage_2(dataset=args.dataset, model_name=model_name, model_id=model_id, count=count * len(selected_splits), splits=selected_splits)


if __name__ == "__main__":
    main()
