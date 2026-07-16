from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from experiment_config import DATASETS, DATA_ROOT, MODEL_IDS, resolve_dataset_name, resolve_dataset_splits, resolve_model_id
from interactive_cli import input_int, select_key
from openai_logprobs import load_local_model

BATCH_SIZE = 100
DEFAULT_COUNT = 200


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


def _similarity_dir(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "token_similarities"


def _similarity_batch_path(split: str, dataset: str, model: str, first_id: int, last_id: int) -> Path:
    return _similarity_dir(split, dataset, model) / f"batch_{first_id:08d}_{last_id:08d}.parquet"


def _similarity_state_path(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "similarity_state.json"


def _read_last_processed_id(split: str, dataset: str, model: str) -> int:
    state_path = _similarity_state_path(split, dataset, model)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("last_id") is not None:
            return int(state["last_id"])

    data_dir = _similarity_dir(split, dataset, model)
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
    state_path = _similarity_state_path(split, dataset, model)
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
        "stage": "token_similarity",
        "similarity_method": "input_embedding_cosine",
        "last_id": last_id,
        "processed_count": previous_count + processed_count,
        "updated_at": _utc_now(),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_parquet_batch(split: str, dataset: str, model: str, rows: list[dict[str, Any]]) -> None:
    first_id = int(rows[0]["id"])
    last_id = int(rows[-1]["id"])
    path = _similarity_batch_path(split, dataset, model, first_id, last_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame = pd.DataFrame(rows)
    data_frame.to_parquet(path, compression="zstd", index=False)


def _write_parquet_batches(split: str, dataset: str, model: str, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BATCH_SIZE):
        _write_parquet_batch(split, dataset, model, rows[start : start + BATCH_SIZE])


def _iter_unprocessed_rows(path: Path, last_id: int, count: int) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stage-1 processed directory: {path}")

    yielded = 0
    for data_path in sorted(path.glob("*.parquet")):
        for row in pd.read_parquet(data_path).to_dict(orient="records"):
            if int(row["id"]) <= last_id:
                continue

            yield row
            yielded += 1
            if yielded >= count:
                return


def _top_logprob_similarities(
    normalized_embeddings: torch.Tensor,
    ground_truth_token_id: int | None,
    top_logprobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if ground_truth_token_id is None:
        return [
            {"token_id": top_logprob.get("token_id"), "similarity_to_gt": None}
            for top_logprob in top_logprobs
        ]

    ground_truth_embedding = normalized_embeddings[ground_truth_token_id]
    similarities: list[dict[str, Any]] = []

    for top_logprob in top_logprobs:
        candidate_token_id = top_logprob.get("token_id")
        if candidate_token_id is None:
            similarities.append({"token_id": None, "similarity_to_gt": None})
            continue

        similarity = torch.dot(normalized_embeddings[int(candidate_token_id)], ground_truth_embedding)
        similarities.append({"token_id": int(candidate_token_id), "similarity_to_gt": float(similarity.item())})

    return similarities


def _row_similarities(row: dict[str, Any], normalized_embeddings: torch.Tensor) -> dict[str, Any]:
    token_similarities = []
    for position, token in enumerate(row["tokens"]):
        token_similarities.append(
            {
                "position": position,
                "token_id": token.get("token_id"),
                "top_logprobs": _top_logprob_similarities(
                    normalized_embeddings,
                    token.get("token_id"),
                    token.get("top_logprobs", []),
                ),
            }
        )

    return {
        "id": int(row["id"]),
        "dataset": row["dataset"],
        "dataset_name": row["dataset_name"],
        "model": row["model"],
        "model_id": row["model_id"],
        "experiment_name": row["experiment_name"],
        "experiment_params": row["experiment_params"],
        "split": row["split"],
        "similarity_method": "input_embedding_cosine",
        "token_similarities": token_similarities,
        "processed_at": _utc_now(),
    }


def _process_split(
    split: str,
    count: int,
    normalized_embeddings: torch.Tensor,
    dataset: str,
    dataset_name: str,
    model_name: str,
    model_id: str,
) -> None:
    last_id = _read_last_processed_id(split, dataset, model_name)
    print(f"{split}: reading stage-1 rows after id {last_id}", flush=True)
    rows = list(_iter_unprocessed_rows(_data_dir(split, dataset, model_name), last_id, count))

    if not rows:
        print(f"{split}: no stage-1 rows left after id {last_id}")
        return

    records = [
        _row_similarities(row, normalized_embeddings)
        for row in tqdm(rows, desc=f"{split}: similarity", unit="text")
    ]
    _write_parquet_batches(split, dataset, model_name, records)
    _write_state(split, dataset, dataset_name, model_name, model_id, int(records[-1]["id"]), len(records))

    print(
        f"{split}: processed {len(records)} similarity rows, "
        f"experiment={_experiment_name(dataset, model_name)}, last_id={int(records[-1]['id'])}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: compute token embedding similarities for processed logprob Parquet files.")
    parser.add_argument("--count", type=int, default=None, help="Number of processed rows to process per split.")
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
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.model is None and args.model_id is None:
        args.model = select_key("Select model", dict(sorted(MODEL_IDS.items())))

    if args.dataset is None:
        args.dataset = select_key("Select dataset", {key: DATASETS[key] for key in sorted(DATASETS)})

    if args.count is None:
        args.count = input_int("Count per split", DEFAULT_COUNT)

    return args


def run_stage_2(dataset: str, model_name: str, model_id: str, count: int) -> None:
    dataset_name = resolve_dataset_name(dataset)
    dataset_splits = resolve_dataset_splits(dataset)

    print(f"Loading model for stage 2: {model_id}", flush=True)
    _, model, _ = load_local_model(model_id)
    print("Normalizing input embeddings", flush=True)
    normalized_embeddings = F.normalize(model.get_input_embeddings().weight.detach(), dim=-1)

    for split in dataset_splits:
        _process_split(split, count, normalized_embeddings, dataset, dataset_name, model_name, model_id)


def main() -> None:
    args = _resolve_args(_parse_args())
    count = args.count
    model_id = args.model_id if args.model_id is not None else resolve_model_id(args.model)
    model_name = args.model if args.model_id is None else args.model_id
    run_stage_2(args.dataset, model_name, model_id, count)


if __name__ == "__main__":
    main()
