from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATA_ROOT: Path | None = None
DATASETS: dict[str, Any] = {}
MODEL_IDS: dict[str, str] = {}
resolve_dataset_name: Any = None
resolve_dataset_splits: Any = None
resolve_model_id: Any = None


def _load_experiment_config() -> None:
    global DATA_ROOT, DATASETS, MODEL_IDS, resolve_dataset_name, resolve_dataset_splits, resolve_model_id
    if DATA_ROOT is not None:
        return
    try:
        from experiment_config import DATASETS as datasets
        from experiment_config import DATA_ROOT as data_root
        from experiment_config import MODEL_IDS as model_ids
        from experiment_config import resolve_dataset_name as dataset_name_resolver
        from experiment_config import resolve_dataset_splits as dataset_splits_resolver
        from experiment_config import resolve_model_id as model_id_resolver
    except ModuleNotFoundError:
        from scripts.experiment_config import DATASETS as datasets
        from scripts.experiment_config import DATA_ROOT as data_root
        from scripts.experiment_config import MODEL_IDS as model_ids
        from scripts.experiment_config import resolve_dataset_name as dataset_name_resolver
        from scripts.experiment_config import resolve_dataset_splits as dataset_splits_resolver
        from scripts.experiment_config import resolve_model_id as model_id_resolver

    DATA_ROOT = data_root
    DATASETS = datasets
    MODEL_IDS = model_ids
    resolve_dataset_name = dataset_name_resolver
    resolve_dataset_splits = dataset_splits_resolver
    resolve_model_id = model_id_resolver


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _experiment_name(dataset: str, model: str) -> str:
    return f"{_path_slug(dataset)}__{_path_slug(model)}"


def _split_dir(split: str, dataset: str, model: str) -> Path:
    if DATA_ROOT is None:
        raise RuntimeError("experiment config is not loaded")
    return DATA_ROOT / "processed" / _experiment_name(dataset, model) / split


def _data_dir(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "data"


def _metrics_dir(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "metrics"


def _metrics_state_path(split: str, dataset: str, model: str) -> Path:
    return _split_dir(split, dataset, model) / "metrics_state.json"


def _text_controls(text: str) -> dict[str, int]:
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]
    return {
        "text_len_symbols": len(text),
        "text_len_words": len(text.split()),
        "text_len_sentences": len(sentences),
    }


def _finite_logprobs(tokens: list[dict[str, Any]]) -> list[float]:
    logprobs = [float(token["logprob"]) for token in tokens if token.get("logprob") is not None]
    if not logprobs:
        raise ValueError("expected at least one token logprob")
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("token logprobs must be finite")
    return logprobs


def compute_logprob_metrics(row: dict[str, Any]) -> dict[str, Any]:
    logprobs = _finite_logprobs(row.get("tokens", []))
    mean = sum(logprobs) / len(logprobs)
    variance = sum((value - mean) ** 2 for value in logprobs) / len(logprobs)

    text = row.get("text") or ""
    metrics = {
        "id": int(row["id"]),
        "split": row.get("split"),
        "perplexity": math.exp(-mean),
        "prompt_mu_score": mean,
        "prompt_sigma_score": math.sqrt(variance),
        "num_tokens": len(logprobs),
        "metrics_version": "logprob_metrics_v1",
        "processed_at": _utc_now(),
    }
    metrics.update(_text_controls(text))
    return metrics


def _read_last_processed_id(split: str, dataset: str, model: str) -> int:
    state_path = _metrics_state_path(split, dataset, model)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("last_id") is not None:
            return int(state["last_id"])

    metrics_dir = _metrics_dir(split, dataset, model)
    if not metrics_dir.exists():
        return -1

    last_id = -1
    for data_path in metrics_dir.glob("*.parquet"):
        data_frame = pd.read_parquet(data_path, columns=["id"])
        if not data_frame.empty:
            last_id = max(last_id, int(data_frame["id"].max()))
    return last_id


def iter_unprocessed_rows(path: Path, last_id: int, count: int | None) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data directory: {path}")

    yielded = 0
    for data_path in sorted(path.glob("*.parquet")):
        for row in pd.read_parquet(data_path).to_dict(orient="records"):
            if int(row["id"]) <= last_id:
                continue
            yield row
            yielded += 1
            if count is not None and yielded >= count:
                return


def recreate_output(split_dir: Path) -> None:
    metrics_dir = split_dir / "metrics"
    state_path = split_dir / "metrics_state.json"
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)
    if state_path.exists():
        state_path.unlink()


def write_metrics_files(metrics_dir: Path, rows: list[dict[str, Any]]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    data_frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    data_frame.to_parquet(metrics_dir / "metrics.parquet", compression="zstd", index=False)
    data_frame.to_csv(metrics_dir / "metrics.csv", index=False)


def _write_state(split: str, dataset: str, dataset_name: str, model: str, model_id: str, last_id: int, count: int) -> None:
    state_path = _metrics_state_path(split, dataset, model)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    previous_count = 0
    if state_path.exists():
        previous_count = int(json.loads(state_path.read_text(encoding="utf-8")).get("processed_count", 0))

    payload = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": model,
        "model_id": model_id,
        "experiment_name": _experiment_name(dataset, model),
        "stage": "logprob_metrics",
        "metrics_version": "logprob_metrics_v1",
        "last_id": last_id,
        "processed_count": previous_count + count,
        "updated_at": _utc_now(),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _process_split(split: str, count: int | None, dataset: str, dataset_name: str, model_name: str, model_id: str, recreate: bool) -> None:
    split_dir = _split_dir(split, dataset, model_name)
    if recreate:
        recreate_output(split_dir)
    last_id = _read_last_processed_id(split, dataset, model_name)
    rows = list(iter_unprocessed_rows(_data_dir(split, dataset, model_name), last_id, count))
    if not rows:
        print(f"{split}: no processed rows left after id {last_id}")
        return

    records = [compute_logprob_metrics(row) for row in rows]
    write_metrics_files(_metrics_dir(split, dataset, model_name), records)
    _write_state(split, dataset, dataset_name, model_name, model_id, int(records[-1]["id"]), len(records))
    print(f"{split}: wrote {len(records)} metric rows, experiment={_experiment_name(dataset, model_name)}, last_id={records[-1]['id']}")


def _parse_args() -> argparse.Namespace:
    _load_experiment_config()
    parser = argparse.ArgumentParser(description="Compute scalar logprob metrics from processed token Parquet files.")
    parser.add_argument("--count", type=int, default=None, help="Number of processed rows to process per split.")
    parser.add_argument("--all", action="store_true", help="Process all remaining rows per split.")
    parser.add_argument("--recreate", action="store_true", help="Delete existing metrics output before processing.")
    parser.add_argument("--dataset", default="kaggle_train", choices=sorted(DATASETS), help="Dataset id from experiment_config.py.")
    parser.add_argument("--model", default="smollm2_360m", choices=sorted(MODEL_IDS), help="Model alias from experiment_config.py.")
    parser.add_argument("--model-id", default=None, help="Raw Hugging Face model id. Overrides --model.")
    return parser.parse_args()


def main() -> None:
    _load_experiment_config()
    args = _parse_args()
    count = None if args.all else args.count
    if count is None and not args.all:
        count = int(input("How many processed texts per split should be processed? "))
    dataset_name = resolve_dataset_name(args.dataset)
    resolve_dataset_splits(args.dataset)
    model_id = args.model_id if args.model_id is not None else resolve_model_id(args.model)
    model_name = args.model if args.model_id is None else args.model_id

    for split in resolve_dataset_splits(args.dataset):
        _process_split(split, count, args.dataset, dataset_name, model_name, model_id, args.recreate)


if __name__ == "__main__":
    main()
