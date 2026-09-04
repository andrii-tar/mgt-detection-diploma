from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

STAGES = (
    "raw_input",
    "logprobs",
    "token_similarity",
    "logprob_metrics",
    "token_similarity_stats",
    "local_evidence",
)

OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "split",
    "stage",
    "processed_texts",
    "status",
    "path",
    "error",
]


def _experiment_name(dataset: str, model: str) -> str:
    slug = lambda value: re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()
    return f"{slug(dataset)}__{slug(model)}"


def _parquet_count(paths: list[Path], distinct: bool = False, split: str | None = None) -> int:
    if distinct:
        identifiers: set[Any] = set()
        for path in paths:
            frame = pd.read_parquet(path)
            if "split" in frame.columns:
                frame = frame[frame["split"] == split]
            identifiers.update(frame["id"].dropna().tolist())
        return len(identifiers)

    total = 0
    for path in paths:
        try:
            import pyarrow.parquet as pq

            total += int(pq.ParquetFile(path).metadata.num_rows)
        except ImportError:
            total += len(pd.read_parquet(path, columns=["id"]))
    return total


def _csv_count(path: Path) -> int:
    return len(pd.read_csv(path, usecols=["id"]))


def _artifact_record(
    dataset: str,
    model: str,
    split: str,
    stage: str,
    paths: list[Path],
    count_fn: Any,
) -> dict[str, Any]:
    display_path = ";".join(str(path) for path in paths)
    if not paths:
        return {
            "dataset": dataset,
            "model": model,
            "split": split,
            "stage": stage,
            "processed_texts": 0,
            "status": "missing",
            "path": display_path,
            "error": "",
        }
    try:
        count = int(count_fn(paths))
    except Exception as exc:  # Keep one malformed artifact from hiding other progress.
        return {
            "dataset": dataset,
            "model": model,
            "split": split,
            "stage": stage,
            "processed_texts": 0,
            "status": "error",
            "path": display_path,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "dataset": dataset,
        "model": model,
        "split": split,
        "stage": stage,
        "processed_texts": count,
        "status": "empty" if count == 0 else "complete",
        "path": display_path,
        "error": "",
    }


def scan_processing_counts(
    data_root: Path,
    datasets: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, Any],
) -> pd.DataFrame:
    processed_root = data_root / "processed"
    records: list[dict[str, Any]] = []
    for dataset, config in sorted(datasets.items()):
        for model in sorted(models):
            experiment_root = processed_root / _experiment_name(dataset, model)
            local_evidence_paths = sorted((experiment_root / "local_evidence").glob("*.parquet"))
            for split, raw_path in sorted(config["splits"].items()):
                split_root = experiment_root / split
                stage_paths = {
                    "raw_input": [raw_path] if raw_path.exists() else [],
                    "logprobs": sorted((split_root / "data").glob("*.parquet")),
                    "token_similarity": sorted((split_root / "token_similarities").glob("*.parquet")),
                    "logprob_metrics": [split_root / "metrics" / "metrics.parquet"]
                    if (split_root / "metrics" / "metrics.parquet").exists()
                    else [],
                    "token_similarity_stats": [
                        split_root / "token_similarity_stats" / "token_similarity_stats.parquet"
                    ]
                    if (split_root / "token_similarity_stats" / "token_similarity_stats.parquet").exists()
                    else [],
                    "local_evidence": local_evidence_paths,
                }
                expected_paths = {
                    "raw_input": raw_path,
                    "logprobs": split_root / "data",
                    "token_similarity": split_root / "token_similarities",
                    "logprob_metrics": split_root / "metrics" / "metrics.parquet",
                    "token_similarity_stats": split_root
                    / "token_similarity_stats"
                    / "token_similarity_stats.parquet",
                    "local_evidence": experiment_root / "local_evidence",
                }
                for stage in STAGES:
                    if stage == "raw_input":
                        count_fn = lambda paths: _csv_count(paths[0])
                    elif stage == "local_evidence":
                        count_fn = lambda paths, current_split=split: _parquet_count(
                            paths, distinct=True, split=current_split
                        )
                    else:
                        count_fn = _parquet_count
                    record = _artifact_record(
                        dataset, model, split, stage, stage_paths[stage], count_fn
                    )
                    if not stage_paths[stage]:
                        record["path"] = str(expected_paths[stage])
                    records.append(record)

    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result["stage"] = pd.Categorical(result["stage"], categories=STAGES, ordered=True)
    return result.sort_values(["dataset", "model", "split", "stage"]).reset_index(drop=True)
