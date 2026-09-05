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

POSTPROCESSING_COLUMNS = [
    "dataset",
    "model",
    "split",
    "stage1_rows",
    "stage1_unique",
    "stage2_rows",
    "stage2_unique",
    "missing_stage2",
    "extra_stage2",
    "duplicate_stage1",
    "duplicate_stage2",
    "metrics_rows",
    "similarity_stats_rows",
    "local_evidence_rows",
    "status",
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


def _postprocess_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [int(value) for value in pd.read_parquet(path, columns=["id"])["id"].tolist()]


def _postprocess_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_parquet(path, columns=["id"]))


def _postprocess_split_dirs(processed_root: Path) -> list[tuple[str, str, Path]]:
    split_dirs: list[tuple[str, str, Path]] = []
    ignored = {"local_evidence", "local_evidence_models", "local_evidence_windows"}
    if not processed_root.exists():
        return split_dirs
    for experiment_root in sorted(path for path in processed_root.iterdir() if path.is_dir() and "__" in path.name):
        dataset, model = experiment_root.name.split("__", 1)
        for split_root in sorted(path for path in experiment_root.iterdir() if path.is_dir() and path.name not in ignored):
            if split_root.name in {"HG", "MG"} or any(
                (split_root / name).exists()
                for name in ("data", "token_similarities", "postprocessed", "state.json", "similarity_state.json")
            ):
                split_dirs.append((dataset, model, split_root))
    return split_dirs


def scan_postprocessing_readiness(processed_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, model, split_root in _postprocess_split_dirs(processed_root):
        stage1_path = split_root / "postprocessed" / "stage1.parquet"
        stage2_path = split_root / "postprocessed" / "stage2.parquet"
        metrics_path = split_root / "metrics" / "metrics.parquet"
        stats_path = split_root / "token_similarity_stats" / "token_similarity_stats.parquet"
        local_path = split_root.parent / "local_evidence" / "windows.parquet"
        error = ""
        try:
            stage1_values = _postprocess_ids(stage1_path)
            stage2_values = _postprocess_ids(stage2_path)
            stage1_ids = set(stage1_values)
            stage2_ids = set(stage2_values)
            local_rows = 0
            if local_path.exists():
                local_frame = pd.read_parquet(local_path, columns=["split", "text_id"])
                local_rows = len(local_frame[local_frame["split"] == split_root.name]["text_id"].unique())
            if not stage1_values and not stage2_values:
                status = "EMPTY"
            elif not stage1_values:
                status = "MISSING_STAGE1"
            elif not stage2_values:
                status = "MISSING_STAGE2"
            elif len(stage1_values) != len(stage1_ids) or len(stage2_values) != len(stage2_ids):
                status = "ID_MISMATCH"
            elif stage1_ids != stage2_ids:
                status = "ID_MISMATCH"
            elif _postprocess_rows(metrics_path) != len(stage1_ids):
                status = "MISSING_METRICS"
            elif _postprocess_rows(stats_path) != len(stage2_ids):
                status = "MISSING_SIMILARITY_STATS"
            elif local_rows != len(stage1_ids):
                status = "MISSING_LOCAL_EVIDENCE"
            else:
                status = "READY"
        except Exception as exc:  # Keep malformed combinations visible in the report.
            stage1_values = []
            stage2_values = []
            local_rows = 0
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"

        records.append(
            {
                "dataset": dataset,
                "model": model,
                "split": split_root.name,
                "stage1_rows": len(stage1_values),
                "stage1_unique": len(set(stage1_values)),
                "stage2_rows": len(stage2_values),
                "stage2_unique": len(set(stage2_values)),
                "missing_stage2": len(set(stage1_values) - set(stage2_values)),
                "extra_stage2": len(set(stage2_values) - set(stage1_values)),
                "duplicate_stage1": len(stage1_values) - len(set(stage1_values)),
                "duplicate_stage2": len(stage2_values) - len(set(stage2_values)),
                "metrics_rows": _postprocess_rows(metrics_path) if not error else 0,
                "similarity_stats_rows": _postprocess_rows(stats_path) if not error else 0,
                "local_evidence_rows": local_rows,
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(records, columns=POSTPROCESSING_COLUMNS).sort_values(
        ["dataset", "model", "split"]
    ).reset_index(drop=True)
