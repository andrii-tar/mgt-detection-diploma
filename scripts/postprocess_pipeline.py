from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

from scripts import export_token_similarity_stats as similarity_stats
from scripts import logprob_metrics
from scripts.data_root import resolve_data_root


@dataclass(frozen=True)
class IdComparison:
    stage1_ids: frozenset[int]
    stage2_ids: frozenset[int]
    stage1_rows: int
    stage2_rows: int
    duplicate_stage1: int
    duplicate_stage2: int

    @property
    def missing_stage2_ids(self) -> set[int]:
        return set(self.stage1_ids - self.stage2_ids)

    @property
    def extra_stage2_ids(self) -> set[int]:
        return set(self.stage2_ids - self.stage1_ids)

    @property
    def is_exact(self) -> bool:
        return not self.missing_stage2_ids and not self.extra_stage2_ids and not self.duplicate_stage1 and not self.duplicate_stage2


def compare_ids(stage1_ids: Iterable[int], stage2_ids: Iterable[int]) -> IdComparison:
    stage1_values = [int(value) for value in stage1_ids]
    stage2_values = [int(value) for value in stage2_ids]
    return IdComparison(
        stage1_ids=frozenset(stage1_values),
        stage2_ids=frozenset(stage2_values),
        stage1_rows=len(stage1_values),
        stage2_rows=len(stage2_values),
        duplicate_stage1=len(stage1_values) - len(set(stage1_values)),
        duplicate_stage2=len(stage2_values) - len(set(stage2_values)),
    )


def _read_ids(paths: list[Path]) -> list[int]:
    values: list[int] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["id"])
        values.extend(int(value) for value in frame["id"].tolist())
    return values


def _read_rows(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)


def _atomic_write_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    frame.to_parquet(temporary, compression="zstd", index=False)
    temporary.replace(output)


def _atomic_write_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def consolidate_parquet_files(paths: list[Path], output: Path) -> Path:
    frame = _read_rows(sorted(paths))
    if frame.empty:
        raise ValueError(f"Cannot consolidate empty Parquet inputs into {output}")
    if frame["id"].duplicated().any():
        raise ValueError(f"Cannot consolidate duplicate IDs into {output}")
    _atomic_write_parquet(frame, output)
    return output


def _experiment_parts(name: str) -> tuple[str, str]:
    if "__" not in name:
        raise ValueError(f"Expected '<dataset>__<model>' directory name, got {name}")
    return tuple(name.split("__", 1))  # type: ignore[return-value]


def _split_dirs(experiment_dir: Path) -> list[Path]:
    ignored = {"local_evidence", "local_evidence_models", "local_evidence_windows"}
    result = []
    for path in experiment_dir.iterdir():
        if not path.is_dir() or path.name in ignored:
            continue
        if any((path / name).exists() for name in ("data", "token_similarities", "state.json", "similarity_state.json")):
            result.append(path)
    return sorted(result)


def _write_metrics(canonical: Path, split_dir: Path) -> None:
    rows = pd.read_parquet(canonical).to_dict(orient="records")
    records = [logprob_metrics.compute_logprob_metrics(row) for row in rows]
    frame = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    metrics_dir = split_dir / "metrics"
    _atomic_write_parquet(frame, metrics_dir / "metrics.parquet")
    _atomic_write_csv(frame, metrics_dir / "metrics.csv")
    source = rows[0]
    state = {
        "dataset": source.get("dataset"),
        "dataset_name": source.get("dataset_name"),
        "model": source.get("model"),
        "model_id": source.get("model_id"),
        "experiment_name": source.get("experiment_name"),
        "experiment_params": source.get("experiment_params"),
        "split": source.get("split"),
        "stage": "logprob_metrics",
        "metrics_version": "logprob_metrics_v3",
        "last_id": int(frame["id"].iloc[-1]),
        "processed_count": len(frame),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path = split_dir / "metrics_state.json"
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(state_path)


def _write_similarity_stats(canonical: Path, split_dir: Path) -> None:
    rows = pd.read_parquet(canonical)
    records = [similarity_stats.token_similarity_stats(row) for _, row in rows.iterrows()]
    frame = pd.DataFrame(records, columns=similarity_stats.STATS_COLUMNS).sort_values("id").reset_index(drop=True)
    _atomic_write_parquet(frame, split_dir / "token_similarity_stats" / "token_similarity_stats.parquet")


def _status(comparison: IdComparison, stage1_paths: list[Path], stage2_paths: list[Path]) -> str:
    if not stage1_paths and not stage2_paths:
        return "NO_DATA"
    if not stage1_paths:
        return "NO_STAGE1"
    if not stage2_paths:
        return "NO_STAGE2"
    if comparison.duplicate_stage1:
        return "DUPLICATE_STAGE1"
    if comparison.duplicate_stage2:
        return "DUPLICATE_STAGE2"
    if comparison.missing_stage2_ids or comparison.extra_stage2_ids:
        return "ID_MISMATCH"
    return "COMPLETE"


def _process_split(
    split_dir: Path,
    report_phase: Callable[[str], None] | None = None,
) -> tuple[str, IdComparison]:
    def report(name: str) -> None:
        if report_phase is not None:
            report_phase(name)

    stage1_paths = sorted((split_dir / "data").glob("*.parquet"))
    stage2_paths = sorted((split_dir / "token_similarities").glob("*.parquet"))
    comparison = compare_ids(_read_ids(stage1_paths), _read_ids(stage2_paths))
    report("read_ids")
    status = _status(comparison, stage1_paths, stage2_paths)

    if stage1_paths and not comparison.duplicate_stage1:
        canonical = split_dir / "postprocessed" / "stage1.parquet"
        consolidate_parquet_files(stage1_paths, canonical)
        report("stage1")
        _write_metrics(canonical, split_dir)
        report("metrics")
    if stage2_paths and not comparison.duplicate_stage2:
        canonical = split_dir / "postprocessed" / "stage2.parquet"
        consolidate_parquet_files(stage2_paths, canonical)
        report("stage2")
        _write_similarity_stats(canonical, split_dir)
        report("similarity_stats")

    print(
        f"{split_dir.parent.name}/{split_dir.name}: "
        f"stage1_rows={comparison.stage1_rows} stage1_unique={len(comparison.stage1_ids)} "
        f"stage2_rows={comparison.stage2_rows} stage2_unique={len(comparison.stage2_ids)} "
        f"missing_stage2={len(comparison.missing_stage2_ids)} "
        f"extra_stage2={len(comparison.extra_stage2_ids)} "
        f"duplicate_stage1={comparison.duplicate_stage1} duplicate_stage2={comparison.duplicate_stage2} "
        f"status={status}"
    )
    return status, comparison


def _rebuild_local_evidence(
    experiment_dir: Path,
    valid_splits: set[str],
    processed_root: Path,
    report_phase: Callable[[str], None] | None = None,
) -> bool:
    if not {"HG", "MG"}.issubset(valid_splits):
        return False
    from notebooks.local_evidence_utils import load_or_build_local_evidence_windows

    dataset, model = _experiment_parts(experiment_dir.name)
    load_or_build_local_evidence_windows(
        processed_root,
        dataset,
        model,
        splits=("HG", "MG"),
        recompute=True,
    )
    if report_phase is not None:
        report_phase("local_evidence")
    return True


def run_postprocessing(
    processed_root: Path,
    dataset: str | None = None,
    model: str | None = None,
    split: str | None = None,
) -> int:
    total = complete = invalid = 0
    local_evidence = 0
    for experiment_dir in sorted(path for path in processed_root.iterdir() if path.is_dir() and "__" in path.name):
        dataset_name, model_name = _experiment_parts(experiment_dir.name)
        if dataset is not None and dataset_name != dataset:
            continue
        if model is not None and model_name != model:
            continue
        valid_splits: set[str] = set()
        split_dirs = _split_dirs(experiment_dir)
        if split is not None:
            split_dirs = [split_dir for split_dir in split_dirs if split_dir.name == split]
        with tqdm(desc=f"{dataset_name}/{model_name}", unit="phase") as progress:
            def report_phase(name: str) -> None:
                progress.update(1)
                progress.set_postfix_str(name)

            for split_dir in split_dirs:
                status, _ = _process_split(split_dir, report_phase=report_phase)
                total += 1
                if status == "COMPLETE":
                    complete += 1
                    valid_splits.add(split_dir.name)
                else:
                    invalid += 1
            if split is None and valid_splits == {"HG", "MG"}:
                if _rebuild_local_evidence(
                    experiment_dir,
                    valid_splits,
                    processed_root,
                    report_phase=report_phase,
                ):
                    local_evidence += 1

    print(f"SUMMARY: splits={total} complete={complete} invalid={invalid} local_evidence={local_evidence}")
    return 1 if invalid else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate and rebuild postprocessing outputs from Stage 1 and Stage 2.")
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--split")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    processed_root = args.processed_root or (resolve_data_root() / "processed")
    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed root: {processed_root}")
    return run_postprocessing(processed_root, args.dataset, args.model, args.split)


if __name__ == "__main__":
    raise SystemExit(main())
