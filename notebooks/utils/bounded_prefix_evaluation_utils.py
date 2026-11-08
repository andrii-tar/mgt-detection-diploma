from __future__ import annotations

from collections.abc import Iterator, Mapping, Set
from numbers import Integral
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from notebooks.utils.classification_experiment_utils import (
    read_existing_result_table,
    successful_result_keys,
)
from notebooks.utils.feature_aggregation_utils import (
    FeatureSet,
    _expected_label,
    _feature_columns,
    _processed_paths,
    build_feature_row,
)
from notebooks.utils.local_evidence_utils import _iter_paired_processed_rows, _window_starts


PREFIX_METRIC_COLUMNS = (
    "run_id", "artifact_id", "artifact_path", "model_family", "scorer_llm", "feature_set",
    "training_regime", "training_window", "evaluation_regime", "dataset", "partition",
    "min_tokens", "max_tokens", "eligible_parent_count", "scored_row_count", "accuracy",
    "balanced_accuracy", "macro_f1", "roc_auc", "false_positive_rate", "hg_support",
    "mg_support", "tn", "fp", "fn", "tp", "status", "error",
)
PREFIX_COVERAGE_COLUMNS = (
    "run_id", "artifact_id", "model_family", "scorer_llm", "feature_set", "training_regime",
    "training_window", "evaluation_regime", "dataset", "partition", "min_tokens", "max_tokens",
    "split", "class", "cohort_parent_count", "eligible_parent_count", "scored_row_count",
    "hg_support", "mg_support", "availability_status", "error",
)
PREFIX_RUN_SUMMARY_COLUMNS = (
    "run_id", "stage", "status", "artifact_id", "artifact_path", "model_family", "scorer_llm",
    "feature_set", "training_regime", "training_window", "evaluation_regime", "dataset", "partition",
    "min_tokens", "max_tokens", "error",
)
PREFIX_WINDOW_SIZE = 32
PREFIX_WINDOW_STRIDE = 16


def validate_prefix_bounds(min_tokens: int, max_tokens: int) -> tuple[int, int]:
    if (
        isinstance(min_tokens, bool) or not isinstance(min_tokens, Integral)
        or isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral)
    ):
        raise ValueError("token bounds must be integers")
    minimum, maximum = int(min_tokens), int(max_tokens)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("token bounds must be ordered positive integers")
    return minimum, maximum


def prefix_window_label(min_tokens: int, max_tokens: int) -> str:
    minimum, maximum = validate_prefix_bounds(min_tokens, max_tokens)
    return f"prefix_{minimum}_{maximum}_window_{PREFIX_WINDOW_SIZE}_stride_{PREFIX_WINDOW_STRIDE}"


def bounded_prefix_prediction_key(
    artifact_path: Path | str, *, models_root: Path | str, evaluation_regime: str,
    dataset: str, partition: str, min_tokens: int, max_tokens: int,
) -> dict[str, object]:
    minimum, maximum = validate_prefix_bounds(min_tokens, max_tokens)
    return {
        "artifact_id": "__".join(Path(artifact_path).relative_to(Path(models_root)).parts),
        "evaluation_regime": evaluation_regime,
        "evaluation_window": prefix_window_label(minimum, maximum),
        "dataset": dataset,
        "partition": partition,
        "min_tokens": minimum,
        "max_tokens": maximum,
    }


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


def _source_label(source: Mapping[str, object], split: str) -> int:
    label = source.get("label", _expected_label(split))
    numeric = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]
    if pd.isna(numeric) or numeric not in (0, 1) or int(numeric) != _expected_label(split):
        raise ValueError(f"label is inconsistent with split {split}")
    return int(numeric)


def _source_batch_total(paths: list[Path], batch_size: int) -> int | None:
    try:
        return sum(
            (pq.ParquetFile(path).metadata.num_rows + batch_size - 1) // batch_size
            for path in paths
        )
    except OSError:
        return None


def _iter_source_rows(
    processed_root: Path, *, scorer_llm: str, dataset: str, split: str, batch_size: int,
    progress_description: str | None = None,
) -> Iterator[tuple[dict[str, object], None]]:
    paths = _processed_paths(processed_root, dataset, scorer_llm, split)
    if progress_description is None:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                for source in batch.to_pylist():
                    yield source, None
        return
    with tqdm(
        total=_source_batch_total(paths, batch_size), desc=progress_description, unit="batch", leave=False,
    ) as progress:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                progress.update(1)
                for source in batch.to_pylist():
                    yield source, None


def iter_bounded_prefix_feature_batches(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, min_tokens: int, max_tokens: int, selected_ids: Set[object],
    batch_size: int, progress_description: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield model-ready leading-prefix feature rows in bounded batches."""
    minimum, maximum = validate_prefix_bounds(min_tokens, max_tokens)
    _validate_batch_size(batch_size)
    _, local_columns = _feature_columns(feature_set)
    root = Path(processed_root)
    source_rows: Iterator[tuple[dict[str, object], Mapping[str, object] | None]]
    if local_columns:
        paired, total = _iter_paired_processed_rows(root, dataset, scorer_llm, split, text_ids=None)
        source_rows = (
            tqdm(paired, total=total, desc=progress_description, unit="text", leave=False)
            if progress_description is not None else paired
        )
    else:
        source_rows = _iter_source_rows(
            root, scorer_llm=scorer_llm, dataset=dataset, split=split, batch_size=batch_size,
            progress_description=progress_description,
        )

    rows: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for source, similarity_source in source_rows:
        text_id = int(source["id"])
        if text_id in seen_ids:
            raise ValueError(f"duplicate source text identity: {split}/{text_id}")
        seen_ids.add(text_id)
        label = _source_label(source, split)
        if text_id not in selected_ids:
            continue
        tokens = [] if source.get("tokens") is None else list(source["tokens"])
        original_token_count = len(tokens)
        if original_token_count < minimum:
            continue
        evaluated_token_count = min(original_token_count, maximum)
        for start_token in _window_starts(
            evaluated_token_count, PREFIX_WINDOW_SIZE, PREFIX_WINDOW_STRIDE,
        ):
            row = build_feature_row(
                source, similarity_source, scorer_llm=scorer_llm, dataset=dataset, split=split,
                feature_set=feature_set, start_token=start_token,
                stop_token=start_token + PREFIX_WINDOW_SIZE,
                window_configuration=str(PREFIX_WINDOW_SIZE),
            )
            row["label"] = label
            row.update(
                original_token_count=original_token_count,
                evaluated_token_count=evaluated_token_count,
            )
            rows.append(row)
            if len(rows) == batch_size:
                yield _validated_feature_batch(rows, feature_set)
                rows = []
    if rows:
        yield _validated_feature_batch(rows, feature_set)


def _validated_feature_batch(rows: list[dict[str, object]], feature_set: FeatureSet) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    values = frame.loc[:, list(feature_set.source_columns)].apply(pd.to_numeric, errors="coerce")
    if (frame.loc[:, list(feature_set.source_columns)].notna() & values.isna()).any().any():
        raise ValueError("feature values must be numeric")
    frame.loc[:, list(feature_set.source_columns)] = values.replace(
        [float("inf"), float("-inf")], float("nan"),
    )
    return frame


def collect_source_parent_identities(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    max_per_split: int, batch_size: int, min_text_tokens: int | None = None,
    progress_description: str | None = None,
) -> pd.DataFrame:
    _validate_batch_size(batch_size)
    if isinstance(max_per_split, bool) or not isinstance(max_per_split, Integral) or max_per_split <= 0:
        raise ValueError("max_per_split must be a positive integer")
    if min_text_tokens is not None and min_text_tokens < 0:
        raise ValueError("min_text_tokens must be non-negative")
    rows: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for source, _ in _iter_source_rows(
        Path(processed_root), scorer_llm=scorer_llm, dataset=dataset, split=split, batch_size=batch_size,
        progress_description=progress_description,
    ):
        text_id = int(source["id"])
        if text_id in seen_ids:
            raise ValueError(f"duplicate source text identity: {split}/{text_id}")
        seen_ids.add(text_id)
        if min_text_tokens is not None and len(source["tokens"]) < min_text_tokens:
            continue
        candidate = {
            "dataset": dataset,
            "split": split,
            "text_id": text_id,
            "label": _source_label(source, split),
            "token_count": len(source["tokens"]),
        }
        if len(rows) < max_per_split:
            rows.append(candidate)
        else:
            largest_index = max(range(len(rows)), key=lambda index: rows[index]["text_id"])
            if text_id < rows[largest_index]["text_id"]:
                rows[largest_index] = candidate
    parents = pd.DataFrame(rows, columns=["dataset", "split", "text_id", "label", "token_count"])
    return parents.sort_values("text_id").head(int(max_per_split)).reset_index(drop=True)


def ensure_bounded_prefix_result_exports(data_root: Path | str, run_id: str) -> dict[str, Path]:
    destination = Path(data_root) / "experiment-results" / "classification" / "evaluations" / run_id
    paths = {
        "prefix_evaluation_metrics": destination / "prefix_evaluation_metrics.csv",
        "prefix_evaluation_coverage": destination / "prefix_evaluation_coverage.csv",
        "prefix_run_summary": destination / "prefix_run_summary.csv",
    }
    columns = {
        "prefix_evaluation_metrics": PREFIX_METRIC_COLUMNS,
        "prefix_evaluation_coverage": PREFIX_COVERAGE_COLUMNS,
        "prefix_run_summary": PREFIX_RUN_SUMMARY_COLUMNS,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        if not path.exists():
            pd.DataFrame(columns=columns[name]).to_csv(path, index=False)
    return paths


def bounded_prefix_successful_keys(path: Path | str, columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    return successful_result_keys(path, columns)
