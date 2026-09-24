from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Set
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def iter_prepared_batches(
    path: Path | str,
    *,
    expected: Mapping[str, object],
    selected_ids: Set[object],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield selected prepared rows in bounded-size Parquet batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    root = Path(path)
    metadata_path = root / "metadata.json"
    data_path = root / "data.parquet"
    if not metadata_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"incomplete prepared partition: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"prepared metadata mismatch for {key.replace('_', ' ')}")
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(isinstance(column, str) for column in feature_columns):
        raise ValueError("prepared metadata has invalid feature columns")
    columns = [
        "split", "text_id", "label", "start_token", "stop_token", "window_configuration", *feature_columns,
    ]
    for record_batch in pq.ParquetFile(data_path).iter_batches(batch_size=batch_size, columns=columns):
        frame = record_batch.to_pandas()
        selected = frame.loc[frame["text_id"].isin(selected_ids)].copy()
        if not selected.empty:
            yield selected


def prediction_artifact_path(results_root: Path | str, evaluation_key: Mapping[str, object]) -> Path:
    required = ("artifact_id", "evaluation_regime", "evaluation_window", "dataset", "partition")
    missing = [key for key in required if not evaluation_key.get(key)]
    if missing:
        raise ValueError(f"prediction artifact key is missing: {missing}")
    destination = Path(results_root) / "predictions"
    for key in required:
        destination /= str(evaluation_key[key])
    return destination


def prediction_artifact_is_complete(path: Path | str, *, expected: Mapping[str, object]) -> bool:
    root = Path(path)
    metadata_path = root / "metadata.json"
    data_path = root / "data.parquet"
    if not metadata_path.is_file() or not data_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return all(metadata.get(key) == value for key, value in expected.items()) and (
            pq.ParquetFile(data_path).metadata.num_rows == metadata.get("row_count")
        )
    except (OSError, ValueError, TypeError):
        return False


def write_prediction_artifact(
    path: Path | str, *, expected: Mapping[str, object], batches: Iterator[pd.DataFrame],
) -> int:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in batches:
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary / "data.parquet", table.schema)
            writer.write_table(table)
            rows += len(batch)
        if writer is None:
            raise ValueError("prediction artifact has no rows")
        writer.close()
        writer = None
        (temporary / "metadata.json").write_text(
            json.dumps({**expected, "row_count": rows}, indent=2), encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        return rows
    except Exception:
        if writer is not None:
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def aggregate_prediction_artifact(path: Path | str, threshold: float) -> dict[str, float | int]:
    """Calculate exact binary metrics without materializing prediction rows in memory."""
    data_path = Path(path) / "data.parquet"
    descriptor, database_path = tempfile.mkstemp(prefix=".prediction-metrics-", suffix=".sqlite", dir=data_path.parent)
    os.close(descriptor)
    connection: sqlite3.Connection | None = None
    try:
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE scores (label INTEGER NOT NULL, score REAL NOT NULL)")
            for batch in pq.ParquetFile(data_path).iter_batches(columns=["label", "mg_score"]):
                connection.executemany(
                    "INSERT INTO scores VALUES (?, ?)", batch.to_pandas().itertuples(index=False, name=None),
                )
            tn, fp, fn, tp = connection.execute(
                """SELECT
                    SUM(label = 0 AND score < ?), SUM(label = 0 AND score >= ?),
                    SUM(label = 1 AND score < ?), SUM(label = 1 AND score >= ?)
                FROM scores""", (threshold, threshold, threshold, threshold),
            ).fetchone()
            negatives, positives, total = connection.execute(
                "SELECT SUM(label = 0), SUM(label = 1), COUNT(*) FROM scores",
            ).fetchone()
            sum_positive_ranks = connection.execute(
                """WITH ranked AS (
                    SELECT label, RANK() OVER (ORDER BY score) AS first_rank,
                           COUNT(*) OVER (PARTITION BY score) AS tie_count
                    FROM scores
                ) SELECT SUM(CASE WHEN label = 1 THEN first_rank + (tie_count - 1) / 2.0 ELSE 0 END)
                    FROM ranked""",
            ).fetchone()[0]
            average_precision = connection.execute(
                """WITH grouped AS (
                    SELECT score, SUM(label = 1) AS positives, COUNT(*) AS total FROM scores GROUP BY score
                ), cumulative AS (
                    SELECT positives, total,
                           SUM(positives) OVER (ORDER BY score DESC) AS cumulative_positives,
                           SUM(total) OVER (ORDER BY score DESC) AS cumulative_total
                    FROM grouped
                ) SELECT SUM(positives * 1.0 / ? * cumulative_positives / cumulative_total) FROM cumulative""",
                (positives,),
            ).fetchone()[0]
        tn, fp, fn, tp = (int(value or 0) for value in (tn, fp, fn, tp))
        negatives, positives, total = int(negatives or 0), int(positives or 0), int(total)
        has_both_labels = bool(negatives and positives)
        hg_precision = _safe_divide(tn, tn + fn)
        hg_recall = _safe_divide(tn, negatives)
        mg_precision = _safe_divide(tp, tp + fp)
        mg_recall = _safe_divide(tp, positives)
        hg_f1 = _f1(hg_precision, hg_recall)
        mg_f1 = _f1(mg_precision, mg_recall)
        present_f1 = [
            f1 for is_present, f1 in (
                (bool(negatives or tn + fn), hg_f1),
                (bool(positives or fp + tp), mg_f1),
            ) if is_present
        ]
        predicted_mg = fp + tp
        return {
            "roc_auc": ((sum_positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)) if has_both_labels else float("nan"),
            "pr_auc": float(average_precision) if has_both_labels else float("nan"),
            "accuracy": (tn + tp) / total,
            "balanced_accuracy": (hg_recall + mg_recall) / 2 if has_both_labels else float("nan"),
            "macro_f1": sum(present_f1) / len(present_f1),
            "weighted_f1": (hg_f1 * negatives + mg_f1 * positives) / total,
            "hg_precision": hg_precision, "hg_recall": hg_recall, "hg_f1": hg_f1, "hg_support": negatives,
            "mg_precision": mg_precision, "mg_recall": mg_recall, "mg_f1": mg_f1, "mg_support": positives,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_positive_rate": fp / negatives if negatives else float("nan"),
            "predicted_mg_rate": predicted_mg / total,
        }
    finally:
        if connection is not None:
            connection.close()
        Path(database_path).unlink(missing_ok=True)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
