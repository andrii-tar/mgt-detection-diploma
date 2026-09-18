from __future__ import annotations

import json
import shutil
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
    required = ("artifact_id", "evaluation_regime", "dataset", "partition")
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
