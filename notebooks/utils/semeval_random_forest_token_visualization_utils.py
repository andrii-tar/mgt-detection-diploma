from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from notebooks.utils.feature_aggregation_utils import prepared_partition_path
from notebooks.utils.model_training_utils import (
    FittedFeatureVectorModel,
    load_model_artifact,
    load_model_metadata,
)
from notebooks.utils.semeval_boundary_utils import gold_token_boundary
from notebooks.utils.streaming_classification_evaluation_utils import iter_prepared_batches
from notebooks.utils.token_candidate_similarity_heatmaps_utils import split_dir


SCORER_LLM = "gemma_3_270m_it"
MODEL_FAMILY = "random_forest"
FEATURE_SET = "simple_similarity"
TRAINING_REGIME = "Mixed"
WINDOW_LENGTHS = (32, 64, 128)
STRIDE = 16


def _artifact_path(data_root: Path, window_length: int, run_id: str) -> Path:
    return (
        data_root
        / "experiment-results"
        / "classification"
        / "models"
        / MODEL_FAMILY
        / SCORER_LLM
        / FEATURE_SET
        / TRAINING_REGIME
        / str(window_length)
        / run_id
    )


def load_visualization_models(
    data_root: Path | str, *, run_id: str
) -> dict[int, FittedFeatureVectorModel]:
    models: dict[int, FittedFeatureVectorModel] = {}
    for window_length in WINDOW_LENGTHS:
        artifact_path = _artifact_path(Path(data_root), window_length, run_id)
        if not artifact_path.is_dir():
            raise FileNotFoundError(
                f"missing visualization artifact for window_length={window_length}: {artifact_path}"
            )
        expected = {
            "model_family": MODEL_FAMILY,
            "scorer_llm": SCORER_LLM,
            "feature_set": FEATURE_SET,
            "training_regime": TRAINING_REGIME,
            "training_window": str(window_length),
            "run_id": run_id,
            "stride": STRIDE,
        }
        metadata = load_model_metadata(artifact_path, expected=expected)
        models[window_length] = load_model_artifact(
            artifact_path,
            expected={**expected, "feature_columns": metadata["feature_columns"]},
            trust_artifact=True,
        )
    return models


def _mg_probability_index(model: FittedFeatureVectorModel) -> int:
    matches = np.flatnonzero(np.asarray(model.estimator.classes_) == 1)
    if len(matches) != 1:
        raise ValueError("random-forest estimator must expose exactly one MG class (1)")
    return int(matches[0])


def score_visualization_windows(
    models: Mapping[int, FittedFeatureVectorModel],
    windows_by_length: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    if set(models) != set(windows_by_length):
        raise ValueError("model and prepared-window keys must match")
    if set(models) != set(WINDOW_LENGTHS):
        raise ValueError("models and prepared windows must include all fixed window lengths")

    required = {"split", "text_id", "label", "start_token", "stop_token", "window_configuration"}
    frames: list[pd.DataFrame] = []
    for window_length, model in sorted(models.items()):
        windows = windows_by_length[window_length]
        missing = required.difference(windows.columns).union(model.feature_columns).difference(windows.columns)
        if missing:
            raise ValueError(
                f"window_length={window_length} is missing columns: {sorted(missing)}"
            )
        if not windows["window_configuration"].astype(str).eq(str(window_length)).all():
            raise ValueError(f"window_length={window_length} has incompatible prepared rows")

        starts = pd.to_numeric(windows["start_token"], errors="coerce")
        stops = pd.to_numeric(windows["stop_token"], errors="coerce")
        if starts.isna().any() or stops.isna().any() or (starts < 0).any() or (stops <= starts).any():
            raise ValueError(f"window_length={window_length} has invalid token spans")

        probabilities = np.asarray(
            model.estimator.predict_proba(windows.loc[:, list(model.feature_columns)])
        )
        mg_scores = probabilities[:, _mg_probability_index(model)]
        if len(mg_scores) != len(windows) or not np.isfinite(mg_scores).all() or not ((0 <= mg_scores) & (mg_scores <= 1)).all():
            raise ValueError(f"window_length={window_length} has invalid MG probabilities")

        scored = windows.loc[:, ["split", "text_id", "label", "start_token", "stop_token"]].copy()
        scored["model"] = MODEL_FAMILY
        scored["window_length"] = window_length
        scored["mg_score"] = mg_scores
        frames.append(scored)

    if not frames:
        raise ValueError("no visualization windows were provided")
    return pd.concat(frames, ignore_index=True)


def resolve_prepared_inputs(
    data_root: Path | str,
    dataset: str,
    models: Mapping[int, FittedFeatureVectorModel],
    *,
    split: str = "MG",
) -> dict[int, tuple[Path, Mapping[str, object]]]:
    prepared: dict[int, tuple[Path, Mapping[str, object]]] = {}
    for window_length in WINDOW_LENGTHS:
        model = models[window_length]
        partition = prepared_partition_path(
            data_root,
            scorer_llm=SCORER_LLM,
            dataset=dataset,
            split=split,
            feature_set=FEATURE_SET,
            window_configuration=window_length,
        )
        expected = {
            "scorer_llm": SCORER_LLM,
            "dataset": dataset,
            "split": split,
            "feature_set": FEATURE_SET,
            "window_configuration": str(window_length),
            "stride": STRIDE,
            "feature_columns": list(model.feature_columns),
        }
        metadata_path = partition / "metadata.json"
        data_path = partition / "data.parquet"
        if not metadata_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"incomplete prepared partition for window_length={window_length}: {partition}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"prepared metadata mismatch for {key} at window_length={window_length}"
                )
        prepared[window_length] = (partition, expected)
    return prepared


def _parquet_ids(path: Path, column: str) -> set[int]:
    values: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=16_384):
        values.update(int(value) for value in batch.column(column).to_pylist() if value is not None)
    return values


def select_eligible_text_ids(
    prepared_paths: Mapping[int, Path],
    *,
    stage1_ids: set[int],
    annotation_ids: set[int],
    count: int,
    seed: int,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be positive")

    shared_ids = set(stage1_ids).intersection(annotation_ids)
    for window_length in WINDOW_LENGTHS:
        try:
            partition = prepared_paths[window_length]
        except KeyError as error:
            raise KeyError(f"missing prepared partition for window_length={window_length}") from error
        shared_ids.intersection_update(_parquet_ids(partition / "data.parquet", "text_id"))

    if len(shared_ids) < count:
        raise ValueError(
            f"requested {count} eligible SemEval texts but found {len(shared_ids)} eligible texts"
        )
    sampled_ids = pd.Series(sorted(shared_ids), dtype="int64").sample(n=count, random_state=seed)
    return pd.DataFrame({"text_id": sampled_ids.to_list()})


def load_selected_windows(
    prepared: Mapping[int, tuple[Path, Mapping[str, object]]], selected_ids: set[int]
) -> dict[int, pd.DataFrame]:
    if not selected_ids:
        raise ValueError("selected IDs must not be empty")

    loaded: dict[int, pd.DataFrame] = {}
    for window_length, (path, expected) in prepared.items():
        batches = list(
            iter_prepared_batches(
                path,
                expected=expected,
                selected_ids=selected_ids,
                batch_size=16_384,
            )
        )
        if not batches:
            raise ValueError(f"no prepared rows for selected texts at window_length={window_length}")
        rows = pd.concat(batches, ignore_index=True)
        found_ids = set(rows["text_id"].astype("int64"))
        if found_ids != selected_ids:
            raise KeyError(
                f"prepared coverage mismatch at window_length={window_length}: "
                f"missing={sorted(selected_ids.difference(found_ids))}"
            )
        loaded[window_length] = rows
    return loaded


def _stage1_data_paths(
    processed_root: Path | str, dataset: str, model: str, split: str
) -> list[Path]:
    paths = sorted((split_dir(Path(processed_root), dataset, model, split) / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no stage-1 data Parquet files for dataset={dataset!r}, model={model!r}, split={split!r}"
        )
    return paths


def scan_stage1_ids(
    processed_root: Path | str, dataset: str, model: str, split: str
) -> set[int]:
    text_ids: set[int] = set()
    for path in _stage1_data_paths(processed_root, dataset, model, split):
        text_ids.update(_parquet_ids(path, "id"))
    return text_ids


def load_selected_token_labels(
    processed_root: Path | str,
    dataset: str,
    model: str,
    split: str,
    selected_ids: set[int],
) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for path in _stage1_data_paths(processed_root, dataset, model, split):
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["id", "tokens"], batch_size=16_384
        ):
            for text_id, tokens in zip(
                batch.column("id").to_pylist(), batch.column("tokens").to_pylist(), strict=True
            ):
                text_id = int(text_id)
                if text_id not in selected_ids:
                    continue
                if text_id in labels:
                    raise ValueError(f"duplicate stage-1 text id: {text_id}")
                labels[text_id] = [str(token["token"]) for token in tokens]
    if set(labels) != selected_ids:
        raise KeyError(
            f"stage-1 token coverage mismatch: missing={sorted(selected_ids.difference(labels))}"
        )
    return labels


ANNOTATION_COLUMNS = ["id", "text", "boundary_word_index"]


def scan_annotation_ids(annotation_path: Path | str) -> set[int]:
    text_ids: set[int] = set()
    for chunk in pd.read_csv(annotation_path, usecols=["id"], chunksize=16_384):
        text_ids.update(chunk["id"].dropna().astype("int64").tolist())
    return text_ids


def load_selected_annotations(
    annotation_path: Path | str, selected_ids: set[int]
) -> pd.DataFrame:
    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(annotation_path, usecols=ANNOTATION_COLUMNS, chunksize=16_384):
        selected_chunks.append(chunk.loc[chunk["id"].isin(selected_ids)])
    annotations = pd.concat(selected_chunks, ignore_index=True)
    if annotations["id"].duplicated().any():
        raise ValueError("selected SemEval annotation IDs must be unique")
    found_ids = set(annotations["id"].astype("int64"))
    if found_ids != selected_ids:
        raise KeyError(
            f"annotation coverage mismatch: missing={sorted(selected_ids.difference(found_ids))}"
        )
    return annotations


def build_gold_boundaries(
    annotations: pd.DataFrame, token_labels: Mapping[int, Sequence[str]]
) -> pd.DataFrame:
    annotation_ids = set(annotations["id"].astype("int64"))
    if annotations["id"].duplicated().any() or annotation_ids != set(token_labels):
        raise ValueError("selected annotations and token labels must cover the same unique text IDs")

    rows: list[dict[str, int]] = []
    for annotation in annotations.itertuples(index=False):
        text_id = int(annotation.id)
        labels = token_labels[text_id]
        try:
            boundary = gold_token_boundary(
                str(annotation.text), int(annotation.boundary_word_index), labels
            )
        except ValueError as error:
            raise ValueError(f"invalid SemEval boundary for text id {text_id}: {error}") from error
        rows.append(
            {
                "text_id": text_id,
                "gold_token_boundary": boundary,
                "token_count": len(labels),
            }
        )
    return pd.DataFrame(rows, columns=["text_id", "gold_token_boundary", "token_count"])


def write_visualization_export(
    output_directory: Path | str,
    *,
    selected_texts: pd.DataFrame,
    scored_windows: pd.DataFrame,
    gold_boundaries: pd.DataFrame,
    metadata: Mapping[str, object],
) -> Path:
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"visualization output already exists: {destination}")

    selected_ids = set(selected_texts["text_id"].astype("int64"))
    scored_ids = set(scored_windows["text_id"].astype("int64"))
    boundary_ids = set(gold_boundaries["text_id"].astype("int64"))
    if not selected_ids or selected_ids != scored_ids or selected_ids != boundary_ids:
        raise ValueError("selected texts, scored windows, and gold boundaries must cover the same IDs")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        selected_texts.to_csv(temporary / "selected_texts.csv", index=False)
        scored_windows.to_parquet(temporary / "scored_windows.parquet", index=False)
        gold_boundaries.to_csv(temporary / "gold_boundaries.csv", index=False)
        (temporary / "metadata.json").write_text(
            json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
