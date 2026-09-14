from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Literal, Mapping

import pandas as pd

from notebooks.utils.local_evidence_utils import (
    LOCAL_METRICS,
    WINDOW_STATISTICS,
    _iter_paired_processed_rows,
    _pool_feature,
    _window_starts,
)
from notebooks.utils.token_candidate_similarity_heatmaps_utils import validated_text_summary
from scripts.logprob_metrics import compute_logprob_metrics


WindowConfiguration = int | Literal["whole_text"]
PREPARATION_SCHEMA_VERSION = 1
CANONICAL_CACHE_SCHEMA_VERSION = 1
CANONICAL_FEATURE_SET = "all_features"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    compute: Callable[[Mapping[str, object]], float]


METRIC_REGISTRY = {
    name: MetricDefinition(name, lambda metrics, key=name: float(metrics[key]))
    for name in (
        "d_score_top20",
        "perplexity",
        "prompt_mu_score",
        "prompt_sigma_score",
        "d_score_top20_coverage",
    )
}

CANONICAL_FEATURE_COLUMNS = (
    *METRIC_REGISTRY,
    *(f"{metric}__{statistic}" for metric in LOCAL_METRICS for statistic in WINDOW_STATISTICS),
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.source_columns:
            raise ValueError("feature set must have a name and at least one feature")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("feature set feature columns must be unique")


def _window_configuration(value: WindowConfiguration) -> str:
    if value == "whole_text":
        return value
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("window_configuration must be a positive integer or 'whole_text'")
    return str(int(value))


def _metadata_window_configuration(value: object) -> str:
    if value == "whole_text":
        return value
    if (isinstance(value, str) and value.isdigit() and int(value) > 0 and
            str(int(value)) == value):
        return value
    raise ValueError("prepared metadata window configuration is invalid")


def _validate_stride(stride: int) -> None:
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride <= 0:
        raise ValueError("stride must be a positive integer")


def _expected_label(split: str) -> int:
    if split == "HG":
        return 0
    if split == "MG":
        return 1
    raise ValueError("split must be HG or MG")


def _validate_frame_labels(frame: pd.DataFrame) -> None:
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if frame.groupby(["split", "text_id"], dropna=False)["label"].nunique().gt(1).any():
        raise ValueError("inconsistent labels per parent")
    for split, group in frame.groupby("split", dropna=False):
        expected = _expected_label(str(split))
        if not labels.loc[group.index].eq(expected).all():
            raise ValueError(f"label is inconsistent with split {split}")


REQUIRED_METADATA = (
    "schema_version", "scorer_llm", "dataset", "split", "feature_set",
    "feature_columns", "window_configuration", "stride", "aggregation_rules", "row_count",
)


def _processed_paths(processed_root: Path, dataset: str, scorer_llm: str, split: str) -> list[Path]:
    paths = sorted((processed_root / f"{dataset}__{scorer_llm}" / split / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing Stage-1 source data for {dataset}/{scorer_llm}/{split}")
    return paths


def _similarity_paths(processed_root: Path, dataset: str, scorer_llm: str, split: str) -> list[Path]:
    return sorted((processed_root / f"{dataset}__{scorer_llm}" / split / "token_similarities").glob("*.parquet"))


def source_manifest(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
) -> list[dict[str, int | str]]:
    root = Path(processed_root) / f"{dataset}__{scorer_llm}" / split
    stage1_paths = _processed_paths(Path(processed_root), dataset, scorer_llm, split)
    entries = [("stage1", path) for path in stage1_paths]
    entries.extend(("stage2", path) for path in _similarity_paths(Path(processed_root), dataset, scorer_llm, split))
    return [
        {"stage": stage, "path": str(path.relative_to(root)), "size": path.stat().st_size,
         "mtime_ns": path.stat().st_mtime_ns}
        for stage, path in entries
    ]


def canonical_partition_path(
    data_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    window_configuration: WindowConfiguration,
) -> Path:
    return (Path(data_root) / "experiment-results" / "classification" / "prepared" /
            CANONICAL_FEATURE_SET / scorer_llm / dataset / split /
            _window_configuration(window_configuration))


def _feature_columns(feature_set: FeatureSet) -> tuple[tuple[str, ...], dict[str, tuple[str, str]]]:
    scalar_columns = tuple(column for column in feature_set.source_columns if column in METRIC_REGISTRY)
    local_columns = {}
    for column in feature_set.source_columns:
        metric, separator, statistic = column.rpartition("__")
        if separator and metric in LOCAL_METRICS and statistic in WINDOW_STATISTICS:
            local_columns[column] = (metric, statistic)
    unknown = set(feature_set.source_columns).difference((*scalar_columns, *local_columns))
    if unknown:
        raise ValueError(f"feature metrics are not registered: {sorted(unknown)}")
    return scalar_columns, local_columns


def prepare_feature_partition(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
) -> pd.DataFrame:
    configuration = _window_configuration(window_configuration)
    _validate_stride(stride)
    scalar_columns, local_columns = _feature_columns(feature_set)
    rows: list[dict[str, object]] = []
    seen_text_ids: set[int] = set()
    if local_columns:
        paired_rows, _ = _iter_paired_processed_rows(
            Path(processed_root), dataset, scorer_llm, split, text_ids=None,
        )
        source_rows = paired_rows
    else:
        source_rows = (
            (source, None)
            for path in _processed_paths(Path(processed_root), dataset, scorer_llm, split)
            for source in pd.read_parquet(path).to_dict(orient="records")
        )
    for source, similarity_source in source_rows:
        text_id = int(source["id"])
        if text_id in seen_text_ids:
            raise ValueError(f"duplicate source text identity: {split}/{text_id}")
        seen_text_ids.add(text_id)
        raw_tokens = source.get("tokens")
        tokens = [] if raw_tokens is None else list(raw_tokens)
        summary = (
            validated_text_summary(source, similarity_source, top_k=20)[0]
            if tokens and similarity_source is not None else None
        )
        if configuration == "whole_text":
            starts = [0]
            length = len(tokens)
        else:
            length = int(configuration)
            starts = _window_starts(len(tokens), length, stride) if tokens else []
        for start in starts:
            stop = len(tokens) if configuration == "whole_text" else start + length
            metrics = ({column: float("nan") for column in feature_set.source_columns}
                       if not tokens[start:stop]
                       else compute_logprob_metrics({**source, "tokens": tokens[start:stop]}))
            local_values = (
                {column: _pool_feature(summary[metric][start:stop])[statistic]
                 for column, (metric, statistic) in local_columns.items()}
                if summary is not None else {column: float("nan") for column in local_columns}
            )
            label = source.get("label", _expected_label(split))
            rows.append({
                "scorer_llm": scorer_llm,
                "dataset": dataset,
                "split": split,
                "text_id": text_id,
                "label": label,
                "start_token": start,
                "stop_token": stop,
                "window_configuration": configuration,
                **{column: METRIC_REGISTRY[column].compute(metrics) for column in scalar_columns},
                **local_values,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("prepared feature partition is empty")
    _validate_frame_labels(frame)
    frame.loc[:, list(feature_set.source_columns)] = (
        frame.loc[:, list(feature_set.source_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
    )
    return frame


def prepare_canonical_partition(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    window_configuration: WindowConfiguration, stride: int,
) -> tuple[pd.DataFrame, list[str]]:
    configuration = _window_configuration(window_configuration)
    _validate_stride(stride)
    processed_root = Path(processed_root)
    has_stage2 = bool(_similarity_paths(processed_root, dataset, scorer_llm, split))
    feature_columns = list(CANONICAL_FEATURE_COLUMNS if has_stage2 else METRIC_REGISTRY)
    rows: list[dict[str, object]] = []
    seen_text_ids: set[int] = set()
    if has_stage2:
        source_rows, _ = _iter_paired_processed_rows(processed_root, dataset, scorer_llm, split, text_ids=None)
    else:
        source_rows = (
            (source, None)
            for path in _processed_paths(processed_root, dataset, scorer_llm, split)
            for source in pd.read_parquet(path).to_dict(orient="records")
        )
    for source, similarity_source in source_rows:
        text_id = int(source["id"])
        if text_id in seen_text_ids:
            raise ValueError(f"duplicate source text identity: {split}/{text_id}")
        seen_text_ids.add(text_id)
        tokens = [] if source.get("tokens") is None else list(source["tokens"])
        summary = validated_text_summary(source, similarity_source, top_k=20)[0] if similarity_source is not None and tokens else None
        if configuration == "whole_text":
            starts, length = [0], len(tokens)
        else:
            length = int(configuration)
            starts = _window_starts(len(tokens), length, stride) if tokens else []
        for start in starts:
            stop = len(tokens) if configuration == "whole_text" else start + length
            metrics = ({column: float("nan") for column in METRIC_REGISTRY}
                       if not tokens[start:stop]
                       else compute_logprob_metrics({**source, "tokens": tokens[start:stop]}))
            local_values = {
                f"{metric}__{statistic}": _pool_feature(
                    summary[metric][start:stop] if summary is not None else [float("nan")]
                )[statistic]
                for metric in LOCAL_METRICS for statistic in WINDOW_STATISTICS
            }
            rows.append({
                "scorer_llm": scorer_llm, "dataset": dataset, "split": split, "text_id": text_id,
                "label": source.get("label", _expected_label(split)), "start_token": start,
                "stop_token": stop, "window_configuration": configuration, **metrics, **local_values,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("prepared feature partition is empty")
    _validate_frame_labels(frame)
    frame.loc[:, feature_columns] = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan"),
    )
    return frame, feature_columns


def prepared_partition_path(
    data_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet | str, window_configuration: WindowConfiguration,
) -> Path:
    configuration = _window_configuration(window_configuration)
    name = feature_set.name if isinstance(feature_set, FeatureSet) else feature_set
    return (Path(data_root) / "experiment-results" / "classification" / "prepared" /
            name / scorer_llm / dataset / split / configuration)


def _publish_partition(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.previous")
    if destination.exists():
        destination.rename(previous)
    try:
        temporary.rename(destination)
    except Exception:
        if previous.exists():
            previous.rename(destination)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _canonical_metadata(
    *, scorer_llm: str, dataset: str, split: str, feature_columns: list[str],
    window_configuration: WindowConfiguration, stride: int, row_count: int,
    source_manifest_value: list[dict[str, int | str]],
) -> dict[str, object]:
    configuration = _window_configuration(window_configuration)
    return {
        "schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
        "scorer_llm": scorer_llm,
        "dataset": dataset,
        "split": split,
        "feature_set": CANONICAL_FEATURE_SET,
        "feature_columns": feature_columns,
        "window_configuration": configuration,
        "stride": None if configuration == "whole_text" else stride,
        "aggregation_rules": "whole_text" if configuration == "whole_text" else "deterministic sliding token windows",
        "row_count": row_count,
        "source_manifest": source_manifest_value,
    }


def load_canonical_partition(path: Path | str, *, expected: Mapping[str, object]) -> pd.DataFrame:
    root = Path(path)
    metadata_path = root / "metadata.json"
    data_path = root / "data.parquet"
    if not metadata_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"incomplete canonical partition: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = (*REQUIRED_METADATA, "source_manifest")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"missing canonical metadata: {[key.replace('_', ' ') for key in missing]}")
    if metadata["schema_version"] != CANONICAL_CACHE_SCHEMA_VERSION:
        raise ValueError("canonical metadata schema version is unsupported")
    if metadata["feature_set"] != CANONICAL_FEATURE_SET:
        raise ValueError("canonical metadata feature set is invalid")
    feature_columns = _validate_feature_columns(metadata["feature_columns"])
    configuration = _metadata_window_configuration(metadata["window_configuration"])
    if configuration == "whole_text":
        if metadata["stride"] is not None or metadata["aggregation_rules"] != "whole_text":
            raise ValueError("canonical metadata has invalid whole-text semantics")
    elif metadata["stride"] is None or metadata["aggregation_rules"] != "deterministic sliding token windows":
        raise ValueError("canonical metadata has invalid sliding-window semantics")
    else:
        _validate_stride(metadata["stride"])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"canonical metadata mismatch for {key.replace('_', ' ')}")
    frame = pd.read_parquet(data_path)
    missing_columns = set(_identity_columns()).difference(frame.columns) | {"label"}.difference(frame.columns) | set(feature_columns).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing canonical columns: {sorted(missing_columns)}")
    if len(frame) != metadata["row_count"]:
        raise ValueError("canonical metadata row count does not match data")
    _validate_partition_rows(frame, split=metadata["split"], window_configuration=configuration)
    _validate_frame_labels(frame)
    if frame.duplicated(_identity_columns()).any():
        raise ValueError("duplicate canonical partition identity")
    return frame


def ensure_canonical_partition(
    data_root: Path | str, processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    window_configuration: WindowConfiguration, stride: int,
) -> dict[str, object]:
    path = canonical_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                    window_configuration=window_configuration)
    try:
        manifest = source_manifest(processed_root, scorer_llm=scorer_llm, dataset=dataset, split=split)
        expected = {"scorer_llm": scorer_llm, "dataset": dataset, "split": split,
                    "window_configuration": _window_configuration(window_configuration),
                    "stride": None if window_configuration == "whole_text" else stride,
                    "source_manifest": manifest}
        if path.exists():
            try:
                load_canonical_partition(path, expected=expected)
            except ValueError as error:
                if "source manifest" not in str(error):
                    raise
                frame, feature_columns = prepare_canonical_partition(
                    processed_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                    window_configuration=window_configuration, stride=stride,
                )
                _save_canonical_partition(path, frame, feature_columns=feature_columns, expected=expected)
        else:
            frame, feature_columns = prepare_canonical_partition(
                processed_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                window_configuration=window_configuration, stride=stride,
            )
            _save_canonical_partition(path, frame, feature_columns=feature_columns, expected=expected)
        return {"status": "available", "path": path, "expected": expected}
    except Exception as error:
        return {"status": "unavailable", "path": path, "error": str(error)}


def _save_canonical_partition(
    destination: Path, frame: pd.DataFrame, *, feature_columns: list[str], expected: Mapping[str, object],
) -> None:
    configuration = str(expected["window_configuration"])
    metadata = _canonical_metadata(
        scorer_llm=str(expected["scorer_llm"]), dataset=str(expected["dataset"]), split=str(expected["split"]),
        feature_columns=feature_columns, window_configuration="whole_text" if configuration == "whole_text" else int(configuration),
        stride=1 if expected["stride"] is None else int(expected["stride"]), row_count=len(frame),
        source_manifest_value=expected["source_manifest"],
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True)
        frame.to_parquet(temporary / "data.parquet", index=False)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        load_canonical_partition(temporary, expected=expected)
        _publish_partition(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _identity_columns() -> list[str]:
    return ["split", "text_id", "start_token", "stop_token", "window_configuration"]


def _validate_feature_columns(feature_columns: object) -> list[str]:
    if (not isinstance(feature_columns, list) or not feature_columns or
            not all(isinstance(column, str) and column for column in feature_columns) or
            len(set(feature_columns)) != len(feature_columns)):
        raise ValueError("feature columns must be a non-empty ordered list of unique names")
    reserved = {"split", "text_id", "label", "start_token", "stop_token", "window_configuration"}
    if reserved.intersection(feature_columns):
        raise ValueError("feature columns must not contain partition identity columns")
    return feature_columns


def _validate_partition_rows(frame: pd.DataFrame, *, split: str, window_configuration: str) -> None:
    missing = set(_identity_columns()).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prepared partition identity columns: {sorted(missing)}")
    if not frame["split"].eq(split).all():
        raise ValueError(f"prepared rows do not match requested split {split}")
    if not frame["window_configuration"].eq(window_configuration).all():
        raise ValueError("prepared rows do not match requested window configuration")


def save_prepared_partition(
    data_root: Path | str, frame: pd.DataFrame, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
    canonical_source_manifest: list[dict[str, int | str]] | None = None, replace: bool = False,
) -> Path:
    configuration = _window_configuration(window_configuration)
    _validate_stride(stride)
    missing = {"split", "text_id", "label"}.difference(frame.columns) | set(feature_set.source_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prepared columns: {sorted(missing)}")
    _validate_partition_rows(frame, split=split, window_configuration=configuration)
    _validate_frame_labels(frame)
    _validate_feature_columns(list(feature_set.source_columns))
    if frame.duplicated(_identity_columns()).any():
        raise ValueError("duplicate prepared partition identity")
    destination = prepared_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                          feature_set=feature_set, window_configuration=window_configuration)
    if destination.exists() and not replace:
        raise FileExistsError(f"prepared partition already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    metadata = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "scorer_llm": scorer_llm,
        "dataset": dataset,
        "split": split,
        "feature_set": feature_set.name,
        "feature_columns": list(feature_set.source_columns),
        "window_configuration": configuration,
        "stride": None if configuration == "whole_text" else stride,
        "aggregation_rules": "whole_text" if configuration == "whole_text" else "deterministic sliding token windows",
        "row_count": len(frame),
    }
    if canonical_source_manifest is not None:
        metadata["canonical_source_manifest"] = canonical_source_manifest
    try:
        temporary.mkdir(parents=True)
        frame.to_parquet(temporary / "data.parquet", index=False)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        loaded = load_prepared_partition(temporary, expected=metadata)
        if len(loaded) != len(frame):
            raise ValueError("persisted row count does not match metadata")
        _publish_partition(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_prepared_partition(path: Path | str, *, expected: Mapping[str, object]) -> pd.DataFrame:
    root = Path(path)
    metadata_path = root / "metadata.json"
    data_path = root / "data.parquet"
    if not metadata_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"incomplete prepared partition: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_metadata = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing_metadata:
        labels = [key.replace("_", " ") for key in missing_metadata]
        raise ValueError(f"missing prepared metadata: {labels}")
    if metadata["schema_version"] != PREPARATION_SCHEMA_VERSION:
        raise ValueError("prepared metadata schema version is unsupported")
    if not isinstance(metadata["row_count"], int) or isinstance(metadata["row_count"], bool):
        raise ValueError("prepared metadata row count must be an integer")
    feature_columns = _validate_feature_columns(metadata["feature_columns"])
    configuration = _metadata_window_configuration(metadata["window_configuration"])
    if configuration == "whole_text":
        if metadata["stride"] is not None or metadata["aggregation_rules"] != "whole_text":
            raise ValueError("prepared metadata has invalid whole-text semantics")
    elif (metadata["stride"] is None or metadata["aggregation_rules"] != "deterministic sliding token windows"):
        raise ValueError("prepared metadata has invalid sliding-window semantics")
    else:
        _validate_stride(metadata["stride"])
    for key, value in expected.items():
        if metadata.get(key) != value:
            label = key.replace("_", " ")
            raise ValueError(f"prepared metadata mismatch for {label}")
    frame = pd.read_parquet(data_path)
    missing_identity = set(_identity_columns()).difference(frame.columns)
    if missing_identity:
        raise ValueError(f"missing prepared partition identity columns: {sorted(missing_identity)}")
    missing = {"split", "text_id", "label"}.difference(frame.columns) | set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prepared columns: {sorted(missing)}")
    if len(frame) != metadata["row_count"]:
        raise ValueError("prepared metadata row count does not match data")
    _validate_partition_rows(frame, split=metadata["split"], window_configuration=configuration)
    _validate_frame_labels(frame)
    if frame.duplicated(_identity_columns()).any():
        raise ValueError("duplicate prepared partition identity")
    return frame


def ensure_prepared_partition(
    data_root: Path | str, processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
) -> dict[str, object]:
    try:
        _, local_columns = _feature_columns(feature_set)
    except Exception as error:
        path = prepared_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                       feature_set=feature_set, window_configuration=window_configuration)
        return {"status": "unavailable", "path": path, "error": str(error)}
    path = prepared_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                   feature_set=feature_set, window_configuration=window_configuration)
    try:
        expected = {"scorer_llm": scorer_llm, "dataset": dataset, "split": split,
                    "feature_set": feature_set.name, "feature_columns": list(feature_set.source_columns),
                    "window_configuration": _window_configuration(window_configuration),
                    "schema_version": PREPARATION_SCHEMA_VERSION,
                    "stride": None if window_configuration == "whole_text" else stride}
        if path.exists():
            load_prepared_partition(path, expected=expected)
        if local_columns and not _similarity_paths(Path(processed_root), dataset, scorer_llm, split):
            raise FileNotFoundError("missing token similarity data for local feature preparation")
        canonical = ensure_canonical_partition(
            data_root, processed_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
            window_configuration=window_configuration, stride=stride,
        )
        if canonical["status"] != "available":
            return {"status": "unavailable", "path": path, "error": canonical["error"]}
        canonical_frame = load_canonical_partition(canonical["path"], expected=canonical["expected"])
        missing_features = set(feature_set.source_columns).difference(canonical_frame.columns)
        if missing_features:
            raise FileNotFoundError("missing token similarity data for local feature preparation")
        projection_columns = [
            "scorer_llm", "dataset", "split", "text_id", "label", "start_token", "stop_token",
            "window_configuration", *feature_set.source_columns,
        ]
        frame = canonical_frame.loc[:, projection_columns]
        if path.exists():
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("canonical_source_manifest") == canonical["expected"]["source_manifest"]:
                pass
            else:
                save_prepared_partition(
                    data_root, frame, scorer_llm=scorer_llm, dataset=dataset, split=split,
                    feature_set=feature_set, window_configuration=window_configuration, stride=stride,
                    canonical_source_manifest=canonical["expected"]["source_manifest"], replace=True,
                )
        else:
            save_prepared_partition(data_root, frame, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                    feature_set=feature_set, window_configuration=window_configuration, stride=stride,
                                    canonical_source_manifest=canonical["expected"]["source_manifest"])
        return {"status": "available", "path": path}
    except Exception as error:
        return {"status": "unavailable", "path": path, "error": str(error)}
