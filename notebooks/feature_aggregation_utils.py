from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Literal, Mapping

import pandas as pd

from notebooks.local_evidence_utils import _window_starts
from scripts.logprob_metrics import compute_logprob_metrics


WindowConfiguration = int | Literal["whole_text"]
PREPARATION_SCHEMA_VERSION = 1


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


def prepare_feature_partition(
    processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
) -> pd.DataFrame:
    configuration = _window_configuration(window_configuration)
    _validate_stride(stride)
    unknown = set(feature_set.source_columns).difference(METRIC_REGISTRY)
    if unknown:
        raise ValueError(f"feature metrics are not registered: {sorted(unknown)}")
    rows: list[dict[str, object]] = []
    seen_text_ids: set[int] = set()
    for path in _processed_paths(Path(processed_root), dataset, scorer_llm, split):
        for source in pd.read_parquet(path).to_dict(orient="records"):
            text_id = int(source["id"])
            if text_id in seen_text_ids:
                raise ValueError(f"duplicate source text identity: {split}/{text_id}")
            seen_text_ids.add(text_id)
            raw_tokens = source.get("tokens")
            tokens = [] if raw_tokens is None else list(raw_tokens)
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
                    **{column: METRIC_REGISTRY[column].compute(metrics) for column in feature_set.source_columns},
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


def prepared_partition_path(
    data_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet | str, window_configuration: WindowConfiguration,
) -> Path:
    configuration = _window_configuration(window_configuration)
    name = feature_set.name if isinstance(feature_set, FeatureSet) else feature_set
    return (Path(data_root) / "experiment-results" / "classification" / "prepared" /
            name / scorer_llm / dataset / split / configuration)


def _identity_columns() -> list[str]:
    return ["split", "text_id", "start_token", "stop_token"]


def save_prepared_partition(
    data_root: Path | str, frame: pd.DataFrame, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
) -> Path:
    configuration = _window_configuration(window_configuration)
    _validate_stride(stride)
    missing = {"split", "text_id", "label"}.difference(frame.columns) | set(feature_set.source_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prepared columns: {sorted(missing)}")
    _validate_frame_labels(frame)
    if set(_identity_columns()).issubset(frame.columns) and frame.duplicated(_identity_columns()).any():
        raise ValueError("duplicate prepared partition identity")
    destination = prepared_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                          feature_set=feature_set, window_configuration=window_configuration)
    if destination.exists():
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
    try:
        temporary.mkdir(parents=True)
        frame.to_parquet(temporary / "data.parquet", index=False)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        loaded = load_prepared_partition(temporary, expected=metadata)
        if len(loaded) != len(frame):
            raise ValueError("persisted row count does not match metadata")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
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
    for key, value in expected.items():
        if metadata.get(key) != value:
            label = key.replace("_", " ")
            raise ValueError(f"prepared metadata mismatch for {label}")
    frame = pd.read_parquet(data_path)
    missing = {"split", "text_id", "label"}.difference(frame.columns) | set(metadata.get("feature_columns", ())).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prepared columns: {sorted(missing)}")
    if len(frame) != metadata["row_count"]:
        raise ValueError("prepared metadata row count does not match data")
    _validate_frame_labels(frame)
    if set(_identity_columns()).issubset(frame.columns) and frame.duplicated(_identity_columns()).any():
        raise ValueError("duplicate prepared partition identity")
    return frame


def ensure_prepared_partition(
    data_root: Path | str, processed_root: Path | str, *, scorer_llm: str, dataset: str, split: str,
    feature_set: FeatureSet, window_configuration: WindowConfiguration, stride: int,
) -> dict[str, object]:
    path = prepared_partition_path(data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                   feature_set=feature_set, window_configuration=window_configuration)
    try:
        if path.exists():
            load_prepared_partition(path, expected={"scorer_llm": scorer_llm, "dataset": dataset,
                                                     "split": split, "feature_set": feature_set.name,
                                                     "feature_columns": list(feature_set.source_columns),
                                                     "window_configuration": _window_configuration(window_configuration),
                                                     "schema_version": PREPARATION_SCHEMA_VERSION,
                                                     "stride": None if window_configuration == "whole_text" else stride})
        else:
            frame = prepare_feature_partition(processed_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                              feature_set=feature_set, window_configuration=window_configuration, stride=stride)
            save_prepared_partition(data_root, frame, scorer_llm=scorer_llm, dataset=dataset, split=split,
                                    feature_set=feature_set, window_configuration=window_configuration, stride=stride)
        return {"status": "available", "path": path}
    except Exception as error:
        return {"status": "unavailable", "path": path, "error": str(error)}
