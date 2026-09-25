from __future__ import annotations

import gc
import json
from itertools import combinations
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import pandas as pd

from notebooks.utils.classification_utils import evaluate_scores
from notebooks.utils.feature_aggregation_utils import (
    PREPARATION_SCHEMA_VERSION,
    WindowConfiguration,
    load_prepared_partition,
    prepared_partition_path,
)
from notebooks.utils.model_training_utils import (
    FittedFeatureVectorModel,
    load_model_metadata,
    score_feature_vector_model,
)
from notebooks.utils.cross_dataset_classification_utils import build_deterministic_parent_cohorts


DATASETS = {
    "Kaggle": "kaggle_train",
    "MAGE": "mage_train",
    "Outfox": "outfox_chatgpt_train",
    "BBC": "bbc_news_content",
}
DATASET_ORDER = tuple(DATASETS)
SOURCE_SPLITS = {name: ("HG", "MG") for name in DATASETS if name != "BBC"} | {"BBC": ("HG",)}
TRAINING_REGIMES = {
    "Kaggle": ("Kaggle",),
    "MAGE": ("MAGE",),
    "Outfox": ("Outfox",),
    "Mixed": ("Kaggle", "MAGE", "Outfox"),
    "Mixed+BBC": ("Kaggle", "MAGE", "Outfox", "BBC"),
}
EVALUATION_REGIMES = {**TRAINING_REGIMES, "BBC": ("BBC",)}


class ExperimentResultKey(NamedTuple):
    training_composition: str
    evaluation_composition: str
    scorer_llm: str
    feature_set: str
    training_window: str
    evaluation_window: str
    model_family: str


def _canonical_dataset_names(dataset_names: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(dataset_names).difference(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unknown dataset names: {unknown}")
    return tuple(name for name in DATASET_ORDER if name in dataset_names)


def dataset_subsets(
    dataset_names: tuple[str, ...] = DATASET_ORDER,
) -> tuple[tuple[str, ...], ...]:
    canonical = _canonical_dataset_names(dataset_names)
    return tuple(
        subset
        for size in range(1, len(canonical) + 1)
        for subset in combinations(canonical, size)
    )


def composition_id(dataset_names: tuple[str, ...]) -> str:
    return "+".join(_canonical_dataset_names(dataset_names))


def composition_has_both_classes(
    composition: tuple[str, ...],
    source_splits: Mapping[str, Sequence[str]],
) -> bool:
    splits = {split for name in composition for split in source_splits.get(name, ())}
    return {"HG", "MG"}.issubset(splits)


def threshold_source_is_valid(threshold_source: str) -> bool:
    return threshold_source in {"training", "validation"}


def result_key(
    *,
    training_composition: str,
    evaluation_composition: str,
    scorer_llm: str,
    feature_set: str,
    training_window: str,
    evaluation_window: str,
    model_family: str,
) -> ExperimentResultKey:
    return ExperimentResultKey(
        training_composition,
        evaluation_composition,
        scorer_llm,
        feature_set,
        training_window,
        evaluation_window,
        model_family,
    )

_TRAINING_RESULT_COLUMNS = (
    "model_family", "scorer_llm", "feature_set", "training_regime", "training_window", "partition",
)
_EVALUATION_RESULT_COLUMNS = (
    "model_family", "scorer_llm", "feature_set", "training_regime",
    "training_window", "evaluation_regime", "evaluation_window", "partition",
)
_COVERAGE_RESULT_COLUMNS = (
    "dataset", "split", "class", "parent_count", "row_count", "availability_status",
)
_RUN_SUMMARY_COLUMNS = (
    "run_id", "stage", "status", "scorer_llm", "dataset", "split", "feature_set",
    "training_window", "evaluation_window", "model_family", "artifact_path", "error",
)
_STANDARD_RESULT_NAMES = (
    "training_metrics", "training_coverage", "evaluation_metrics",
    "evaluation_coverage", "run_summary",
)
_SEMEVAL_RESULT_NAMES = (
    "semeval_summary", "semeval_evaluations", "semeval_predicted_regions",
    "semeval_boundary_summary", "semeval_boundary_predictions",
    "semeval_boundary_exclusions",
)


def build_feature_parent_cohorts(
    frame: pd.DataFrame, *, seed: int, max_per_split: int = 600,
    min_text_tokens: int | None = None,
) -> dict[str, set[tuple[str, str, object]]]:
    required = {"dataset", "split", "text_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing cohort columns: {missing}")
    result = {"train": set(), "validation": set(), "test": set()}
    for dataset, source in frame.groupby("dataset", sort=True):
        parents = source[["split", "text_id", "label"]].drop_duplicates()
        if min_text_tokens is not None:
            if "token_count" in source:
                token_counts = source.groupby(["split", "text_id"], as_index=False)["token_count"].max()
            elif "stop_token" in source:
                token_counts = source.groupby(["split", "text_id"], as_index=False)["stop_token"].max().rename(
                    columns={"stop_token": "token_count"}
                )
            else:
                raise ValueError("token_count or stop_token is required for minimum token filtering")
            parents = parents.merge(token_counts, on=["split", "text_id"], how="left", validate="one_to_one")
        partitions = build_deterministic_parent_cohorts(
            parents, id_column="text_id", max_per_split=max_per_split,
            min_text_tokens=min_text_tokens, seed=seed,
        )
        for partition, rows in partitions.items():
            result[partition].update(
                (str(dataset), str(split), text_id)
                for split, text_id in rows[["split", "text_id"]].itertuples(index=False, name=None)
            )
    return result


def _prepared_expected(
    *, scorer_llm: str, dataset: str, split: str, feature_set: str,
    window_configuration: WindowConfiguration, feature_columns: Sequence[str],
) -> dict[str, object]:
    configuration = "whole_text" if window_configuration == "whole_text" else str(int(window_configuration))
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "scorer_llm": scorer_llm,
        "dataset": dataset,
        "split": split,
        "feature_set": feature_set,
        "feature_columns": list(feature_columns),
        "window_configuration": configuration,
    }


def load_episode_partitions(
    data_root: Path | str, *, scorer_llm: str, datasets: tuple[str, ...], partition: str,
    feature_set: str, window_configuration: WindowConfiguration,
    feature_columns: tuple[str, ...], cohort: set[tuple[str, str, object]],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        rows: list[pd.DataFrame] = []
        for split in ("HG", "MG"):
            selected_ids = {identity[2] for identity in cohort if identity[:2] == (dataset, split)}
            if not selected_ids:
                continue
            path = prepared_partition_path(
                data_root, scorer_llm=scorer_llm, dataset=dataset, split=split,
                feature_set=feature_set, window_configuration=window_configuration,
            )
            prepared = load_prepared_partition(
                path,
                expected=_prepared_expected(
                    scorer_llm=scorer_llm, dataset=dataset, split=split,
                    feature_set=feature_set, window_configuration=window_configuration,
                    feature_columns=feature_columns,
                ),
            )
            filtered = prepared[prepared["text_id"].isin(selected_ids)].copy()
            filtered["dataset"] = dataset
            rows.append(filtered)
        if rows:
            frames[dataset] = pd.concat(rows, ignore_index=True)
    return frames


def load_training_episode(
    data_root: Path | str, *, scorer_llm: str, datasets: tuple[str, ...],
    feature_set: str, window_configuration: WindowConfiguration,
    feature_columns: tuple[str, ...], train_cohort: set[tuple[str, str, object]],
    validation_cohort: set[tuple[str, str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_episode_partitions(
        data_root, scorer_llm=scorer_llm, datasets=datasets, partition="train",
        feature_set=feature_set, window_configuration=window_configuration,
        feature_columns=feature_columns, cohort=train_cohort,
    )
    validation = load_episode_partitions(
        data_root, scorer_llm=scorer_llm, datasets=datasets, partition="validation",
        feature_set=feature_set, window_configuration=window_configuration,
        feature_columns=feature_columns, cohort=validation_cohort,
    )
    if not train or not validation:
        raise ValueError("training episode has no train or validation frames")
    return pd.concat(train.values(), ignore_index=True), pd.concat(validation.values(), ignore_index=True)


def _read_artifact_metadata(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def discover_model_artifacts(models_root: Path | str, *, filters: Mapping[str, object]) -> list[Path]:
    matches: list[Path] = []
    for metadata_path in sorted(Path(models_root).rglob("metadata.json")):
        metadata = _read_artifact_metadata(metadata_path.parent)
        if metadata is not None and all(metadata.get(key) == value for key, value in filters.items()):
            matches.append(metadata_path.parent)
    return matches


def prediction_artifact_key(
    artifact_path: Path | str,
    *,
    models_root: Path | str,
    evaluation_regime: str,
    evaluation_window: str,
    dataset: str,
    partition: str,
) -> dict[str, str]:
    """Build the prediction-artifact key from the model path and evaluation row."""
    artifact_id = "__".join(Path(artifact_path).relative_to(Path(models_root)).parts)
    return {
        "artifact_id": artifact_id,
        "evaluation_regime": evaluation_regime,
        "evaluation_window": evaluation_window,
        "dataset": dataset,
        "partition": partition,
    }


def validate_evaluation_metadata(
    artifact_metadata: Mapping[str, object], prepared_metadata: Mapping[str, object]
) -> None:
    if "scorer_llm" not in artifact_metadata or "scorer_llm" not in prepared_metadata:
        raise ValueError("scorer_llm metadata is required")
    if artifact_metadata["scorer_llm"] != prepared_metadata["scorer_llm"]:
        raise ValueError("scorer_llm does not match")
    if "feature_set" not in artifact_metadata or "feature_set" not in prepared_metadata:
        raise ValueError("feature set metadata is required")
    if artifact_metadata["feature_set"] != prepared_metadata["feature_set"]:
        raise ValueError("feature set does not match")
    if "feature_columns" not in artifact_metadata or "feature_columns" not in prepared_metadata:
        raise ValueError("feature columns metadata is required")
    try:
        artifact_columns = tuple(artifact_metadata["feature_columns"])  # type: ignore[arg-type]
        prepared_columns = tuple(prepared_metadata["feature_columns"])  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("feature columns metadata must be iterable") from error
    if artifact_columns != prepared_columns:
        raise ValueError("feature columns do not match")


def evaluate_feature_vector_model(
    artifact: FittedFeatureVectorModel, frames: Mapping[str, pd.DataFrame], *,
    partition: str, metadata: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not frames:
        raise ValueError("evaluation frames must not be empty")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        if frame.empty:
            raise ValueError(f"empty evaluation frame for dataset={dataset!r}")
        prepared_metadata = {
            "scorer_llm": frame.get("scorer_llm", pd.Series([metadata.get("scorer_llm")])).iloc[0],
            "feature_set": frame.get("feature_set", pd.Series([metadata.get("feature_set")])).iloc[0],
            "feature_columns": list(artifact.feature_columns),
            "window_configuration": frame.get("window_configuration", pd.Series([metadata.get("evaluation_window")])).iloc[0],
        }
        validate_evaluation_metadata(metadata, prepared_metadata)
        scored = score_feature_vector_model(artifact, frame)
        output_metadata = {
            key: value for key, value in metadata.items()
            if key != "feature_columns" and not isinstance(value, (list, tuple, dict))
        }
        scored = scored.assign(**output_metadata, dataset=dataset, partition=partition)
        predictions.append(scored)
        scores = scored["mg_score"]
        metrics.append({**output_metadata, "dataset": dataset, "partition": partition, **evaluate_scores(scored["label"], scores, 0.5)})
        split_values = tuple(sorted(str(value) for value in scored["split"].dropna().unique()))
        coverage.append({
            **output_metadata,
            "dataset": dataset,
            "partition": partition,
            "split": "+".join(split_values),
            "class": "binary" if set(scored["label"]) == {0, 1} else split_values[0],
            "parent_count": int(scored[["split", "text_id"]].drop_duplicates().shape[0]),
            "row_count": len(scored),
            "availability_status": "available",
            "rows": len(scored),
            "hg_rows": int(scored["label"].eq(0).sum()),
            "mg_rows": int(scored["label"].eq(1).sum()),
        })
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(coverage)


def availability_row(
    *, stage: str, status: str, scorer_llm: str, dataset: str, split: str,
    feature_set: str, window_configuration: WindowConfiguration, error: object,
) -> dict[str, object]:
    return {
        "stage": stage, "status": status, "scorer_llm": scorer_llm, "dataset": dataset,
        "split": split, "feature_set": feature_set, "window_configuration": str(window_configuration),
        "error": str(error),
    }


def append_result_frame(path: Path | str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        frame.to_csv(destination, index=False)
        return
    try:
        existing_columns = list(pd.read_csv(destination, nrows=0).columns)
    except pd.errors.EmptyDataError:
        frame.to_csv(destination, index=False)
        return
    if existing_columns != list(frame.columns):
        if pd.read_csv(destination, nrows=1).empty:
            frame.to_csv(destination, index=False)
            return
        if set(frame.columns).issubset(existing_columns):
            frame = frame.reindex(columns=existing_columns)
        else:
            raise ValueError(f"result frame columns do not match existing CSV: {destination}")
    frame.to_csv(destination, mode="a", header=False, index=False)


def result_export_paths(data_root: Path | str, run_id: str) -> dict[str, Path]:
    destination = Path(data_root) / "experiment-results" / "classification" / "evaluations" / run_id
    return {
        name: destination / f"{name}.csv"
        for name in (*_STANDARD_RESULT_NAMES, *_SEMEVAL_RESULT_NAMES)
    }


def ensure_result_exports(
    data_root: Path | str, run_id: str,
) -> tuple[dict[str, Path], dict[str, Path]]:
    paths = result_export_paths(data_root, run_id)
    destination = next(iter(paths.values())).parent
    destination.mkdir(parents=True, exist_ok=True)
    standard = {name: paths[name] for name in _STANDARD_RESULT_NAMES}
    semeval = {name: paths[name] for name in _SEMEVAL_RESULT_NAMES}
    standard_columns = {
        "training_metrics": _TRAINING_RESULT_COLUMNS,
        "training_coverage": _COVERAGE_RESULT_COLUMNS,
        "evaluation_metrics": _EVALUATION_RESULT_COLUMNS,
        "evaluation_coverage": _COVERAGE_RESULT_COLUMNS,
        "run_summary": _RUN_SUMMARY_COLUMNS,
    }
    for name, path in standard.items():
        if not path.exists():
            pd.DataFrame(columns=standard_columns[name]).to_csv(path, index=False)
    for path in semeval.values():
        if not path.exists():
            path.touch()
    return standard, semeval


def read_existing_result_table(path: Path | str) -> pd.DataFrame:
    destination = Path(path)
    if not destination.exists() or destination.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(destination)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _result_key_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def successful_result_keys(
    path: Path | str,
    columns: tuple[str, ...],
    *,
    stage: str | None = None,
) -> set[tuple[str, ...]]:
    table = read_existing_result_table(path)
    if table.empty or any(column not in table.columns for column in columns):
        return set()
    status_column = "status" if "status" in table.columns else "availability_status"
    if status_column in table.columns:
        table = table.loc[table[status_column].astype(str).eq("available")]
    if stage is not None and "stage" in table.columns:
        table = table.loc[table["stage"].astype(str).eq(stage)]
    keys: set[tuple[str, ...]] = set()
    for values in table.loc[:, list(columns)].itertuples(index=False, name=None):
        key = tuple(_result_key_value(value) for value in values)
        if all(value is not None for value in key):
            keys.add(key)  # type: ignore[arg-type]
    return keys


def export_result_frames(
    root: Path | str, *, run_id: str, frames: Mapping[str, pd.DataFrame]
) -> dict[str, Path]:
    destination = Path(root) / "experiment-results" / "classification" / "evaluations" / run_id
    names = (
        "training_metrics", "training_coverage", "evaluation_metrics",
        "evaluation_coverage", "run_summary",
    )
    output = {name: frames.get(name, pd.DataFrame()) for name in names}
    for name in names[:-1]:
        if name not in frames or (
            output[name].empty and not set(output[name].columns)
        ):
            if name.endswith("coverage"):
                columns = _COVERAGE_RESULT_COLUMNS
            else:
                columns = _TRAINING_RESULT_COLUMNS if name.startswith("training_") else _EVALUATION_RESULT_COLUMNS
            output[name] = pd.DataFrame(columns=columns)
    run_summary_columns = _RUN_SUMMARY_COLUMNS
    if output["run_summary"].empty and not set(output["run_summary"].columns):
        output["run_summary"] = pd.DataFrame(columns=run_summary_columns)
    elif "run_id" not in output["run_summary"]:
        output["run_summary"] = output["run_summary"].assign(run_id=run_id)
    for name, frame in output.items():
        if name == "run_summary":
            required = run_summary_columns
        elif name.endswith("coverage"):
            required = _COVERAGE_RESULT_COLUMNS
        elif name.startswith("training_"):
            required = _TRAINING_RESULT_COLUMNS
        else:
            required = _EVALUATION_RESULT_COLUMNS
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"result frame is missing required dimension columns: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in output.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


def export_semeval_result_frames(
    root: Path | str, *, run_id: str, frames: Mapping[str, pd.DataFrame]
) -> dict[str, Path]:
    destination = Path(root) / "experiment-results" / "classification" / "evaluations" / run_id
    names = (
        "semeval_summary", "semeval_evaluations", "semeval_predicted_regions",
        "semeval_boundary_summary", "semeval_boundary_predictions",
        "semeval_boundary_exclusions",
    )
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in names:
        path = destination / f"{name}.csv"
        frames.get(name, pd.DataFrame()).to_csv(path, index=False)
        written[name] = path
    return written


def release_frames(*frames: pd.DataFrame | None) -> None:
    del frames
    gc.collect()
