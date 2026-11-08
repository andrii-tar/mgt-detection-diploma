"""Loading, scoring, and metric helpers for the paper comparison report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from notebooks.utils.document_level_model_experiment_utils import MODEL_FEATURE_COLUMNS
from notebooks.utils.model_training_utils import (
    load_model_artifact,
    load_model_metadata,
    score_feature_vector_model,
)
from notebooks.utils.semeval_boundary_utils import gold_token_boundary


SIMPLE_SIMILARITY_COLUMNS = (
    "mean_candidate_similarity__mean",
    "mean_candidate_similarity__std",
    "top20_coverage__mean",
    "top20_coverage__std",
    "observed_logprob__mean",
    "observed_logprob__std",
    "mean_candidate_logprob_delta__mean",
    "mean_candidate_logprob_delta__std",
)


def _is_temporary_path(path: Path, root: Path) -> bool:
    return any(part.startswith(".") or part.endswith(".tmp") for part in path.relative_to(root).parts[:-1])


def load_evaluation_window_rows(
    prediction_root: Path,
    *,
    run_id: str,
    scorer_llm: str,
    model_family: str,
    training_window: int | str,
    evaluation_regime: str,
    evaluation_window: int | str,
    feature_set: str = "simple_similarity",
    training_regime: str = "Kaggle+MAGE+Outfox",
    partition: str = "test",
) -> pd.DataFrame:
    """Load finalized first-stage window prediction artifacts for one exact config."""
    artifact_id = "__".join((
        model_family,
        scorer_llm,
        feature_set,
        training_regime.replace("+", "_"),
        str(training_window),
        run_id,
    ))
    source_aware_frames = []
    legacy_frames = []
    for metadata_path in sorted(Path(prediction_root).rglob("metadata.json")):
        if _is_temporary_path(metadata_path, Path(prediction_root)):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("artifact_id") != artifact_id
            or str(metadata.get("evaluation_regime")) != evaluation_regime
            or str(metadata.get("evaluation_window")) != str(evaluation_window)
            or metadata.get("partition") != partition
        ):
            continue
        data_path = metadata_path.parent / "data.parquet"
        if not data_path.is_file():
            continue
        try:
            frame = pd.read_parquet(data_path)
        except (OSError, ValueError):
            continue
        has_source_dataset = "source_dataset" in frame
        if not has_source_dataset and "dataset" in frame:
            frame["source_dataset"] = frame["dataset"]
        required = {
            "source_dataset", "split", "text_id", "label", "start_token", "mg_score", "prediction",
            "model_family", "scorer_llm", "training_regime", "training_window", "evaluation_regime",
            "evaluation_window", "partition",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"prediction artifact {data_path} is missing columns: {missing}")
        expected = {
            "model_family": model_family,
            "scorer_llm": scorer_llm,
            "feature_set": feature_set,
            "training_regime": training_regime,
            "training_window": str(training_window),
            "evaluation_regime": evaluation_regime,
            "evaluation_window": str(evaluation_window),
            "partition": partition,
        }
        for column, value in expected.items():
            if not frame[column].astype(str).eq(value).all():
                raise ValueError(f"prediction artifact {data_path} has inconsistent {column}")
        (source_aware_frames if has_source_dataset else legacy_frames).append(frame)
    # Evaluation exports may contain both a canonical combined partition with
    # source_dataset provenance and per-dataset child partitions. Prefer the
    # canonical source-aware rows to avoid counting each window twice.
    frames = source_aware_frames or legacy_frames
    if not frames:
        raise FileNotFoundError(
            f"no finalized prediction artifacts for {model_family}/{scorer_llm}/"
            f"train{training_window}/eval{evaluation_window}/{evaluation_regime}/{partition}"
        )
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["source_dataset", "split", "text_id", "start_token"]).any():
        raise ValueError("duplicate window identities across prediction artifacts")
    result["training_window"] = int(training_window) if str(training_window).isdigit() else str(training_window)
    result["evaluation_window"] = int(evaluation_window) if str(evaluation_window).isdigit() else str(evaluation_window)
    return result


def score_prepared_window_partition(
    data_root: Path,
    *,
    run_id: str,
    scorer_llm: str,
    dataset: str,
    split: str,
    model_family: str,
    training_regime: str,
    training_window: int | str,
    evaluation_window: int | str,
) -> pd.DataFrame:
    """Score cached prepared windows with a stored first-stage model, without fitting it."""
    from notebooks.utils.classification_experiment_utils import discover_model_artifacts
    from notebooks.utils.feature_aggregation_utils import FeatureSet, load_prepared_partition, prepared_partition_path

    feature_set = FeatureSet("simple_similarity", SIMPLE_SIMILARITY_COLUMNS)
    prepared_path = prepared_partition_path(
        data_root,
        scorer_llm=scorer_llm,
        dataset=dataset,
        split=split,
        feature_set=feature_set,
        window_configuration=evaluation_window,
    )
    expected_prepared = {
        "scorer_llm": scorer_llm,
        "dataset": dataset,
        "split": split,
        "feature_set": feature_set.name,
        "feature_columns": list(SIMPLE_SIMILARITY_COLUMNS),
        "window_configuration": str(evaluation_window),
        "schema_version": 1,
        "stride": 16 if str(evaluation_window).isdigit() else None,
    }
    windows = load_prepared_partition(prepared_path, expected=expected_prepared)

    models_root = Path(data_root) / "experiment-results" / "classification" / "models"
    matches = discover_model_artifacts(models_root, filters={
        "run_id": run_id,
        "model_family": model_family,
        "scorer_llm": scorer_llm,
        "feature_set": feature_set.name,
        "training_regime": training_regime,
        "training_window": str(training_window),
    })
    if len(matches) != 1:
        raise ValueError(f"expected one stored classifier artifact, found {matches}")
    artifact = matches[0]
    expected_model = {
        "run_id": run_id,
        "model_family": model_family,
        "scorer_llm": scorer_llm,
        "feature_set": feature_set.name,
        "training_regime": training_regime,
        "training_window": str(training_window),
    }
    model_metadata = load_model_metadata(artifact, expected=expected_model)
    if model_metadata["feature_columns"] != list(SIMPLE_SIMILARITY_COLUMNS):
        raise ValueError(f"stored model feature schema mismatch: {artifact}")
    model = load_model_artifact(
        artifact,
        expected={"feature_columns": list(SIMPLE_SIMILARITY_COLUMNS)},
        trust_artifact=True,
    )
    if model_family == "random_forest":
        model.estimator.set_params(model__n_jobs=1)
    scoring_columns = ["split", "text_id", "label", "start_token", "stop_token", *SIMPLE_SIMILARITY_COLUMNS]
    scored = score_feature_vector_model(model, windows[scoring_columns])
    scored["source_dataset"] = dataset
    scored["model_family"] = model_family
    scored["scorer_llm"] = scorer_llm
    scored["training_regime"] = training_regime
    scored["training_window"] = int(training_window) if str(training_window).isdigit() else str(training_window)
    scored["evaluation_regime"] = dataset
    scored["evaluation_window"] = int(evaluation_window) if str(evaluation_window).isdigit() else str(evaluation_window)
    scored["partition"] = split
    return scored


def load_processed_token_counts(
    data_root: Path,
    *,
    scorer_llm: str,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """Return scorer-tokenizer lengths from existing Stage-1 token artifacts."""
    paths = sorted((Path(data_root) / "processed" / f"{dataset}__{scorer_llm}" / split / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Stage-1 token data found for {dataset}/{scorer_llm}/{split}")
    frames = [pd.read_parquet(path, columns=["id", "tokens"]) for path in paths]
    tokens = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    tokens["id"] = tokens["id"].astype(str).str.strip()
    tokens["token_count"] = tokens["tokens"].map(len).astype("int64")
    return tokens[["id", "token_count"]]


def load_semeval_token_labels(
    data_root: Path,
    *,
    scorer_llm: str,
    dataset: str = "semeval2024_task8_subtask_c_train",
    split: str = "MG",
) -> dict[int, list[str]]:
    """Load decoded scorer token strings needed to map gold SemEval word boundaries."""
    paths = sorted((Path(data_root) / "processed" / f"{dataset}__{scorer_llm}" / split / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Stage-1 token data found for {dataset}/{scorer_llm}/{split}")
    tokens = pd.concat([pd.read_parquet(path, columns=["id", "tokens"]) for path in paths], ignore_index=True)
    tokens = tokens.drop_duplicates("id")
    return {
        int(text_id): [str(token["token"]) for token in token_rows]
        for text_id, token_rows in zip(tokens["id"], tokens["tokens"])
    }


def load_document_word_counts(data_root: Path, documents: pd.DataFrame) -> pd.DataFrame:
    """Join model-independent whitespace word counts from raw files to selected document IDs."""
    from scripts.experiment_config import build_datasets

    required = {"source_dataset", "split", "text_id"}
    missing = sorted(required.difference(documents.columns))
    if missing:
        raise ValueError(f"document word-count input is missing columns: {missing}")
    if documents.empty:
        raise ValueError("document word-count input cannot be empty")
    dataset_configs = build_datasets(Path(data_root))
    outputs = []
    for (dataset, split), group in documents.groupby(["source_dataset", "split"], sort=True):
        if dataset not in dataset_configs or split not in dataset_configs[dataset]["splits"]:
            raise KeyError(f"raw text path is not configured for {dataset}/{split}")
        path = Path(dataset_configs[dataset]["splits"][split])
        if not path.is_file():
            raise FileNotFoundError(f"raw text file is missing: {path}")
        requested_ids = set(group["text_id"].astype(str).str.strip())
        raw = pd.read_csv(path, usecols=["id", "text"], dtype={"id": str})
        raw["text_id"] = raw["id"].astype(str).str.strip()
        selected = raw.loc[raw["text_id"].isin(requested_ids), ["text_id", "text"]].copy()
        found_ids = set(selected["text_id"])
        if found_ids != requested_ids:
            raise KeyError(f"raw texts missing for {dataset}/{split}: {sorted(requested_ids - found_ids)[:10]}")
        selected["source_dataset"] = dataset
        selected["split"] = split
        selected["word_count"] = selected["text"].fillna("").astype(str).map(lambda value: len(value.split()))
        outputs.append(selected[["source_dataset", "split", "text_id", "word_count"]])
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["source_dataset", "split", "text_id"]).any():
        raise ValueError("raw word-count mapping produced duplicate document keys")
    return result


def normalize_t08_aggregation_labels(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep T08 aggregation labels consistent with each row's evaluation representation."""
    if "evaluation_representation" not in predictions or "aggregation" not in predictions:
        raise ValueError("T08 predictions require evaluation_representation and aggregation columns")
    normalized = predictions.copy()
    representation = normalized["evaluation_representation"].astype(str)
    normalized.loc[representation.eq("whole_text"), "aggregation"] = "single_whole_text_vector"
    normalized.loc[representation.str.startswith("mean_window"), "aggregation"] = "mean_window_score"
    return normalized


def label_semeval_windows(
    windows: pd.DataFrame,
    annotations: pd.DataFrame,
    token_labels_by_id: Mapping[int, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label pure SemEval windows from gold boundaries and exclude boundary-crossing windows."""
    required_windows = {"text_id", "start_token", "stop_token", "prediction", "mg_score"}
    required_annotations = {"id", "text", "boundary_word_index"}
    if required_windows.difference(windows.columns):
        raise ValueError(f"SemEval windows are missing columns: {sorted(required_windows.difference(windows.columns))}")
    if required_annotations.difference(annotations.columns):
        raise ValueError(f"SemEval annotations are missing columns: {sorted(required_annotations.difference(annotations.columns))}")
    if annotations["id"].duplicated().any():
        raise ValueError("SemEval annotation IDs must be unique")
    annotation_by_id = annotations.set_index("id")
    pure_frames = []
    excluded_rows = []
    for text_id, group in windows.groupby("text_id", sort=False):
        text_id_int = int(text_id)
        if text_id_int not in annotation_by_id.index or text_id_int not in token_labels_by_id:
            raise KeyError(f"missing SemEval annotation or token labels for text_id={text_id_int}")
        annotation = annotation_by_id.loc[text_id_int]
        try:
            gold_boundary = gold_token_boundary(
                str(annotation["text"]),
                int(annotation["boundary_word_index"]),
                token_labels_by_id[text_id_int],
            )
        except ValueError as error:
            reason = str(error)
            allowed_reasons = (
                "token labels do not reconstruct the annotated text",
                "word boundary does not map to a decoded token",
                "boundary_word_index is outside the text word range",
            )
            if not reason.startswith(allowed_reasons):
                raise
            excluded_rows.append({
                "text_id": text_id_int,
                "gold_token_boundary": np.nan,
                "window_count": int(len(group)),
                "excluded_boundary_window_count": int(len(group)),
                "exclusion_reason": reason,
            })
            continue
        starts = pd.to_numeric(group["start_token"], errors="raise")
        stops = pd.to_numeric(group["stop_token"], errors="raise")
        if (stops <= starts).any():
            raise ValueError(f"invalid window spans for SemEval text_id={text_id_int}")
        hg_mask = stops <= gold_boundary
        mg_mask = starts >= gold_boundary
        crossing_mask = ~(hg_mask | mg_mask)
        pure = group.loc[hg_mask | mg_mask].copy()
        pure["gold_label"] = np.where(hg_mask[hg_mask | mg_mask].to_numpy(), 0, 1).astype(int)
        pure_frames.append(pure)
        excluded_rows.append({
            "text_id": text_id_int,
            "gold_token_boundary": int(gold_boundary),
            "window_count": int(len(group)),
            "excluded_boundary_window_count": int(crossing_mask.sum()),
            "exclusion_reason": "",
        })
    labeled = pd.concat(pure_frames, ignore_index=True) if pure_frames else windows.head(0).assign(gold_label=pd.Series(dtype="int64"))
    excluded = pd.DataFrame(excluded_rows)
    return labeled, excluded


def filter_window_rows_to_common_documents(
    windows: pd.DataFrame,
    *,
    comparison_group_columns: tuple[str, ...],
    configuration_columns: tuple[str, ...],
    document_key_columns: tuple[str, ...] = ("source_dataset", "split", "text_id"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep a common document cohort across configurations within each comparison group."""
    required = set(comparison_group_columns) | set(configuration_columns) | set(document_key_columns)
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"common-cohort rows are missing columns: {missing}")
    if windows.empty:
        raise ValueError("common-cohort input cannot be empty")
    filtered_frames = []
    coverage_rows = []
    outer_groups = windows.groupby(list(comparison_group_columns), dropna=False, sort=True) if comparison_group_columns else [((), windows)]
    for outer_key, outer_frame in outer_groups:
        outer_values = outer_key if isinstance(outer_key, tuple) else (outer_key,)
        config_groups = list(outer_frame.groupby(list(configuration_columns), dropna=False, sort=True))
        key_sets = []
        normalized_groups = []
        for config_key, config_frame in config_groups:
            keys = set(config_frame.loc[:, document_key_columns].astype(str).itertuples(index=False, name=None))
            key_sets.append(keys)
            normalized_groups.append((config_key, config_frame, keys))
        common_keys = set.intersection(*key_sets) if key_sets else set()
        if not common_keys:
            raise ValueError(f"no common document IDs for comparison group {outer_values}")
        for config_key, config_frame, keys in normalized_groups:
            config_values = config_key if isinstance(config_key, tuple) else (config_key,)
            row_keys = config_frame.loc[:, document_key_columns].astype(str).itertuples(index=False, name=None)
            mask = pd.Series([key in common_keys for key in row_keys], index=config_frame.index)
            filtered_frames.append(config_frame.loc[mask].copy())
            coverage = dict(zip(comparison_group_columns, outer_values))
            coverage.update(dict(zip(configuration_columns, config_values)))
            coverage.update({
                "available_document_count": int(len(keys)),
                "common_document_count": int(len(common_keys)),
                "excluded_document_count": int(len(keys - common_keys)),
            })
            coverage_rows.append(coverage)
    return pd.concat(filtered_frames, ignore_index=True), pd.DataFrame(coverage_rows)


def build_cohort_manifest(
    available_rows: pd.DataFrame,
    included_rows: pd.DataFrame,
    *,
    configuration_columns: tuple[str, ...],
    document_key_columns: tuple[str, ...] = ("source_dataset", "split", "text_id"),
    exclusion_reason: str = "excluded_by_comparison_protocol",
) -> pd.DataFrame:
    """Describe included/excluded document IDs and stable cohort hashes per config."""
    required = set(configuration_columns) | set(document_key_columns)
    for frame_name, frame in (("available_rows", available_rows), ("included_rows", included_rows)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} is missing manifest columns: {missing}")
    if available_rows.empty:
        raise ValueError("available_rows cannot be empty")
    rows = []
    groups = available_rows.groupby(list(configuration_columns), dropna=False, sort=True) if configuration_columns else [((), available_rows)]
    for config_key, config_rows in groups:
        config_values = config_key if isinstance(config_key, tuple) else (config_key,)
        config_mask = pd.Series(True, index=included_rows.index)
        for column, value in zip(configuration_columns, config_values):
            config_mask &= included_rows[column].eq(value)
        included_config = included_rows.loc[config_mask]
        included_keys = set(included_config.loc[:, document_key_columns].astype(str).itertuples(index=False, name=None))
        payload = "\n".join("\x1f".join(key) for key in sorted(included_keys))
        cohort_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        documents = config_rows.drop_duplicates(list(document_key_columns))
        for record in documents.to_dict(orient="records"):
            doc_key = tuple(str(record[column]) for column in document_key_columns)
            included = doc_key in included_keys
            row = {column: value for column, value in zip(configuration_columns, config_values)}
            row.update({column: record[column] for column in document_key_columns})
            row.update({
                "label": record.get("label", np.nan),
                "partition": record.get("partition", record.get("evaluation_partition", "")),
                "cohort_id": cohort_id,
                "inclusion_status": "included" if included else "excluded",
                "exclusion_reason": "" if included else exclusion_reason,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def build_configuration_coverage(
    expected: pd.DataFrame,
    available: pd.DataFrame,
    *,
    block: str,
    key_columns: tuple[str, ...],
    missing_reason: str,
) -> pd.DataFrame:
    """Return one explicit available/unavailable row for every expected config."""
    for name, frame in (("expected", expected), ("available", available)):
        missing = sorted(set(key_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} configuration table is missing keys: {missing}")
    if expected.duplicated(list(key_columns)).any():
        raise ValueError("expected configuration keys must be unique")
    available_keys = set(available.loc[:, key_columns].astype(str).itertuples(index=False, name=None))
    rows = []
    for record in expected.loc[:, key_columns].to_dict(orient="records"):
        key = tuple(str(record[column]) for column in key_columns)
        is_available = key in available_keys
        rows.append({
            "block": block,
            **record,
            "status": "available" if is_available else "unavailable",
            "reason": "" if is_available else missing_reason,
        })
    return pd.DataFrame(rows)


def binary_window_metrics(
    windows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    label_column: str = "label",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Compute macro-F1/accuracy per group; macro-F1 is NA if either class is absent."""
    required = {label_column, prediction_column, *group_columns}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"window metric input is missing columns: {missing}")
    if windows.empty:
        raise ValueError("window metric input cannot be empty")
    rows = []
    groups = windows.groupby(list(group_columns), dropna=False, sort=True) if group_columns else [((), windows)]
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        labels = group[label_column].astype(int).to_numpy()
        predictions = group[prediction_column].astype(int).to_numpy()
        if not set(np.unique(labels)).issubset({0, 1}) or not set(np.unique(predictions)).issubset({0, 1}):
            raise ValueError("window labels and predictions must be binary 0/1")
        classes = set(np.unique(labels))
        row = dict(zip(group_columns, key_values))
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        row.update({
            "window_count": int(len(group)),
            "hg_window_count": int(np.sum(labels == 0)),
            "mg_window_count": int(np.sum(labels == 1)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "mg_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
            "mg_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
            "mg_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_positive_rate": fp / (tn + fp) if tn + fp else np.nan,
            "roc_auc": float(roc_auc_score(labels, group["mg_score"].to_numpy(dtype=float)))
            if classes == {0, 1} and "mg_score" in group else np.nan,
            "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0))
            if classes == {0, 1} else np.nan,
            "metric_status": "two_class" if classes == {0, 1} else "single_class_no_macro_f1",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def single_class_rates(
    rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    label_column: str = "label",
    prediction_column: str = "prediction",
    unit_name: str = "sample",
) -> pd.DataFrame:
    """Report MG detection rate for MG-only groups or false-positive rate for HG-only groups."""
    required = {label_column, prediction_column, *group_columns}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"single-class metric input is missing columns: {missing}")
    if rows.empty:
        raise ValueError("single-class metric input cannot be empty")
    groups = rows.groupby(list(group_columns), dropna=False, sort=True) if group_columns else [((), rows)]
    results = []
    for key, group in groups:
        labels = group[label_column].astype(int).to_numpy()
        unique = set(np.unique(labels))
        if not unique.issubset({0, 1}):
            raise ValueError("single-class metrics require binary 0/1 labels")
        if len(unique) != 1:
            raise ValueError("single-class metrics require single-class groups")
        key_values = key if isinstance(key, tuple) else (key,)
        result = dict(zip(group_columns, key_values))
        actual_class = next(iter(unique))
        result.update({
            f"{unit_name}_count": int(len(group)),
            "actual_class": "MG" if actual_class == 1 else "HG",
            "positive_prediction_count": int(group[prediction_column].astype(int).sum()),
        })
        if actual_class == 1:
            result["mg_detection_rate"] = float(group[prediction_column].astype(int).mean())
            result["false_positive_rate"] = np.nan
        else:
            result["mg_detection_rate"] = np.nan
            result["false_positive_rate"] = float(group[prediction_column].astype(int).mean())
        results.append(result)
    return pd.DataFrame(results)
