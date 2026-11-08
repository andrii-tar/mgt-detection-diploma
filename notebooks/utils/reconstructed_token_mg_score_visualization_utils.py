from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb

from notebooks.utils.classification_experiment_utils import discover_model_artifacts
from notebooks.utils.feature_aggregation_utils import (
    FeatureSet,
    load_prepared_partition,
    prepared_partition_path,
)
from notebooks.utils.model_training_utils import (
    FittedFeatureVectorModel,
    load_model_artifact,
    load_model_metadata,
    score_feature_vector_model,
)
from notebooks.utils.paper_model_comparison_utils import SIMPLE_SIMILARITY_COLUMNS
from notebooks.utils.semeval_boundary_utils import gold_token_boundary
from notebooks.utils.semeval_boundary_utils import predict_boundary_positions


MODEL_FAMILIES = ("logistic_regression", "random_forest")
FEATURE_SET = FeatureSet("simple_similarity", SIMPLE_SIMILARITY_COLUMNS)
WINDOW_CONFIGURATION = 64
STRIDE = 16
TRAINING_REGIME = "Kaggle+MAGE+Outfox"


def load_visualization_models(
    data_root: Path | str,
    *,
    run_id: str,
    scorer_llm: str = "littlelamb",
    training_regime: str = TRAINING_REGIME,
) -> dict[str, FittedFeatureVectorModel]:
    """Load and validate the saved LR and RF 64-token classifier artifacts."""
    models_root = Path(data_root) / "experiment-results" / "classification" / "models"
    models: dict[str, FittedFeatureVectorModel] = {}
    for model_family in MODEL_FAMILIES:
        matches = discover_model_artifacts(
            models_root,
            filters={
                "run_id": run_id,
                "model_family": model_family,
                "scorer_llm": scorer_llm,
                "feature_set": FEATURE_SET.name,
                "training_regime": training_regime,
                "training_window": str(WINDOW_CONFIGURATION),
            },
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {model_family} artifact for {scorer_llm}/64/{run_id}, found {len(matches)}"
            )
        expected = {
            "run_id": run_id,
            "model_family": model_family,
            "scorer_llm": scorer_llm,
            "feature_set": FEATURE_SET.name,
            "training_regime": training_regime,
            "training_window": str(WINDOW_CONFIGURATION),
        }
        metadata = load_model_metadata(matches[0], expected=expected)
        if metadata["feature_columns"] != list(SIMPLE_SIMILARITY_COLUMNS):
            raise ValueError(f"artifact feature schema mismatch for {model_family}: {matches[0]}")
        models[model_family] = load_model_artifact(
            matches[0],
            expected={**expected, "feature_columns": list(SIMPLE_SIMILARITY_COLUMNS)},
            trust_artifact=True,
        )
    return models


def score_prepared_features(
    prepared: pd.DataFrame,
    model: FittedFeatureVectorModel,
    *,
    split: str = "MG",
) -> pd.DataFrame:
    """Score prepared windows as MG probabilities without fitting a model."""
    required = {"split", "text_id", "label", "start_token", "stop_token"}
    missing = required.difference(prepared.columns)
    missing.update(set(model.feature_columns).difference(prepared.columns))
    if missing:
        raise ValueError(f"prepared features are missing columns: {sorted(missing)}")
    if prepared.empty:
        raise ValueError("prepared feature frame is empty")
    expected_label = {"MG": 1, "HG": 0}.get(split)
    if expected_label is None:
        raise ValueError("visualization split must be 'MG' or 'HG'")
    if not prepared["split"].astype(str).eq(split).all() or not prepared["label"].eq(expected_label).all():
        raise ValueError(f"visualization inputs must contain only {split} rows with label={expected_label}")
    scored = score_feature_vector_model(model, prepared)
    if not np.isfinite(scored["mg_score"]).all() or not scored["mg_score"].between(0.0, 1.0).all():
        raise ValueError("classifier returned invalid MG probabilities")
    return scored


def score_prepared_partition(
    data_root: Path | str,
    *,
    dataset: str,
    split: str,
    model: FittedFeatureVectorModel,
    scorer_llm: str = "littlelamb",
) -> pd.DataFrame:
    """Load a previously prepared 64-token partition and score its windows."""
    if split not in {"MG", "HG"}:
        raise ValueError("visualization prepared partition must use split='MG' or split='HG'")
    if list(model.feature_columns) != list(SIMPLE_SIMILARITY_COLUMNS):
        raise ValueError("classifier artifact does not match the simple_similarity feature schema")
    partition = prepared_partition_path(
        data_root,
        scorer_llm=scorer_llm,
        dataset=dataset,
        split=split,
        feature_set=FEATURE_SET,
        window_configuration=WINDOW_CONFIGURATION,
    )
    expected = {
        "schema_version": 1,
        "scorer_llm": scorer_llm,
        "dataset": dataset,
        "split": split,
        "feature_set": FEATURE_SET.name,
        "feature_columns": list(SIMPLE_SIMILARITY_COLUMNS),
        "window_configuration": str(WINDOW_CONFIGURATION),
        "stride": STRIDE,
    }
    prepared = load_prepared_partition(partition, expected=expected)
    scored = score_prepared_features(prepared, model, split=split)
    scored["text_id"] = scored["text_id"].astype(str).str.strip()
    scored["model"] = model.model_family
    return scored


def load_text_records(
    data_root: Path | str,
    *,
    dataset: str,
    scorer_llm: str = "littlelamb",
) -> pd.DataFrame:
    """Join raw text and annotations to saved Stage-1 token surfaces by ID."""
    root = Path(data_root)
    if dataset == "name_that_llm_train":
        source_path = root / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
        source = pd.read_csv(source_path, usecols=["id", "text", "source_llm"], dtype={"id": str})
        dataset_kind = "name_that_llm"
    elif dataset == "bbc_news_content":
        source_path = root / "raw" / "bbc_news" / "bbc_news_content_HG.csv"
        source = pd.read_csv(source_path, usecols=["id", "text", "origin"], dtype={"id": str})
        if not source["origin"].astype(str).str.strip().str.upper().eq("HG").all():
            raise ValueError("BBC visualization source must contain only origin='HG' rows")
        dataset_kind = "bbc"
    elif dataset == "semeval2024_task8_subtask_c_train":
        source_path = (
            root
            / "raw"
            / "semevals_2024-task8"
            / "SemEval2024-task8"
            / "subtaskC"
            / "SemEval2024_task8_C_train_MG.csv"
        )
        source = pd.read_csv(
            source_path,
            usecols=["id", "text", "boundary_word_index"],
            dtype={"id": str},
        )
        dataset_kind = "semeval"
    else:
        raise ValueError(f"unsupported visualization dataset: {dataset!r}")

    split = "HG" if dataset == "bbc_news_content" else "MG"
    token_paths = sorted(
        (root / "processed" / f"{dataset}__{scorer_llm}" / split / "data").glob("*.parquet")
    )
    if not token_paths:
        raise FileNotFoundError(f"no Stage-1 token data found for {dataset}/{scorer_llm}/{split}")
    token_frames = [pd.read_parquet(path, columns=["id", "tokens"]) for path in token_paths]
    token_records = pd.concat(token_frames, ignore_index=True)
    token_ids = set(token_records["id"].astype(str).str.strip())
    source_ids = source["id"].astype(str).str.strip()
    if not token_ids.issubset(set(source_ids)):
        missing_source = sorted(token_ids - set(source_ids))
        raise KeyError(f"raw source texts are missing for token IDs: {missing_source[:10]}")
    source = source.loc[source_ids.isin(token_ids)].copy()
    return align_text_records(source, token_records, dataset=dataset_kind)


def align_text_records(
    source_texts: pd.DataFrame,
    token_records: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Join passage metadata and Stage-1 token surfaces with strict ID coverage."""
    if dataset not in {"semeval", "name_that_llm", "bbc"}:
        raise ValueError(f"unsupported text record kind: {dataset!r}")
    source_required = {"id", "text"}
    if dataset == "semeval":
        source_required.add("boundary_word_index")
    elif dataset == "name_that_llm":
        source_required.add("source_llm")
    else:
        source_required.add("origin")
    missing_source = source_required.difference(source_texts.columns)
    missing_tokens = {"id", "tokens"}.difference(token_records.columns)
    if missing_source or missing_tokens:
        raise ValueError(
            f"missing source columns={sorted(missing_source)}, token columns={sorted(missing_tokens)}"
        )

    source = source_texts.copy()
    tokens = token_records.copy()
    source["text_id"] = source["id"].astype(str).str.strip()
    tokens["text_id"] = tokens["id"].astype(str).str.strip()
    if source["text_id"].eq("").any() or tokens["text_id"].eq("").any():
        raise ValueError("text IDs cannot be empty")
    if source["text_id"].duplicated().any():
        raise ValueError("duplicate source text IDs")
    if tokens["text_id"].duplicated().any():
        raise ValueError("duplicate token IDs")
    if source["text"].isna().any() or source["text"].astype(str).str.strip().eq("").any():
        raise ValueError("source passage text cannot be blank")

    source_ids, token_ids = set(source["text_id"]), set(tokens["text_id"])
    missing_ids = sorted(source_ids - token_ids)
    if missing_ids:
        raise KeyError(f"Stage-1 token rows are missing for text IDs: {missing_ids[:10]}")
    merged = source.merge(tokens[["text_id", "tokens"]], on="text_id", how="inner", validate="one_to_one")
    if merged["tokens"].map(lambda value: not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0).any():
        raise ValueError("each passage must have a non-empty Stage-1 token list")
    merged["tokens"] = merged["tokens"].map(
        lambda value: [str(token["token"]) if isinstance(token, Mapping) else str(token) for token in value]
    )
    merged["token_count"] = merged["tokens"].map(len).astype("int64")
    keep = ["text_id", "text", "tokens", "token_count"]
    if dataset == "semeval":
        merged["boundary_word_index"] = pd.to_numeric(merged["boundary_word_index"], errors="raise").astype("int64")
        keep.append("boundary_word_index")
    elif dataset == "name_that_llm":
        if merged["source_llm"].isna().any() or merged["source_llm"].astype(str).str.strip().eq("").any():
            raise ValueError("Name That LLM source identity cannot be blank")
        keep.append("source_llm")
    else:
        if not merged["origin"].astype(str).str.strip().str.upper().eq("HG").all():
            raise ValueError("BBC visualization records must contain only origin='HG' rows")
    return merged[keep].sort_values("text_id", kind="stable").reset_index(drop=True)


def semeval_boundary_errors_from_results(
    predictions: pd.DataFrame,
    text_records: pd.DataFrame,
    *,
    artifact_run_id: str,
    scorer_llm: str = "littlelamb",
    feature_set: str = FEATURE_SET.name,
    training_regime: str = TRAINING_REGIME,
    training_window: int = WINDOW_CONFIGURATION,
    evaluation_window: int = WINDOW_CONFIGURATION,
) -> pd.DataFrame:
    """Compute per-text localization errors from saved SemEval boundary results."""
    required = {
        "text_id", "model", "window_length", "training_window",
        "predicted_token_boundary", "scorer_llm", "feature_set", "training_regime",
        "evaluation_dataset", "semeval_split", "artifact_run_id",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"SemEval boundary results are missing columns: {sorted(missing)}")
    record_required = {"text_id", "text", "boundary_word_index", "tokens"}
    missing_records = record_required.difference(text_records.columns)
    if missing_records:
        raise ValueError(f"SemEval text records are missing columns: {sorted(missing_records)}")
    selected = predictions.loc[
        predictions["scorer_llm"].astype(str).eq(scorer_llm)
        & predictions["feature_set"].astype(str).eq(feature_set)
        & predictions["training_regime"].astype(str).eq(training_regime)
        & predictions["training_window"].astype(str).eq(str(training_window))
        & predictions["window_length"].astype(str).eq(str(evaluation_window))
        & predictions["model"].astype(str).isin(MODEL_FAMILIES)
        & predictions["artifact_run_id"].astype(str).eq(artifact_run_id)
        & predictions["evaluation_dataset"].astype(str).eq("semeval2024_task8_subtask_c_train")
        & predictions["semeval_split"].astype(str).eq("train")
    ].copy()
    if selected.empty:
        raise ValueError("saved SemEval boundary results contain no rows for the selected model configuration")
    if selected.duplicated(["text_id", "model"]).any():
        raise ValueError("saved SemEval boundary results contain duplicate text/model rows")

    records = text_records.copy()
    records["text_id"] = records["text_id"].astype(str).str.strip()
    if records["text_id"].duplicated().any():
        raise ValueError("SemEval text records contain duplicate IDs")
    records_by_id = records.set_index("text_id")
    rows = []
    for prediction in selected.itertuples(index=False):
        text_id = str(prediction.text_id).strip()
        if text_id not in records_by_id.index:
            raise KeyError(f"SemEval source text is missing for boundary result {text_id}")
        record = records_by_id.loc[text_id]
        gold_boundary = gold_token_boundary(
            str(record["text"]), int(record["boundary_word_index"]), record["tokens"]
        )
        predicted_boundary = pd.to_numeric(pd.Series([prediction.predicted_token_boundary]), errors="coerce").iloc[0]
        if not np.isfinite(predicted_boundary):
            raise ValueError(f"SemEval predicted boundary is invalid for text_id={text_id}")
        rows.append(
            {
                "text_id": text_id,
                "model": str(prediction.model),
                "gold_token_boundary": int(gold_boundary),
                "predicted_token_boundary": int(predicted_boundary),
                "absolute_token_error": abs(int(predicted_boundary) - int(gold_boundary)),
            }
        )
    return pd.DataFrame(rows)


def semeval_boundary_errors_from_scored_windows(
    scored_windows: pd.DataFrame,
    text_records: pd.DataFrame,
    *,
    threshold: float = 0.5,
    smoothing_window: int = 5,
    sustained_length: int = 3,
) -> pd.DataFrame:
    """Compute gold-relative localization errors from the selected classifiers' scores."""
    required_windows = {"text_id", "model", "start_token", "stop_token", "mg_score"}
    missing_windows = required_windows.difference(scored_windows.columns)
    if missing_windows:
        raise ValueError(f"SemEval scored windows are missing columns: {sorted(missing_windows)}")
    required_records = {"text_id", "text", "boundary_word_index", "tokens"}
    missing_records = required_records.difference(text_records.columns)
    if missing_records:
        raise ValueError(f"SemEval text records are missing columns: {sorted(missing_records)}")
    if text_records["text_id"].astype(str).str.strip().duplicated().any():
        raise ValueError("SemEval text records contain duplicate IDs")

    windows = scored_windows.loc[scored_windows["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if windows.empty:
        raise ValueError("no SemEval scored windows for the selected classifier models")
    windows["text_id"] = pd.to_numeric(windows["text_id"], errors="raise").astype("int64")
    windows["window_length"] = WINDOW_CONFIGURATION
    records = text_records.copy()
    records["text_id"] = pd.to_numeric(records["text_id"], errors="raise").astype("int64")
    token_counts = dict(zip(records["text_id"], records["tokens"].map(len).astype("int64"), strict=True))
    predictions = predict_boundary_positions(
        windows,
        token_counts,
        threshold=threshold,
        smoothing_window=smoothing_window,
        sustained_length=sustained_length,
    )
    records_by_id = records.set_index("text_id")
    rows = []
    exclusions = []
    for prediction in predictions.itertuples(index=False):
        text_id = int(prediction.text_id)
        if text_id not in records_by_id.index:
            raise KeyError(f"SemEval source text is missing for scored text {text_id}")
        record = records_by_id.loc[text_id]
        try:
            gold = gold_token_boundary(
                str(record["text"]), int(record["boundary_word_index"]), record["tokens"]
            )
        except ValueError as error:
            allowed_reasons = (
                "token labels do not reconstruct the annotated text",
                "word boundary does not map to a decoded token",
                "boundary_word_index is outside the text word range",
            )
            if not str(error).startswith(allowed_reasons):
                raise
            exclusions.append(
                {"text_id": str(text_id), "model": str(prediction.model), "exclusion_reason": str(error)}
            )
            continue
        predicted = int(prediction.predicted_token_boundary)
        rows.append(
            {
                "text_id": str(text_id),
                "model": str(prediction.model),
                "gold_token_boundary": int(gold),
                "predicted_token_boundary": predicted,
                "absolute_token_error": abs(predicted - int(gold)),
                "no_crossing": bool(prediction.no_crossing),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["exclusions"] = pd.DataFrame(
        exclusions, columns=["text_id", "model", "exclusion_reason"]
    )
    return result


def detect_window_run_boundaries(
    scored_windows: pd.DataFrame,
    *,
    threshold: float = 0.5,
    run_length: int = 3,
) -> pd.DataFrame:
    """Detect the first sustained HG-to-MG transition from ordered window scores."""
    required = {"model", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"scored windows are missing columns: {sorted(missing)}")
    if scored_windows.empty:
        raise ValueError("scored windows cannot be empty")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if isinstance(run_length, (bool, np.bool_)) or not isinstance(run_length, (int, np.integer)) or run_length <= 0:
        raise ValueError("run_length must be a positive integer")

    windows = scored_windows.loc[:, ["model", "start_token", "stop_token", "mg_score"]].copy()
    if windows["model"].isna().any() or windows["model"].astype(str).str.strip().eq("").any():
        raise ValueError("scored windows must have non-empty model identifiers")
    for column in ("start_token", "stop_token", "mg_score"):
        windows[column] = pd.to_numeric(windows[column], errors="coerce")
    valid_spans = (
        np.isfinite(windows["start_token"])
        & np.isfinite(windows["stop_token"])
        & windows["start_token"].eq(np.floor(windows["start_token"]))
        & windows["stop_token"].eq(np.floor(windows["stop_token"]))
        & windows["start_token"].ge(0)
        & windows["stop_token"].gt(windows["start_token"])
    )
    if not valid_spans.all():
        raise ValueError("scored windows contain an invalid scoring span")
    valid_scores = np.isfinite(windows["mg_score"]) & windows["mg_score"].between(0.0, 1.0)
    if not valid_scores.all():
        raise ValueError("scored windows contain a non-finite or out-of-range MG score")
    if windows.duplicated(["model", "start_token"]).any():
        raise ValueError("each model must have one score per window start")

    results: list[dict[str, object]] = []
    for model, group in windows.groupby("model", sort=False):
        ordered = group.sort_values(["start_token", "stop_token"], kind="stable")
        starts = ordered["start_token"].to_numpy(dtype="float64")
        stops = ordered["stop_token"].to_numpy(dtype="float64")
        scores = ordered["mg_score"].to_numpy(dtype="float64")
        centers = (starts + stops) / 2.0
        hg = scores < float(threshold)
        predicted_boundary = np.nan
        for first_mg_index in range(int(run_length), len(scores) - int(run_length) + 1):
            previous_hg = hg[first_mg_index - int(run_length) : first_mg_index]
            following_mg = ~hg[first_mg_index : first_mg_index + int(run_length)]
            if previous_hg.all() and following_mg.all():
                predicted_boundary = float((centers[first_mg_index - 1] + centers[first_mg_index]) / 2.0)
                break
        results.append(
            {
                "model": str(model),
                "status": "found" if np.isfinite(predicted_boundary) else "not_found",
                "predicted_token_boundary": predicted_boundary,
            }
        )
    return pd.DataFrame(results, columns=["model", "status", "predicted_token_boundary"])


def select_visualization_examples(
    token_scores: pd.DataFrame,
    *,
    semeval_boundary_errors: pd.DataFrame,
    manual_ids: Mapping[str, Sequence[object]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select manual examples and deterministic low/middle/high/error cases."""
    required = {"dataset", "text_id", "model", "token_position", "mg_score"}
    missing = required.difference(token_scores.columns)
    if missing:
        raise ValueError(f"token scores are missing columns: {sorted(missing)}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    scores = token_scores.copy()
    scores["text_id"] = scores["text_id"].astype(str).str.strip()
    scores["dataset"] = scores["dataset"].replace(
        {"semeval2024_task8_subtask_c_train": "semeval"}
    )
    scores = scores.loc[scores["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if scores.empty:
        raise ValueError("no supported classifier token scores are available")

    result_rows: list[dict[str, object]] = []

    def add(dataset: str, text_id: object, group: str, reason: str, score: float | None = None) -> None:
        result_rows.append(
            {
                "dataset": dataset,
                "text_id": str(text_id),
                "selection_group": group,
                "selection_reason": reason,
                "selection_seed": int(seed),
                "selection_score": np.nan if score is None else float(score),
            }
        )

    available_by_dataset = {
        dataset: set(group.text_id.astype(str))
        for dataset, group in scores.groupby("dataset", sort=False)
    }
    for dataset, requested_ids in (manual_ids or {}).items():
        normalized_dataset = "semeval" if dataset == "semeval2024_task8_subtask_c_train" else dataset
        available = available_by_dataset.get(normalized_dataset, set())
        for text_id in requested_ids:
            normalized_id = str(text_id).strip()
            if normalized_id not in available:
                raise KeyError(f"manual text ID {normalized_id!r} is unavailable for {normalized_dataset}")
            add(normalized_dataset, normalized_id, "manual", "configured text ID")

    semeval = scores.loc[scores["dataset"].eq("semeval")]
    boundary = semeval_boundary_errors.copy()
    if not boundary.empty and "text_id" in boundary:
        boundary["text_id"] = boundary["text_id"].astype(str).str.strip()
    boundary_required = {"text_id", "model", "absolute_token_error"}
    if not boundary_required.issubset(boundary.columns):
        raise ValueError(f"SemEval boundary errors are missing columns: {sorted(boundary_required - set(boundary.columns))}")
    boundary = boundary.loc[boundary["model"].astype(str).isin(MODEL_FAMILIES)]
    semeval_scores = (
        boundary.loc[boundary["text_id"].isin(set(semeval["text_id"]))]
        .pivot_table(index="text_id", columns="model", values="absolute_token_error", aggfunc="first")
        .dropna(subset=list(MODEL_FAMILIES))
    )
    if len(semeval_scores) < 3:
        raise ValueError("at least three shared SemEval texts with both classifier boundary errors are required")
    semeval_scores["mean_error"] = semeval_scores[list(MODEL_FAMILIES)].mean(axis=1)
    semeval_ranked = semeval_scores["mean_error"].sort_values(kind="stable")
    median_error = float(semeval_ranked.median())
    selections = [
        ("semeval_good", str(semeval_ranked.index[0]), float(semeval_ranked.iloc[0]), "lowest mean absolute boundary error"),
        (
            "semeval_typical",
            str((semeval_ranked - median_error).abs().sort_values(kind="stable").index[0]),
            float(median_error),
            "mean absolute boundary error nearest the median",
        ),
        ("semeval_poor", str(semeval_ranked.index[-1]), float(semeval_ranked.iloc[-1]), "highest mean absolute boundary error"),
    ]
    used_semeval: set[str] = set()
    for group, text_id, value, reason in selections:
        if text_id in used_semeval:
            continue
        add("semeval", text_id, group, reason, value)
        used_semeval.add(text_id)

    name_that_llm = scores.loc[scores["dataset"].eq("name_that_llm_train")]
    model_means = name_that_llm.groupby(["text_id", "model"], sort=True)["mg_score"].mean().unstack("model")
    common = model_means.dropna(subset=list(MODEL_FAMILIES)).copy()
    if len(common) < 3:
        raise ValueError("at least three Name That LLM texts with both classifier scores are required")
    common["mean_score"] = common[list(MODEL_FAMILIES)].mean(axis=1)
    ranked = common["mean_score"].sort_values(kind="stable")
    middle_score = float(ranked.median())
    name_selections = [
        ("name_that_llm_low", str(ranked.index[0]), float(ranked.iloc[0]), "lowest mean reconstructed MG score"),
        (
            "name_that_llm_middle",
            str((ranked - middle_score).abs().sort_values(kind="stable").index[0]),
            float(middle_score),
            "mean reconstructed MG score nearest the median",
        ),
        ("name_that_llm_high", str(ranked.index[-1]), float(ranked.iloc[-1]), "highest mean reconstructed MG score"),
    ]
    for group, text_id, value, reason in name_selections:
        add("name_that_llm_train", text_id, group, reason, value)
    token_differences = (
        name_that_llm.pivot_table(
            index=["text_id", "token_position"], columns="model", values="mg_score", aggfunc="mean"
        )
        .dropna(subset=list(MODEL_FAMILIES))
        .assign(disagreement=lambda frame: (frame[MODEL_FAMILIES[0]] - frame[MODEL_FAMILIES[1]]).abs())
        .groupby(level="text_id")["disagreement"].mean()
    )
    if not token_differences.empty:
        disagreement_id = str(token_differences.sort_values(ascending=False, kind="stable").index[0])
        add(
            "name_that_llm_train",
            disagreement_id,
            "name_that_llm_disagreement",
            "largest mean absolute LR/RF token-score difference",
            float(token_differences.loc[disagreement_id]),
        )
    bbc = scores.loc[scores["dataset"].eq("bbc")]
    if not bbc.empty:
        bbc_means = bbc.groupby(["text_id", "model"], sort=True)["mg_score"].mean().unstack("model")
        bbc_common = bbc_means.dropna(subset=list(MODEL_FAMILIES)).copy()
        if len(bbc_common) < 3:
            raise ValueError("at least three BBC texts with both classifier scores are required")
        bbc_common["mean_score"] = bbc_common[list(MODEL_FAMILIES)].mean(axis=1)
        bbc_ranked = bbc_common["mean_score"].sort_values(kind="stable")
        bbc_median = float(bbc_ranked.median())
        for group, text_id, score, reason in [
            ("bbc_good", str(bbc_ranked.index[0]), float(bbc_ranked.iloc[0]), "lowest mean reconstructed MG score"),
            ("bbc_average", str((bbc_ranked - bbc_median).abs().sort_values(kind="stable").index[0]), bbc_median, "mean reconstructed MG score nearest the median"),
            ("bbc_bad", str(bbc_ranked.index[-1]), float(bbc_ranked.iloc[-1]), "highest mean reconstructed MG score"),
        ]:
            add("bbc", text_id, group, reason, score)
    return pd.DataFrame(result_rows)


def build_excerpt_rows(
    token_scores: pd.DataFrame,
    token_surfaces: Sequence[str],
    *,
    dataset: str,
    excerpt_width: int,
    gold_boundary: int | None = None,
    excerpt_start: int | None = None,
) -> pd.DataFrame:
    """Build a token-aligned excerpt centered on the dataset-specific point of interest."""
    if isinstance(excerpt_width, (bool, np.bool_)) or not isinstance(excerpt_width, (int, np.integer)) or excerpt_width <= 0:
        raise ValueError("excerpt_width must be a positive integer")
    if dataset not in {"semeval", "name_that_llm_train", "bbc"}:
        raise ValueError(f"unsupported visualization dataset: {dataset!r}")
    surfaces = [str(surface) for surface in token_surfaces]
    token_count = len(surfaces)
    if token_count == 0:
        raise ValueError("token_surfaces cannot be empty")
    required = {"model", "token_position", "mg_score"}
    missing = required.difference(token_scores.columns)
    if missing:
        raise ValueError(f"token scores are missing columns: {sorted(missing)}")
    current = token_scores.loc[token_scores["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if current.duplicated(["model", "token_position"]).any():
        raise ValueError("token scores contain duplicate model/token positions")
    positions = pd.to_numeric(current["token_position"], errors="coerce")
    if positions.isna().any() or not positions.eq(np.floor(positions)).all() or not positions.between(0, token_count - 1).all():
        raise ValueError("token positions are outside the token surface range")
    pivot = current.assign(token_position=positions.astype("int64")).pivot(
        index="token_position", columns="model", values="mg_score"
    ).reindex(range(token_count))
    missing_models = set(MODEL_FAMILIES).difference(pivot.columns)
    if missing_models:
        raise ValueError(f"token scores are missing classifier models: {sorted(missing_models)}")

    width = min(int(excerpt_width), token_count)
    if excerpt_start is not None:
        if isinstance(excerpt_start, (bool, np.bool_)) or not isinstance(excerpt_start, (int, np.integer)):
            raise ValueError("excerpt_start must be an integer")
        start = min(max(0, int(excerpt_start)), token_count - width)
    elif dataset == "semeval":
        if gold_boundary is None or isinstance(gold_boundary, (bool, np.bool_)) or not isinstance(gold_boundary, (int, np.integer)) or not 0 <= gold_boundary <= token_count:
            raise ValueError("SemEval excerpts require a valid gold_boundary")
        start = min(max(0, int(gold_boundary) - width // 2), token_count - width)
    elif dataset == "name_that_llm_train":
        differences = (pivot[MODEL_FAMILIES[0]] - pivot[MODEL_FAMILIES[1]]).abs()
        if differences.notna().any():
            center = int(differences.idxmax())
        else:
            center = token_count // 2
        start = min(max(0, center - width // 2), token_count - width)
    else:
        center_scores = pivot[list(MODEL_FAMILIES)].mean(axis=1)
        center = int(center_scores.idxmax()) if center_scores.notna().any() else token_count // 2
        start = min(max(0, center - width // 2), token_count - width)
    stop = start + width
    result = pd.DataFrame(
        {
            "token_position": np.arange(start, stop),
            "token_surface": surfaces[start:stop],
            "logistic_regression": pivot.loc[start : stop - 1, "logistic_regression"].to_numpy(),
            "random_forest": pivot.loc[start : stop - 1, "random_forest"].to_numpy(),
        }
    )
    if dataset == "semeval":
        result["is_gold_boundary"] = result["token_position"].eq(int(gold_boundary))
    return result


def _format_token_surface(value: object) -> str:
    surface = str(value)
    if surface.startswith(("Ġ", "▁")):
        surface = surface[1:]
    surface = surface.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "↵").replace("ĉ", "⇥")
    return surface or "∅"


def _text_color_for_background(color: tuple[float, float, float, float]) -> str:
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.52 else "white"


def plot_passage_scores(
    example: Mapping[str, object],
    token_scores: pd.DataFrame,
    excerpt_rows: pd.DataFrame,
    *,
    title: str,
) -> plt.Figure:
    """Render full-text score curves and a wrapped, score-colored token excerpt."""
    required_scores = {"model", "token_position", "mg_score"}
    missing_scores = required_scores.difference(token_scores.columns)
    if missing_scores:
        raise ValueError(f"token scores are missing columns: {sorted(missing_scores)}")
    required_excerpt = {"token_position", "token_surface", *MODEL_FAMILIES}
    missing_excerpt = required_excerpt.difference(excerpt_rows.columns)
    if missing_excerpt:
        raise ValueError(f"excerpt rows are missing columns: {sorted(missing_excerpt)}")
    dataset = str(example.get("dataset", ""))
    if dataset not in {"semeval", "semeval2024_task8_subtask_c_train", "name_that_llm_train", "bbc"}:
        raise ValueError(f"unsupported visualization dataset: {dataset!r}")
    scores = token_scores.loc[token_scores["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if scores.empty or scores.duplicated(["model", "token_position"]).any():
        raise ValueError("token score rows must contain unique positions for both classifiers")
    scores["token_position"] = pd.to_numeric(scores["token_position"], errors="coerce")
    scores["mg_score"] = pd.to_numeric(scores["mg_score"], errors="coerce")
    if scores["token_position"].isna().any() or not scores["token_position"].eq(np.floor(scores["token_position"])).all():
        raise ValueError("token positions must be integers")
    finite_scores = scores["mg_score"].dropna()
    if not np.isfinite(finite_scores).all() or not finite_scores.between(0.0, 1.0).all():
        raise ValueError("finite token MG scores must be within [0, 1]")
    if not set(MODEL_FAMILIES).issubset(set(scores["model"].astype(str))):
        raise ValueError("full-text profile requires both classifier models")
    excerpts = excerpt_rows.sort_values("token_position", kind="stable").reset_index(drop=True)
    excerpt_positions = pd.to_numeric(excerpts["token_position"], errors="coerce")
    if excerpt_positions.isna().any() or not excerpt_positions.eq(np.floor(excerpt_positions)).all():
        raise ValueError("excerpt token positions must be integers")
    if excerpts.empty:
        raise ValueError("excerpt rows cannot be empty")
    if not np.isin(excerpt_positions.astype("int64"), scores["token_position"].astype("int64")).all():
        raise ValueError("excerpt positions are missing from the full token score profile")

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1, 1.8])
    profile_axis = figure.add_subplot(grid[0])
    text_axis = figure.add_subplot(grid[1])
    colors = {"logistic_regression": "#2369a1", "random_forest": "#d97706"}
    for model in MODEL_FAMILIES:
        current = scores.loc[scores["model"].astype(str).eq(model)].sort_values("token_position")
        profile_axis.plot(
            current["token_position"],
            current["mg_score"],
            color=colors[model],
            linewidth=1.5,
            label="Logistic Regression" if model == "logistic_regression" else "Random Forest",
        )
    token_count = int(scores["token_position"].max()) + 1
    excerpt_start = int(excerpt_positions.min())
    excerpt_stop = int(excerpt_positions.max()) + 1
    profile_axis.axvspan(excerpt_start, excerpt_stop, color="0.3", alpha=0.1, label="text excerpt")
    if dataset.startswith("semeval"):
        gold_boundary = example.get("gold_token_boundary")
        if gold_boundary is None and "is_gold_boundary" in excerpts:
            marked_positions = excerpts.loc[excerpts["is_gold_boundary"].astype(bool), "token_position"]
            if not marked_positions.empty:
                gold_boundary = int(marked_positions.iloc[0])
        if gold_boundary is not None:
            profile_axis.axvline(
                int(gold_boundary), color="#111827", linestyle="--", linewidth=1.3,
                label="gold HG/MG boundary",
            )
    profile_axis.set_xlim(0, max(1, token_count - 1))
    profile_axis.set_ylim(0.0, 1.0)
    profile_axis.set_xlabel("Token position")
    profile_axis.set_ylabel("Reconstructed MG score")
    profile_axis.grid(axis="y", alpha=0.25)
    profile_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    token_labels = [_format_token_surface(value) for value in excerpts["token_surface"]]
    max_chars_per_line = 82
    chunks: list[list[int]] = []
    chunk: list[int] = []
    current_length = 0
    for index, label in enumerate(token_labels):
        token_length = min(len(label), max_chars_per_line)
        if chunk and current_length + token_length + 1 > max_chars_per_line:
            chunks.append(chunk)
            chunk = []
            current_length = 0
        chunk.append(index)
        current_length += token_length + 1
    if chunk:
        chunks.append(chunk)

    line_count = len(chunks)
    row_gap = line_count + 1
    boundary_markers: list[tuple[float, float]] = []
    for model_index, model in enumerate(MODEL_FAMILIES):
        model_label = "Logistic Regression" if model == "logistic_regression" else "Random Forest"
        text_axis.text(
            -2.0,
            model_index * row_gap + (line_count - 1) / 2,
            model_label,
            ha="right",
            va="center",
            fontsize=9,
        )
        for line_index, indices in enumerate(chunks):
            y = model_index * row_gap + line_index
            x = 0.0
            for index in indices:
                label = token_labels[index]
                box_width = min(max(len(label), 1), max_chars_per_line)
                score = pd.to_numeric(pd.Series([excerpts.iloc[index][model]]), errors="coerce").iloc[0]
                color = "#d9d9d9" if pd.isna(score) else cmap(norm(float(score)))
                text_axis.text(
                    x,
                    y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=9,
                    color=_text_color_for_background(color),
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "edgecolor": "white", "linewidth": 0.4},
                )
                if dataset.startswith("semeval") and bool(excerpts.iloc[index].get("is_gold_boundary", False)):
                    boundary_markers.append((x - 0.2, y))
                x += box_width + 1.2
    for boundary_x, boundary_y in boundary_markers:
        text_axis.plot(
            [boundary_x, boundary_x],
            [boundary_y - 0.45, boundary_y + 0.45],
            color="#111827",
            linestyle="--",
            linewidth=1.3,
        )
    text_axis.set_xlim(-24, max_chars_per_line * 1.2)
    text_axis.set_ylim(row_gap * len(MODEL_FAMILIES) - 1, -1)
    text_axis.set_yticks([])
    text_axis.set_xticks([])
    excerpt_title = "Token surfaces colored by reconstructed MG score"
    if dataset.startswith("semeval"):
        excerpt_title += " · dashed line marks gold HG/MG boundary"
    text_axis.set_title(excerpt_title)
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[profile_axis, text_axis], label="MG score", shrink=0.82)

    source_llm = example.get("source_llm")
    context = f"{dataset} · text {example.get('text_id')}"
    if source_llm is not None and not pd.isna(source_llm):
        context += f" · source LLM: {source_llm}"
    figure.suptitle(f"{title}\n{context}", fontsize=14)
    return figure


def plot_passage_score_dispersion(scores: pd.DataFrame, *, title: str) -> plt.Figure:
    """Plot token-level mean MG scores with population-standard-deviation bands."""
    required = {"model", "token_position", "mg_score_mean", "mg_score_std", "window_count"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"token score dispersion is missing columns: {sorted(missing)}")
    plotted = scores.loc[scores["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if plotted.empty or plotted.duplicated(["model", "token_position"]).any():
        raise ValueError("token score dispersion rows must contain unique positions for both classifiers")
    plotted["token_position"] = pd.to_numeric(plotted["token_position"], errors="coerce")
    plotted["mg_score_mean"] = pd.to_numeric(plotted["mg_score_mean"], errors="coerce")
    plotted["mg_score_std"] = pd.to_numeric(plotted["mg_score_std"], errors="coerce")
    plotted["window_count"] = pd.to_numeric(plotted["window_count"], errors="coerce")
    if plotted["token_position"].isna().any() or not plotted["token_position"].eq(np.floor(plotted["token_position"])).all():
        raise ValueError("token positions must be integers")
    if plotted["window_count"].isna().any() or not plotted["window_count"].eq(np.floor(plotted["window_count"])).all():
        raise ValueError("window counts must be non-negative integers")
    if plotted["window_count"].lt(0).any():
        raise ValueError("window counts must be non-negative integers")
    has_mean = plotted["mg_score_mean"].notna()
    has_std = plotted["mg_score_std"].notna()
    if not has_mean.eq(has_std).all():
        raise ValueError("token means and standard deviations must both be finite or both be missing")
    if (plotted.loc[has_mean, "window_count"] <= 0).any() or (plotted.loc[~has_mean, "window_count"] != 0).any():
        raise ValueError("window counts must be positive for scored positions and zero for gaps")
    finite_means = plotted.loc[has_mean, "mg_score_mean"]
    finite_stds = plotted.loc[has_std, "mg_score_std"]
    if not np.isfinite(finite_means).all() or not finite_means.between(0.0, 1.0).all():
        raise ValueError("finite token MG means must be within [0, 1]")
    if not np.isfinite(finite_stds).all() or finite_stds.lt(0.0).any():
        raise ValueError("finite token MG standard deviations must be non-negative")
    if not set(MODEL_FAMILIES).issubset(set(plotted["model"].astype(str))):
        raise ValueError("score dispersion profile requires both classifier models")

    colors = {"logistic_regression": "#2369a1", "random_forest": "#d97706"}
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
    figure, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    for model in MODEL_FAMILIES:
        current = plotted.loc[plotted["model"].astype(str).eq(model)].sort_values("token_position")
        positions = current["token_position"].to_numpy(dtype="int64")
        means = current["mg_score_mean"].to_numpy(dtype="float64")
        standard_deviations = current["mg_score_std"].to_numpy(dtype="float64")
        axis.plot(positions, means, color=colors[model], linewidth=1.5, label=labels[model])
        axis.fill_between(
            positions,
            np.clip(means - standard_deviations, 0.0, 1.0),
            np.clip(means + standard_deviations, 0.0, 1.0),
            where=np.isfinite(means) & np.isfinite(standard_deviations),
            color=colors[model],
            alpha=0.2,
            linewidth=0,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Token position")
    axis.set_ylabel("Reconstructed MG score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    return figure


def plot_window_boundary_profile(
    scored_windows: pd.DataFrame,
    boundary_predictions: pd.DataFrame,
    *,
    gold_boundary: int,
    threshold: float,
    title: str,
) -> plt.Figure:
    """Plot window MG scores and their gold/predicted boundary positions."""
    required_windows = {"model", "start_token", "stop_token", "mg_score"}
    missing_windows = required_windows.difference(scored_windows.columns)
    if missing_windows:
        raise ValueError(f"scored windows are missing columns: {sorted(missing_windows)}")
    required_predictions = {"model", "status", "predicted_token_boundary"}
    missing_predictions = required_predictions.difference(boundary_predictions.columns)
    if missing_predictions:
        raise ValueError(f"window boundary predictions are missing columns: {sorted(missing_predictions)}")
    if isinstance(gold_boundary, (bool, np.bool_)) or not isinstance(gold_boundary, (int, np.integer)) or gold_boundary < 0:
        raise ValueError("gold_boundary must be a non-negative integer")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    windows = scored_windows.loc[scored_windows["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    predictions = boundary_predictions.loc[boundary_predictions["model"].astype(str).isin(MODEL_FAMILIES)].copy()
    if windows.empty or predictions.empty:
        raise ValueError("window profile requires scored windows and boundary predictions")
    if predictions["model"].astype(str).duplicated().any():
        raise ValueError("window boundary predictions must contain one row per model")
    if not set(MODEL_FAMILIES).issubset(set(windows["model"].astype(str))):
        raise ValueError("window profile requires scores for both classifier models")
    if not set(MODEL_FAMILIES).issubset(set(predictions["model"].astype(str))):
        raise ValueError("window profile requires boundary predictions for both classifier models")
    if not predictions["status"].isin(("found", "not_found")).all():
        raise ValueError("window boundary status must be 'found' or 'not_found'")

    for column in ("start_token", "stop_token", "mg_score"):
        windows[column] = pd.to_numeric(windows[column], errors="coerce")
    valid = (
        np.isfinite(windows["start_token"])
        & np.isfinite(windows["stop_token"])
        & np.isfinite(windows["mg_score"])
        & windows["start_token"].eq(np.floor(windows["start_token"]))
        & windows["stop_token"].eq(np.floor(windows["stop_token"]))
        & windows["start_token"].ge(0)
        & windows["stop_token"].gt(windows["start_token"])
        & windows["mg_score"].between(0.0, 1.0)
    )
    if not valid.all():
        raise ValueError("scored windows contain invalid spans or MG scores")
    predictions["predicted_token_boundary"] = pd.to_numeric(
        predictions["predicted_token_boundary"], errors="coerce"
    )
    found = predictions["status"].eq("found")
    if not np.isfinite(predictions.loc[found, "predicted_token_boundary"]).all():
        raise ValueError("found window boundaries must have finite token positions")

    colors = {"logistic_regression": "#2369a1", "random_forest": "#d97706"}
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for model in MODEL_FAMILIES:
        current = windows.loc[windows["model"].astype(str).eq(model)].sort_values("start_token")
        centers = (current["start_token"].to_numpy(dtype="float64") + current["stop_token"].to_numpy(dtype="float64")) / 2.0
        axis.plot(
            centers,
            current["mg_score"].to_numpy(dtype="float64"),
            color=colors[model],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=labels[model],
        )
    axis.axhline(float(threshold), color="0.35", linestyle="--", linewidth=1.0, label=f"Decision threshold ({threshold:.2f})")
    axis.axvline(gold_boundary, color="#111827", linestyle="--", linewidth=1.2, label="Gold boundary")
    for model in MODEL_FAMILIES:
        prediction = predictions.loc[predictions["model"].astype(str).eq(model)].iloc[0]
        if prediction["status"] == "found":
            axis.axvline(
                float(prediction["predicted_token_boundary"]),
                color=colors[model],
                linestyle=":",
                linewidth=1.3,
                label=f"{labels[model]} boundary",
            )
    axis.set_xlim(0, float(windows["stop_token"].max()))
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Token position")
    axis.set_ylabel("Window MG score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    return figure


def save_visualization_outputs(
    figures: Mapping[str, plt.Figure],
    selection_metadata: pd.DataFrame,
    output_root: Path | str,
) -> list[Path]:
    """Atomically save PNG/SVG figures and selection metadata to a new run folder."""
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if not figures:
        raise ValueError("at least one figure is required")
    if selection_metadata.empty:
        raise ValueError("selection metadata cannot be empty")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True)
        for name, figure in figures.items():
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "figure"
            for extension in ("png", "svg"):
                path = temporary / f"{stem}.{extension}"
                figure.savefig(path, dpi=300, bbox_inches="tight")
        selection_metadata.to_csv(temporary / "selection_metadata.csv", index=False)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return sorted(destination.iterdir())


def reconstruct_token_scores(
    scored_windows: pd.DataFrame,
    *,
    text_id: object,
    token_count: int,
) -> pd.DataFrame:
    """Average window MG scores onto every covered token position for one text."""
    required = {"text_id", "model", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"scored_windows is missing columns: {', '.join(sorted(missing))}")
    if isinstance(token_count, (bool, np.bool_)) or not isinstance(token_count, (int, np.integer)) or token_count <= 0:
        raise ValueError("token_count must be a positive integer")

    selected = scored_windows.loc[scored_windows["text_id"].astype(str).eq(str(text_id))].copy()
    if selected.empty:
        raise ValueError(f"no scored windows found for text_id={text_id!r}")
    if selected["model"].isna().any() or selected["model"].astype(str).str.strip().eq("").any():
        raise ValueError(f"text_id={text_id!r} has an empty model identifier")

    starts = pd.to_numeric(selected["start_token"], errors="coerce")
    stops = pd.to_numeric(selected["stop_token"], errors="coerce")
    scores = pd.to_numeric(selected["mg_score"], errors="coerce")
    valid_spans = (
        starts.notna()
        & stops.notna()
        & starts.eq(np.floor(starts))
        & stops.eq(np.floor(stops))
        & starts.ge(0)
        & stops.gt(starts)
        & stops.le(token_count)
    )
    if not valid_spans.all():
        raise ValueError(f"text_id={text_id!r} has an invalid scoring span")
    valid_scores = scores.notna() & np.isfinite(scores) & scores.between(0.0, 1.0)
    if not valid_scores.all():
        raise ValueError(f"text_id={text_id!r} has a non-finite or out-of-range MG score")

    rows: list[dict[str, object]] = []
    for model, group in selected.assign(
        start_index=starts.astype("int64"),
        stop_index=stops.astype("int64"),
        score_value=scores.astype("float64"),
    ).groupby("model", sort=False):
        totals = np.zeros(token_count, dtype="float64")
        counts = np.zeros(token_count, dtype="int64")
        for row in group.itertuples(index=False):
            start = int(row.start_index)
            stop = int(row.stop_index)
            totals[start:stop] += float(row.score_value)
            counts[start:stop] += 1
        token_scores = np.full(token_count, np.nan, dtype="float64")
        covered = counts > 0
        token_scores[covered] = totals[covered] / counts[covered]
        rows.extend(
            {
                "text_id": text_id,
                "model": str(model),
                "token_position": position,
                "mg_score": float(token_scores[position]) if covered[position] else np.nan,
            }
            for position in range(token_count)
        )
    return pd.DataFrame(rows, columns=["text_id", "model", "token_position", "mg_score"])


def reconstruct_token_score_dispersion(
    scored_windows: pd.DataFrame,
    *,
    text_id: object,
    token_count: int,
) -> pd.DataFrame:
    """Project window scores onto covered tokens and summarize mean, population SD, and count."""
    # Reuse the existing validation contract for IDs, spans, models, and probability values.
    reconstruct_token_scores(scored_windows, text_id=text_id, token_count=token_count)
    selected = scored_windows.loc[scored_windows["text_id"].astype(str).eq(str(text_id))]
    rows: list[dict[str, object]] = []
    for model, group in selected.groupby("model", sort=False):
        scores_by_position: list[list[float]] = [[] for _ in range(token_count)]
        for row in group.itertuples(index=False):
            for position in range(int(row.start_token), int(row.stop_token)):
                scores_by_position[position].append(float(row.mg_score))
        for position, values in enumerate(scores_by_position):
            rows.append(
                {
                    "text_id": text_id,
                    "model": str(model),
                    "token_position": position,
                    "mg_score_mean": float(np.mean(values)) if values else np.nan,
                    "mg_score_std": float(np.std(values, ddof=0)) if values else np.nan,
                    "window_count": len(values),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["text_id", "model", "token_position", "mg_score_mean", "mg_score_std", "window_count"],
    )
