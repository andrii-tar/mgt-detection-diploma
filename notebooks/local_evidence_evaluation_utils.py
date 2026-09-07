from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from notebooks.classification_utils import evaluate_scores
from notebooks.local_evidence_models_utils import score_local_evidence_classifier
from notebooks.local_evidence_training_utils import LocalEvidenceModelBundle
from notebooks.local_evidence_utils import (
    aggregate_evidence_fraction,
    aggregate_maximum_window,
    load_local_evidence_cohort,
)
from notebooks.semeval_boundary_utils import (
    add_gold_boundaries,
    build_gold_mg_regions,
    evaluate_mg_regions,
    predict_boundary_positions,
    predict_mg_regions,
    summarize_boundary_predictions,
    summarize_mg_region_evaluations,
)
from notebooks.token_candidate_similarity_heatmaps_utils import load_token_labels


@dataclass
class LocalEvidenceDatasetEvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    scored_windows: pd.DataFrame
    coverage: pd.DataFrame


@dataclass
class LocalEvidenceSemEvalEvaluationResult:
    summary: pd.DataFrame
    evaluations: pd.DataFrame
    predicted_regions: pd.DataFrame
    scored_windows: pd.DataFrame
    boundary_summary: pd.DataFrame
    boundary_predictions: pd.DataFrame
    boundary_exclusions: pd.DataFrame


def combine_local_evidence_dataset_results(
    results: list[LocalEvidenceDatasetEvaluationResult] | tuple[LocalEvidenceDatasetEvaluationResult, ...],
    *,
    evaluation_regime: str,
) -> LocalEvidenceDatasetEvaluationResult:
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    scored_windows = pd.concat([result.scored_windows for result in results], ignore_index=True)
    coverage = pd.concat([result.coverage for result in results], ignore_index=True)

    rule_columns = [
        "model", "window_length", "aggregation", "window_threshold",
        "document_threshold", "score_direction",
    ]
    metric_rows: list[dict[str, object]] = []
    for rule_values, documents in predictions.groupby(rule_columns, dropna=False, sort=False):
        rule = dict(zip(rule_columns, rule_values, strict=True))
        direction = str(rule["score_direction"])
        decision_scores = documents["mg_score"] if direction == "higher_is_mg" else -documents["mg_score"]
        threshold = float(rule["document_threshold"])
        metric_rows.append({
            "evaluation_dataset": evaluation_regime,
            **rule,
            "evaluation_regime": evaluation_regime,
            **evaluate_scores(documents["label"], decision_scores, threshold),
        })

    for frame in (predictions, scored_windows, coverage):
        frame["evaluation_regime"] = evaluation_regime
    return LocalEvidenceDatasetEvaluationResult(
        pd.DataFrame(metric_rows), predictions, scored_windows, coverage,
    )


def _namespace(windows: pd.DataFrame, dataset: str) -> pd.DataFrame:
    result = windows.copy()
    result["text_id"] = result.apply(
        lambda row: f"{dataset}::{row['split']}::{int(row['text_id'])}", axis=1
    )
    return result


def _select_requested(windows: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    return windows.merge(cohort[["split", "text_id"]], on=["split", "text_id"], how="inner", validate="many_to_many")


def evaluate_local_evidence_datasets(
    model: LocalEvidenceModelBundle,
    processed_root: Path | str,
    datasets: list[str] | tuple[str, ...],
    text_ids: Mapping[str, pd.DataFrame],
    *,
    show_progress: bool = False,
) -> LocalEvidenceDatasetEvaluationResult:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    rules = model.decision_rules
    for dataset in datasets:
        windows = load_local_evidence_cohort(
            Path(processed_root), dataset, model.llm_name, text_ids[dataset],
            window_lengths=model.window_lengths, stride=model.stride, top_k=model.top_k,
            show_progress=show_progress,
        )
        windows = _select_requested(windows, text_ids[dataset])
        windows = _namespace(windows, dataset)
        coverage_rows.append({"evaluation_dataset": dataset, "texts": windows["text_id"].nunique(), "window_rows": len(windows)})
        for (classifier_name, window_length), fitted in model.classifiers.items():
            selected = windows.loc[windows["window_length"].eq(window_length)]
            scored = score_local_evidence_classifier(fitted, selected, model.feature_columns)
            scored_frames.append(scored.assign(evaluation_dataset=dataset))
            for rule in rules.loc[
                rules["model"].eq(classifier_name) & rules["window_length"].eq(window_length)
            ].to_dict("records"):
                if rule["aggregation"] == "maximum_window":
                    documents = aggregate_maximum_window(scored)
                else:
                    documents = aggregate_evidence_fraction(scored, float(rule["window_threshold"]))
                direction = str(rule["score_direction"])
                decision_scores = documents["mg_score"] if direction == "higher_is_mg" else -documents["mg_score"]
                threshold = float(rule["document_threshold"])
                predictions = documents.assign(
                    evaluation_dataset=dataset,
                    model=classifier_name,
                    aggregation=rule["aggregation"],
                    window_threshold=rule["window_threshold"],
                    document_threshold=threshold,
                    score_direction=direction,
                    prediction=(decision_scores >= threshold).astype("int64"),
                )
                prediction_frames.append(predictions)
                metrics = evaluate_scores(documents["label"], decision_scores, threshold)
                metric_rows.append({
                    "evaluation_dataset": dataset, "model": classifier_name,
                    "window_length": int(window_length), **rule, **metrics,
                })
    return LocalEvidenceDatasetEvaluationResult(
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame(),
        pd.DataFrame(coverage_rows),
    )


def evaluate_local_evidence_semeval(
    model: LocalEvidenceModelBundle,
    processed_root: Path | str,
    annotations: pd.DataFrame,
    *,
    dataset: str = "semeval2024_task8_subtask_c_dev",
    split: str = "MG",
    text_ids: list[int] | tuple[int, ...] | None = None,
    threshold: float = 0.5,
    smoothing_window: int = 5,
    minimum_region_length: int = 3,
    sustained_length: int = 3,
) -> LocalEvidenceSemEvalEvaluationResult:
    requested = annotations[["id"]].rename(columns={"id": "text_id"}).copy()
    if text_ids is not None:
        requested = requested.loc[requested["text_id"].isin(text_ids)]
    requested = requested.assign(split=split)
    token_labels = load_token_labels(Path(processed_root), dataset, model.llm_name, split)
    available_ids = set(token_labels).intersection(requested["text_id"])
    missing_ids = sorted(set(requested["text_id"]) - available_ids)
    exclusions = [
        {"text_id": int(text_id), "reason": "missing processed token labels"}
        for text_id in missing_ids
    ]
    available_requested = requested.loc[requested["text_id"].isin(available_ids)]
    available_annotations = annotations.loc[annotations["id"].isin(available_ids)].copy()
    empty = pd.DataFrame()
    if available_requested.empty:
        return LocalEvidenceSemEvalEvaluationResult(empty, empty, empty, empty, empty, empty, pd.DataFrame(exclusions))
    windows = load_local_evidence_cohort(
        Path(processed_root), dataset, model.llm_name, available_requested,
        window_lengths=model.window_lengths, stride=model.stride, top_k=model.top_k,
    )
    windows = _select_requested(windows, available_requested)
    token_counts = {text_id: len(labels) for text_id, labels in token_labels.items()}
    scored_frames = []
    for (classifier_name, window_length), fitted in model.classifiers.items():
        selected = windows.loc[windows["window_length"].eq(window_length)]
        if selected.empty:
            continue
        scored_frames.append(score_local_evidence_classifier(fitted, selected, model.feature_columns))
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if scored.empty:
        return LocalEvidenceSemEvalEvaluationResult(empty, empty, empty, empty, empty, empty, pd.DataFrame(exclusions))
    boundary_candidates = predict_boundary_positions(
        scored, token_counts, threshold=threshold, smoothing_window=smoothing_window,
        sustained_length=sustained_length,
    )
    valid_ids: set[int] = set()
    boundary_frames: list[pd.DataFrame] = []
    annotation_by_id = available_annotations.set_index("id")
    for text_id, predictions_for_text in boundary_candidates.groupby("text_id", sort=True):
        annotation = annotation_by_id.loc[[text_id]].reset_index()
        try:
            labeled = add_gold_boundaries(predictions_for_text, annotation, token_labels)
        except ValueError as error:
            reason = str(error)
            if not reason.startswith((
                "token labels do not reconstruct the annotated text",
                "word boundary does not map to a decoded token",
                "boundary_word_index is outside the text word range",
            )):
                raise
            exclusions.append({"text_id": int(text_id), "reason": str(error)})
            continue
        valid_ids.add(int(text_id))
        boundary_frames.append(labeled)
    boundary_predictions = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else empty
    scored = scored.loc[scored["text_id"].isin(valid_ids)].copy()
    predicted = predict_mg_regions(
        scored, token_counts, threshold=threshold,
        smoothing_window=smoothing_window, minimum_region_length=minimum_region_length,
    ) if not scored.empty else empty
    gold = build_gold_mg_regions(available_annotations.loc[available_annotations["id"].isin(valid_ids)], token_labels)
    evaluations = evaluate_mg_regions(predicted, gold) if not predicted.empty else empty
    return LocalEvidenceSemEvalEvaluationResult(
        summarize_mg_region_evaluations(evaluations) if not evaluations.empty else empty,
        evaluations, predicted, scored,
        summarize_boundary_predictions(boundary_predictions) if not boundary_predictions.empty else empty,
        boundary_predictions, pd.DataFrame(exclusions),
    )
