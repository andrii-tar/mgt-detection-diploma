from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from notebooks.classification_utils import evaluate_scores
from notebooks.local_evidence_models_utils import (
    LOCAL_FEATURE_COLUMNS,
    MODEL_NAMES,
    BayesianConfig,
    GaussianMixtureNBConfig,
    FittedLocalEvidenceClassifier,
    fit_local_evidence_classifier,
    score_local_evidence_classifier,
    training_input_fingerprint,
)
from notebooks.local_evidence_utils import (
    _fit_document_threshold,
    _select_evidence_parameters,
    aggregate_evidence_fraction,
    aggregate_maximum_window,
    load_local_evidence_cohort,
)


@dataclass
class LocalEvidenceModelBundle:
    llm_name: str
    classifiers: dict[tuple[str, int], FittedLocalEvidenceClassifier]
    decision_rules: pd.DataFrame
    feature_columns: tuple[str, ...]
    window_lengths: tuple[int, ...]
    stride: int
    top_k: int
    training_datasets: tuple[str, ...]


@dataclass
class LocalEvidenceTrainingResult:
    model: LocalEvidenceModelBundle
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    training_input_fingerprint: str


def _cohort_key(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return {(str(row.split), int(row.text_id)) for row in frame.itertuples(index=False)}


def _namespace(windows: pd.DataFrame, dataset: str) -> pd.DataFrame:
    result = windows.copy()
    result["text_id"] = result.apply(
        lambda row: f"{dataset}::{row['split']}::{int(row['text_id'])}", axis=1
    )
    return result


def _select_requested(windows: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    return windows.merge(cohort[["split", "text_id"]], on=["split", "text_id"], how="inner", validate="many_to_many")


def _load_cohorts(
    processed_root: Path,
    datasets: list[str],
    llm_name: str,
    cohorts: Mapping[str, pd.DataFrame],
    window_lengths: tuple[int, ...],
    stride: int,
    top_k: int,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for dataset in datasets:
        cohort = cohorts.get(dataset)
        if cohort is None:
            raise ValueError(f"missing cohort for dataset={dataset!r}")
        windows = load_local_evidence_cohort(
            processed_root, dataset, llm_name, cohort,
            window_lengths=window_lengths, stride=stride, top_k=top_k,
        )
        windows = _select_requested(windows, cohort)
        missing_features = [feature for feature in feature_columns if feature not in windows]
        if missing_features:
            raise ValueError(f"missing feature columns: {missing_features}")
        namespaced = _namespace(windows, dataset)
        frames.append(namespaced)
        coverage.append({
            "dataset": dataset,
            "texts": namespaced["text_id"].nunique(),
            "window_rows": len(namespaced),
        })
    if not frames:
        raise ValueError("datasets must not be empty")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage)


def _score_metrics(
    scored: pd.DataFrame,
    model_name: str,
    window_length: int,
    partition: str,
    rules: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    selected = scored.loc[scored["window_length"].eq(window_length)]
    maximum = aggregate_maximum_window(selected)
    for rule in rules:
        if rule["aggregation"] == "maximum_window":
            document = maximum
        else:
            document = aggregate_evidence_fraction(selected, float(rule["window_threshold"]))
        direction = str(rule["score_direction"])
        values = document["mg_score"] if direction == "higher_is_mg" else -document["mg_score"]
        metrics = evaluate_scores(document["label"], values, float(rule["document_threshold"]))
        rows.append({
            "model": model_name, "window_length": window_length,
            "partition": partition, **rule, **metrics,
        })
    return rows


def train_local_evidence_models(
    processed_root: Path | str,
    datasets: list[str] | tuple[str, ...],
    llm_name: str,
    train_text_ids: Mapping[str, pd.DataFrame],
    validation_text_ids: Mapping[str, pd.DataFrame],
    *,
    classifier_names: tuple[str, ...] = ("logistic_regression", "gaussian_nb", "gaussian_mixture_nb"),
    window_lengths: tuple[int, ...] = (32, 64, 128),
    feature_columns: tuple[str, ...] = LOCAL_FEATURE_COLUMNS,
    stride: int = 16,
    top_k: int = 20,
    seed: int = 42,
    bayesian_config: BayesianConfig | None = None,
    gaussian_mixture_config: GaussianMixtureNBConfig | None = None,
) -> LocalEvidenceTrainingResult:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("datasets must not be empty")
    unknown = set(classifier_names).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown classifier name: {sorted(unknown)[0]!r}")
    for dataset in datasets:
        if not _cohort_key(train_text_ids[dataset]).isdisjoint(_cohort_key(validation_text_ids[dataset])):
            raise ValueError(f"train and validation cohorts overlap for dataset={dataset!r}")
    root = Path(processed_root)
    train_windows, train_coverage = _load_cohorts(root, datasets, llm_name, train_text_ids, window_lengths, stride, top_k, tuple(feature_columns))
    validation_windows, validation_coverage = _load_cohorts(root, datasets, llm_name, validation_text_ids, window_lengths, stride, top_k, tuple(feature_columns))
    fingerprint = training_input_fingerprint(
        pd.concat([train_windows, validation_windows], ignore_index=True),
        {
            "train": pd.concat(
                [_namespace(train_text_ids[dataset], dataset) for dataset in datasets],
                ignore_index=True,
            ),
            "validation": pd.concat(
                [_namespace(validation_text_ids[dataset], dataset) for dataset in datasets],
                ignore_index=True,
            ),
        },
    )
    registry: dict[tuple[str, int], FittedLocalEvidenceClassifier] = {}
    rule_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for classifier_name in classifier_names:
        for window_length in window_lengths:
            train = train_windows.loc[train_windows["window_length"].eq(window_length)]
            validation = validation_windows.loc[validation_windows["window_length"].eq(window_length)]
            fit_kwargs: dict[str, Any] = {}
            if classifier_name == "pymc_bayesian_logistic":
                fit_kwargs["bayesian_config"] = bayesian_config
            if classifier_name == "gaussian_mixture_nb":
                fit_kwargs["gaussian_mixture_config"] = gaussian_mixture_config
            fitted = fit_local_evidence_classifier(classifier_name, train, tuple(feature_columns), seed, **fit_kwargs)
            registry[(classifier_name, int(window_length))] = fitted
            validation_scored = score_local_evidence_classifier(fitted, validation, tuple(feature_columns))
            maximum = aggregate_maximum_window(validation_scored)
            maximum_parameters = _fit_document_threshold(maximum["label"], maximum["mg_score"])
            rule_rows.append({
                "model": classifier_name, "window_length": int(window_length),
                "aggregation": "maximum_window", "window_threshold": np.nan,
                "document_threshold": maximum_parameters["threshold"],
                "score_direction": maximum_parameters["direction"],
            })
            candidates = []
            for window_threshold in np.percentile(validation_scored["mg_score"], np.arange(5, 100, 5)):
                fraction = aggregate_evidence_fraction(validation_scored, float(window_threshold))
                parameters = _fit_document_threshold(fraction["label"], fraction["mg_score"])
                candidates.append({"window_threshold": float(window_threshold), **parameters})
            evidence = _select_evidence_parameters(candidates)
            rule_rows.append({
                "model": classifier_name, "window_length": int(window_length),
                "aggregation": "evidence_fraction", **evidence,
                "document_threshold": evidence["threshold"],
                "score_direction": evidence["direction"],
            })
            rules = [row for row in rule_rows if row["model"] == classifier_name and row["window_length"] == window_length]
            train_scored = score_local_evidence_classifier(fitted, train, tuple(feature_columns))
            metric_rows.extend(_score_metrics(train_scored, classifier_name, int(window_length), "train", rules))
            metric_rows.extend(_score_metrics(validation_scored, classifier_name, int(window_length), "validation", rules))
    return LocalEvidenceTrainingResult(
        LocalEvidenceModelBundle(llm_name, registry, pd.DataFrame(rule_rows), tuple(feature_columns), tuple(window_lengths), stride, top_k, tuple(datasets)),
        pd.DataFrame(metric_rows),
        pd.concat([train_coverage.assign(partition="train"), validation_coverage.assign(partition="validation")], ignore_index=True),
        fingerprint,
    )
