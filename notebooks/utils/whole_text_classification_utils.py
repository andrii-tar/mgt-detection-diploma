from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from notebooks.utils.classification_utils import (
    LABELS,
    SCALAR_FEATURE_COLUMNS,
    evaluate_scores,
    load_metrics,
)
from notebooks.utils.logprob_metrics_distributions_utils import (
    truncate_logprob_row_by_words,
    validate_word_limits,
)
from scripts.logprob_metrics import compute_logprob_metrics


@dataclass(frozen=True)
class WholeTextClassifierBundle:
    scorer_llm: str
    source_datasets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    logistic_pipeline: Pipeline


@dataclass(frozen=True)
class WholeTextTrainingResult:
    bundle: WholeTextClassifierBundle
    metrics: pd.DataFrame
    coverage: pd.DataFrame


@dataclass(frozen=True)
class WholeTextEvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    coverage: pd.DataFrame


def _validated_feature_columns(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must not be empty")
    duplicates = list(dict.fromkeys(feature for feature in features if features.count(feature) > 1))
    if duplicates:
        raise ValueError(f"duplicate feature names: {duplicates}")
    return features


def _normalized_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("null id values are not allowed")
    text = str(value).strip()
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return text
    if not numeric.is_finite():
        return text
    if numeric == 0:
        return "0"
    return format(numeric.normalize(), "f")


def _identity(frame: pd.DataFrame) -> pd.Series:
    identities = [
        (str(dataset), str(split), _normalized_id(row_id))
        for dataset, split, row_id in frame[["dataset", "split", "id"]].itertuples(index=False, name=None)
    ]
    return pd.Series(identities, index=frame.index, dtype="object")


def _prepare_cohort(
    frame: pd.DataFrame,
    *,
    dataset: str,
    partition: str,
    feature_columns: tuple[str, ...],
    scorer_llm: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"empty cohort for dataset={dataset!r}, partition={partition!r}")
    required = {"dataset", "model", "split", "id", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        missing_features = [column for column in feature_columns if column in missing]
        if missing_features:
            raise ValueError(f"missing feature columns: {missing_features}")
        raise ValueError(f"missing provenance columns: {missing}")

    cohort = frame.copy()
    if not cohort["dataset"].eq(dataset).all():
        raise ValueError(f"cohort dataset provenance does not match dataset={dataset!r}")
    if not cohort["model"].eq(scorer_llm).all():
        raise ValueError(f"cohort scorer provenance does not match scorer_llm={scorer_llm!r}")
    if not cohort["split"].isin(LABELS).all():
        raise ValueError("cohort splits must be HG or MG")

    mapped_labels = cohort["split"].map(LABELS).astype("int64")
    if "label" in cohort and not pd.to_numeric(cohort["label"], errors="coerce").eq(mapped_labels).all():
        raise ValueError("cohort labels do not match HG/MG split provenance")
    cohort["label"] = mapped_labels
    cohort["identity"] = _identity(cohort)
    if cohort["identity"].duplicated().any():
        raise ValueError("duplicate (dataset, split, id) rows")

    cohort.loc[:, list(feature_columns)] = (
        cohort.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    return cohort


def load_whole_text_cohort(
    processed_root: Path,
    *,
    dataset: str,
    scorer_llm: str,
    feature_columns: tuple[str, ...] = SCALAR_FEATURE_COLUMNS,
) -> pd.DataFrame:
    features = _validated_feature_columns(feature_columns)
    try:
        metrics = load_metrics(
            processed_root,
            selected_datasets=(dataset,),
            selected_models=(scorer_llm,),
            columns=("id", "split", *features),
        )
    except FileNotFoundError as error:
        raise ValueError(f"empty cohort for dataset={dataset!r}, scorer_llm={scorer_llm!r}") from error
    selected = metrics.loc[
        metrics["dataset"].eq(dataset) & metrics["model"].eq(scorer_llm) & metrics["split"].isin(LABELS)
    ].copy()
    if selected.empty:
        raise ValueError(f"empty cohort for dataset={dataset!r}, scorer_llm={scorer_llm!r}")
    unavailable = [feature for feature in features if feature not in selected or selected[feature].isna().all()]
    if unavailable:
        raise ValueError(f"missing feature columns: {unavailable}")

    selected = _prepare_cohort(
        selected,
        dataset=dataset,
        partition="loaded",
        feature_columns=features,
        scorer_llm=scorer_llm,
    )
    return selected[["dataset", "model", "split", "id", "label", "identity", *features]].reset_index(drop=True)


def _coverage(cohorts: Mapping[str, pd.DataFrame], partition: str) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "partition": partition,
            "rows": len(frame),
            "hg_rows": int(frame["label"].eq(0).sum()),
            "mg_rows": int(frame["label"].eq(1).sum()),
        }
        for dataset, frame in cohorts.items()
    ]


def _result_metadata(
    bundle: WholeTextClassifierBundle,
    *,
    dataset: str,
    partition: str,
    classifier: str,
    feature_set: str,
    threshold: float,
    direction: str,
) -> dict[str, object]:
    return {
        "scorer_llm": bundle.scorer_llm,
        "source_datasets": ", ".join(bundle.source_datasets),
        "dataset": dataset,
        "partition": partition,
        "classifier": classifier,
        "feature_set": feature_set,
        "threshold": threshold,
        "direction": direction,
    }


def _score_partition(
    bundle: WholeTextClassifierBundle,
    frame: pd.DataFrame,
    *,
    dataset: str,
    partition: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    feature_set = ", ".join(bundle.feature_columns)
    metadata = _result_metadata(
        bundle,
        dataset=dataset,
        partition=partition,
        classifier="logistic_regression",
        feature_set=feature_set,
        threshold=0.5,
        direction="higher_is_mg",
    )
    scores = pd.Series(
        bundle.logistic_pipeline.predict_proba(frame[list(bundle.feature_columns)])[:, 1],
        index=frame.index,
    )
    metric_rows.append({**metadata, **evaluate_scores(frame["label"], scores, 0.5)})
    prediction_frames.append(
        frame[["split", "id", "identity", "label"]]
        .assign(**metadata, mg_score=scores.to_numpy(), prediction=(scores >= 0.5).astype("int64").to_numpy())
    )
    coverage_rows.append(
        {
            **metadata,
            "requested_rows": len(frame),
            "rows": len(frame),
            "hg_rows": int(frame["label"].eq(0).sum()),
            "mg_rows": int(frame["label"].eq(1).sum()),
        }
    )
    return metric_rows, prediction_frames, coverage_rows


def fit_whole_text_classifiers(
    train_cohorts: Mapping[str, pd.DataFrame],
    validation_cohorts: Mapping[str, pd.DataFrame],
    *,
    scorer_llm: str,
    feature_columns: tuple[str, ...] = SCALAR_FEATURE_COLUMNS,
) -> WholeTextTrainingResult:
    features = _validated_feature_columns(feature_columns)
    if not train_cohorts or set(train_cohorts) != set(validation_cohorts):
        raise ValueError("train and validation cohorts must contain the same source datasets")

    prepared_train = {
        dataset: _prepare_cohort(
            frame,
            dataset=dataset,
            partition="train",
            feature_columns=features,
            scorer_llm=scorer_llm,
        )
        for dataset, frame in train_cohorts.items()
    }
    prepared_validation = {
        dataset: _prepare_cohort(
            validation_cohorts[dataset],
            dataset=dataset,
            partition="validation",
            feature_columns=features,
            scorer_llm=scorer_llm,
        )
        for dataset in train_cohorts
    }
    for dataset in prepared_train:
        train_ids = set(prepared_train[dataset]["identity"])
        validation_ids = set(prepared_validation[dataset]["identity"])
        if train_ids.intersection(validation_ids):
            raise ValueError(f"train/validation parent overlap for source dataset={dataset!r}")

    train = pd.concat(prepared_train.values(), ignore_index=True)
    validation = pd.concat(prepared_validation.values(), ignore_index=True)
    if train["label"].nunique() != 2:
        raise ValueError("training data must contain both classes")
    unavailable = [feature for feature in features if train[feature].isna().all()]
    if unavailable:
        raise ValueError(f"training data has no finite values for features: {unavailable}")

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )
    pipeline.fit(train[list(features)], train["label"])
    bundle = WholeTextClassifierBundle(
        scorer_llm=scorer_llm,
        source_datasets=tuple(train_cohorts),
        feature_columns=features,
        logistic_pipeline=pipeline,
    )
    metrics: list[dict[str, object]] = []
    for partition, frame in (("train", train), ("validation", validation)):
        partition_metrics, _, _ = _score_partition(
            bundle,
            frame,
            dataset=", ".join(bundle.source_datasets),
            partition=partition,
        )
        metrics.extend(partition_metrics)
    coverage = [*_coverage(prepared_train, "train"), *_coverage(prepared_validation, "validation")]
    return WholeTextTrainingResult(bundle=bundle, metrics=pd.DataFrame(metrics), coverage=pd.DataFrame(coverage))


def evaluate_whole_text_classifiers(
    bundle: WholeTextClassifierBundle,
    target_cohorts: Mapping[str, pd.DataFrame],
    *,
    partition: str,
) -> WholeTextEvaluationResult:
    if not target_cohorts:
        raise ValueError("target cohorts must not be empty")
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for dataset, frame in target_cohorts.items():
        prepared = _prepare_cohort(
            frame,
            dataset=dataset,
            partition=partition,
            feature_columns=bundle.feature_columns,
            scorer_llm=bundle.scorer_llm,
        )
        dataset_metrics, dataset_predictions, dataset_coverage = _score_partition(
            bundle,
            prepared,
            dataset=dataset,
            partition=partition,
        )
        metrics.extend(dataset_metrics)
        predictions.extend(dataset_predictions)
        coverage.extend(dataset_coverage)
    return WholeTextEvaluationResult(
        metrics=pd.DataFrame(metrics),
        predictions=pd.concat(predictions, ignore_index=True),
        coverage=pd.DataFrame(coverage),
    )


def _validation_length_metadata(
    result: WholeTextEvaluationResult,
    *,
    validation_variant: str,
    word_limit: int | None,
) -> WholeTextEvaluationResult:
    return WholeTextEvaluationResult(
        metrics=result.metrics.assign(validation_variant=validation_variant, word_limit=word_limit),
        predictions=result.predictions.assign(validation_variant=validation_variant, word_limit=word_limit),
        coverage=result.coverage.assign(validation_variant=validation_variant, word_limit=word_limit),
    )


def evaluate_whole_text_validation_lengths(
    bundle: WholeTextClassifierBundle,
    validation_cohorts: Mapping[str, pd.DataFrame],
    stage1_rows: Mapping[str, Sequence[Mapping[str, object]]],
    word_limits: Sequence[int],
) -> WholeTextEvaluationResult:
    limits = validate_word_limits(word_limits)
    original = _validation_length_metadata(
        evaluate_whole_text_classifiers(bundle, validation_cohorts, partition="validation"),
        validation_variant="original",
        word_limit=None,
    )
    results = [original]
    for word_limit in limits:
        short_cohorts: dict[str, pd.DataFrame] = {}
        for dataset, cohort in validation_cohorts.items():
            raw_rows = {
                (str(row.get("split")), _normalized_id(row.get("id"))): row
                for row in stage1_rows.get(dataset, ())
            }
            short_rows: list[dict[str, object]] = []
            for source in cohort.itertuples(index=False):
                key = (str(source.split), _normalized_id(source.id))
                if key not in raw_rows:
                    raise ValueError(f"missing Stage 1 row for validation text {dataset}/{key}")
                prefix = truncate_logprob_row_by_words(raw_rows[key], word_limit)
                metrics = compute_logprob_metrics(prefix)
                short_rows.append({
                    "dataset": dataset,
                    "model": bundle.scorer_llm,
                    "split": source.split,
                    "id": source.id,
                    "label": source.label,
                    **{feature: metrics[feature] for feature in bundle.feature_columns},
                })
            short_cohorts[dataset] = pd.DataFrame(short_rows)
        short = evaluate_whole_text_classifiers(bundle, short_cohorts, partition="validation")
        results.append(_validation_length_metadata(short, validation_variant="first_N_words", word_limit=word_limit))
    return WholeTextEvaluationResult(
        metrics=pd.concat([result.metrics for result in results], ignore_index=True),
        predictions=pd.concat([result.predictions for result in results], ignore_index=True),
        coverage=pd.concat([result.coverage for result in results], ignore_index=True),
    )
