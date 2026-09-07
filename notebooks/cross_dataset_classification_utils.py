from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from notebooks.classification_utils import evaluate_scores
from notebooks.classification_utils import limit_parent_texts, split_cohort
from notebooks.whole_text_classification_utils import (
    WholeTextEvaluationResult,
    WholeTextEvaluationResult,
    WholeTextTrainingResult,
    evaluate_whole_text_classifiers,
    fit_whole_text_classifiers,
)


SCORER_LLMS = (
    "gemma_3_1b_it",
    "gemma_3_270m_it",
    "littlelamb",
    "smollm2_360m",
)
DATASETS = {
    "Kaggle": "kaggle_train",
    "MAGE": "mage_train",
    "Outfox": "outfox_chatgpt_train",
    "BBC": "bbc_news_content",
}
TRAINING_REGIMES = {
    "Kaggle": ("Kaggle",),
    "MAGE": ("MAGE",),
    "Outfox": ("Outfox",),
    "Mixed": ("Kaggle", "MAGE", "Outfox"),
}
EVALUATION_REGIMES = {
    **TRAINING_REGIMES,
    "BBC": ("BBC",),
}


def build_deterministic_parent_cohorts(
    frame: pd.DataFrame,
    *,
    id_column: str,
    max_per_split: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Cap each HG/MG parent split by numeric ID before stratified partitioning."""
    parents = frame.drop_duplicates(subset=["split", id_column, "label"]).reset_index(drop=True)
    limited = limit_parent_texts(parents, id_column=id_column, max_per_split=max_per_split)
    if limited["label"].nunique() == 2:
        return split_cohort(limited, seed=seed)
    if limited.empty or not (limited["split"].eq("HG") & limited["label"].eq(0)).all():
        raise ValueError("one-class cohort must use the HG split with label 0")
    development, test = train_test_split(limited, test_size=0.2, random_state=seed)
    train, validation = train_test_split(development, test_size=0.25, random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def train_whole_text_models(
    train_cohorts: Mapping[str, pd.DataFrame],
    validation_cohorts: Mapping[str, pd.DataFrame],
    *,
    scorer_llm: str,
) -> WholeTextTrainingResult:
    return fit_whole_text_classifiers(
        train_cohorts,
        validation_cohorts,
        scorer_llm=scorer_llm,
    )


def evaluate_whole_text_datasets(
    bundle: object,
    target_cohorts: Mapping[str, pd.DataFrame],
    *,
    partition: str,
) -> WholeTextEvaluationResult:
    return evaluate_whole_text_classifiers(bundle, target_cohorts, partition=partition)


def combine_whole_text_dataset_results(
    results: list[WholeTextEvaluationResult] | tuple[WholeTextEvaluationResult, ...],
    *,
    evaluation_regime: str,
) -> WholeTextEvaluationResult:
    """Pool document predictions from several datasets before calculating metrics."""
    if not results:
        raise ValueError("whole-text results must not be empty")
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    coverage = pd.concat([result.coverage for result in results], ignore_index=True)
    rule_columns = ["classifier", "feature_set", "threshold", "direction"]
    metric_rows: list[dict[str, object]] = []
    for rule_values, documents in predictions.groupby(rule_columns, dropna=False, sort=False):
        rule = dict(zip(rule_columns, rule_values, strict=True))
        scores = documents["mg_score"] if rule["direction"] == "higher_is_mg" else -documents["mg_score"]
        metric_rows.append({
            **{column: documents.iloc[0][column] for column in ("scorer_llm", "source_datasets", "partition") if column in documents},
            "dataset": evaluation_regime,
            "evaluation_dataset": evaluation_regime,
            "evaluation_regime": evaluation_regime,
            **rule,
            **evaluate_scores(documents["label"], scores, float(rule["threshold"])),
        })
    predictions = predictions.assign(evaluation_dataset=evaluation_regime, evaluation_regime=evaluation_regime)
    coverage = coverage.assign(evaluation_dataset=evaluation_regime, evaluation_regime=evaluation_regime)
    return WholeTextEvaluationResult(pd.DataFrame(metric_rows), predictions, coverage)


def build_shared_parent_manifest(
    scalar: pd.DataFrame,
    local: pd.DataFrame,
    *,
    seed: int = 42,
    max_per_split: int = 500,
) -> dict[str, pd.DataFrame]:
    """Partition only parents represented by both scalar and local data."""
    scalar_parents = scalar[["split", "id", "label"]].rename(columns={"id": "parent_id"})
    local_parents = local[["split", "text_id", "label"]].drop_duplicates()
    scalar_parents["numeric_parent_id"] = pd.to_numeric(scalar_parents["parent_id"], errors="raise")
    local_parents["numeric_parent_id"] = pd.to_numeric(local_parents["text_id"], errors="raise")
    eligible = scalar_parents.merge(
        local_parents,
        on=["split", "numeric_parent_id"],
        suffixes=("_scalar", "_local"),
        how="inner",
        validate="one_to_one",
    )
    if not eligible["label_scalar"].eq(eligible["label_local"]).all():
        raise ValueError("scalar and local parent labels do not match")
    parents = eligible.rename(columns={"label_scalar": "label"})[
        ["split", "text_id", "label", "numeric_parent_id"]
    ]
    parents["text_id"] = parents["numeric_parent_id"].astype("int64")
    return build_deterministic_parent_cohorts(
        parents,
        id_column="numeric_parent_id",
        max_per_split=max_per_split,
        seed=seed,
    )


def select_scalar_partition(scalar: pd.DataFrame, manifest_partition: pd.DataFrame) -> pd.DataFrame:
    """Select scalar rows using the manifest's canonical numeric parent IDs."""
    selected = scalar.copy()
    selected["_canonical_parent_id"] = pd.to_numeric(selected["id"], errors="raise").astype("int64")
    manifest = manifest_partition[["split", "text_id"]].copy()
    manifest["_canonical_parent_id"] = pd.to_numeric(manifest["text_id"], errors="raise").astype("int64")
    return selected.merge(
        manifest[["split", "_canonical_parent_id"]],
        on=["split", "_canonical_parent_id"],
        how="inner",
        validate="one_to_one",
    ).drop(columns="_canonical_parent_id")


def prepare_matrix_frame(frame: pd.DataFrame, *, block: str) -> pd.DataFrame:
    """Normalize model output names used by all matrix displays."""
    result = frame.copy()
    if "classifier" not in result:
        if "model" in result:
            result["classifier"] = result["model"]
        else:
            result["classifier"] = block
    return result


def pivot_matrix(
    frame: pd.DataFrame,
    *,
    value: str,
    index: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    required = {*index, *columns, value}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"matrix frame is missing columns: {missing}")
    return frame.pivot_table(index=list(index), columns=list(columns), values=value, aggfunc="first")


def label_result_frame(
    frame: pd.DataFrame,
    *,
    llm: str,
    training_regime: str,
    evaluation_regime: str,
    evaluation_dataset: str,
    block: str,
) -> pd.DataFrame:
    return prepare_matrix_frame(frame, block=block).assign(
        block=block,
        llm=llm,
        training_regime=training_regime,
        evaluation_regime=evaluation_regime,
        evaluation_dataset=evaluation_dataset,
    )


def concat_or_empty(frames: list[pd.DataFrame], columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=list(columns))


def source_coverage_row(llm: str, source_dataset: str, error: Exception) -> dict[str, object]:
    return {
        "llm": llm,
        "source_dataset": source_dataset,
        "status": "unavailable",
        "classifier": "source_coverage",
        "error": str(error),
        "rows": 0,
    }
