"""Sampling and paired evaluation helpers for the mixed similarity ablation."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROBABILISTIC_FEATURE_COLUMNS = (
    "d_score_top20",
    "perplexity",
    "prompt_mu_score",
    "prompt_sigma_score",
    "d_score_top20_coverage",
    "top20_coverage__mean",
    "top20_coverage__std",
    "observed_logprob__mean",
    "observed_logprob__std",
    "mean_candidate_logprob_delta__mean",
    "mean_candidate_logprob_delta__std",
)
SIMILARITY_FEATURE_COLUMNS = (
    "mean_candidate_similarity__mean",
    "mean_candidate_similarity__std",
)
FEATURE_SET_COLUMNS = {
    "P": PROBABILISTIC_FEATURE_COLUMNS,
    "P+S": (*PROBABILISTIC_FEATURE_COLUMNS, *SIMILARITY_FEATURE_COLUMNS),
}
MIXED_DATASETS = ("kaggle_train", "mage_train", "outfox_chatgpt_train")
SPLIT_LABELS = {"HG": 0, "MG": 1}
DOCUMENT_KEY_COLUMNS = ("dataset", "split", "text_id")
MODEL_CONFIGURATION_COLUMNS = ("scorer_llm", "model_family", "window")


def intersect_eligible_texts_by_stratum(
    partitions: Mapping[tuple[str, str], Sequence[pd.DataFrame]],
) -> pd.DataFrame:
    """Intersect variant-specific IDs within each source/class stratum, then combine strata."""
    if not partitions:
        raise ValueError("At least one dataset/split stratum is required")
    eligible: list[pd.DataFrame] = []
    for (dataset, split), frames in sorted(partitions.items()):
        if not frames:
            raise ValueError(f"No prepared feature partitions were supplied for {dataset}/{split}")
        normalized: list[pd.DataFrame] = []
        identity_sets: list[set[tuple[object, ...]]] = []
        for frame in frames:
            required = {*DOCUMENT_KEY_COLUMNS, "label"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"Eligible text partition is missing columns: {missing}")
            current = frame.loc[:, [*DOCUMENT_KEY_COLUMNS, "label"]].drop_duplicates()
            if current.empty:
                raise ValueError(f"Prepared feature partition is empty for {dataset}/{split}")
            if not current["dataset"].astype(str).eq(str(dataset)).all() or not current["split"].astype(str).eq(str(split)).all():
                raise ValueError(f"Prepared feature partition contains rows outside {dataset}/{split}")
            if current.duplicated(list(DOCUMENT_KEY_COLUMNS)).any():
                raise ValueError(f"Documents have conflicting labels in {dataset}/{split}")
            normalized.append(current)
            identity_sets.append(set(map(tuple, current[list(DOCUMENT_KEY_COLUMNS)].itertuples(index=False, name=None))))
        common = set.intersection(*identity_sets)
        if not common:
            raise ValueError(f"No common eligible texts across feature variants for {dataset}/{split}")
        representative = normalized[0]
        identities = pd.Series(
            list(map(tuple, representative[list(DOCUMENT_KEY_COLUMNS)].itertuples(index=False, name=None))),
            index=representative.index,
        )
        eligible.append(representative.loc[identities.isin(common)])
    return (
        pd.concat(eligible, ignore_index=True)
        .sort_values([*DOCUMENT_KEY_COLUMNS], kind="stable")
        .reset_index(drop=True)
    )


def sample_paired_text_cohorts(
    eligible_texts: pd.DataFrame,
    *,
    train_texts_per_stratum: int = 50,
    evaluation_texts_per_stratum: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Select disjoint, deterministic train/evaluation texts per dataset and class split."""
    required = {*DOCUMENT_KEY_COLUMNS, "label"}
    missing = sorted(required - set(eligible_texts.columns))
    if missing:
        raise ValueError(f"Eligible text table is missing columns: {missing}")
    for name, value in (
        ("train_texts_per_stratum", train_texts_per_stratum),
        ("evaluation_texts_per_stratum", evaluation_texts_per_stratum),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    parents = eligible_texts.loc[:, [*DOCUMENT_KEY_COLUMNS, "label"]].drop_duplicates()
    if parents.empty or parents[list(DOCUMENT_KEY_COLUMNS)].isna().any().any():
        raise ValueError("Eligible text table must contain non-null document identities")
    if not parents["label"].isin([0, 1]).all():
        raise ValueError("Eligible text labels must be binary 0/1")
    if parents.duplicated(list(DOCUMENT_KEY_COLUMNS)).any():
        raise ValueError("A document identity has conflicting labels")
    if not parents.apply(lambda row: SPLIT_LABELS.get(str(row["split"])) == int(row["label"]), axis=1).all():
        raise ValueError("HG/MG split names must match binary labels 0/1")

    selected: list[pd.DataFrame] = []
    strata = list(parents.groupby(["dataset", "split"], sort=True, dropna=False))
    for stratum_index, ((dataset, split), group) in enumerate(strata):
        required_count = train_texts_per_stratum + evaluation_texts_per_stratum
        if len(group) < required_count:
            raise ValueError(
                f"Insufficient eligible texts for {dataset}/{split}: "
                f"need {required_count}, found {len(group)}"
            )
        shuffled = group.sample(frac=1, random_state=int(seed) + stratum_index).reset_index(drop=True)
        train = shuffled.iloc[:train_texts_per_stratum].copy().assign(partition="train")
        evaluation = shuffled.iloc[
            train_texts_per_stratum:required_count
        ].copy().assign(partition="evaluation")
        selected.extend((train, evaluation))
    if not selected:
        raise ValueError("No dataset/class strata were available for sampling")
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["partition", "dataset", "split", "text_id"], kind="stable")
        .reset_index(drop=True)
    )


def _with_dataset_aliases(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    if "dataset" not in frame and "source_dataset" in frame:
        frame["dataset"] = frame["source_dataset"]
    if "source_dataset" not in frame and "dataset" in frame:
        frame["source_dataset"] = frame["dataset"]
    return frame


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> float:
    return float(f1_score(labels, predictions, average="macro", sample_weight=weights, zero_division=0))


def paired_macro_f1_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare P and P+S macro-F1 with paired, stratified document-cluster bootstrap."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    frame = _with_dataset_aliases(predictions)
    required = {
        *MODEL_CONFIGURATION_COLUMNS,
        "feature_set",
        *DOCUMENT_KEY_COLUMNS,
        "window_id",
        "label",
        "prediction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction table cannot be empty")
    if set(frame["feature_set"].unique()) != {"P", "P+S"}:
        raise ValueError("Predictions must contain exactly the P and P+S feature sets")
    if not frame["label"].isin([0, 1]).all() or not frame["prediction"].isin([0, 1]).all():
        raise ValueError("Labels and predictions must be binary 0/1")
    row_keys = [*MODEL_CONFIGURATION_COLUMNS, *DOCUMENT_KEY_COLUMNS, "window_id"]
    if frame.duplicated([*row_keys, "feature_set"]).any():
        raise ValueError("Duplicate prediction rows found for a configuration/window/feature set")

    grouped = frame.groupby(list(MODEL_CONFIGURATION_COLUMNS), sort=True, dropna=False)
    summaries: list[dict[str, object]] = []
    random = np.random.default_rng(seed)
    for configuration, group in grouped:
        if not isinstance(configuration, tuple):
            configuration = (configuration,)
        p_rows = group.loc[group["feature_set"].eq("P"), [*row_keys, "label", "prediction"]]
        ps_rows = group.loc[group["feature_set"].eq("P+S"), [*row_keys, "label", "prediction"]]
        paired = p_rows.merge(
            ps_rows,
            on=row_keys,
            how="outer",
            suffixes=("_P", "_P_plus_S"),
            indicator=True,
            validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError("P and P+S must have exactly the same evaluation windows")
        if not paired["label_P"].equals(paired["label_P_plus_S"]):
            raise ValueError("P and P+S evaluation labels do not match")
        paired = paired.drop(columns="_merge")
        if paired.groupby(list(DOCUMENT_KEY_COLUMNS), dropna=False)["label_P"].nunique().gt(1).any():
            raise ValueError("Documents have conflicting labels across windows")

        labels = paired["label_P"].to_numpy(dtype=int)
        p_predictions = paired["prediction_P"].to_numpy(dtype=int)
        ps_predictions = paired["prediction_P_plus_S"].to_numpy(dtype=int)
        point_p = _macro_f1(labels, p_predictions)
        point_ps = _macro_f1(labels, ps_predictions)

        document_keys = list(DOCUMENT_KEY_COLUMNS)
        paired["_document_key"] = paired[document_keys].astype(str).agg("\x1f".join, axis=1)
        document_labels = paired[["_document_key", "dataset", "label_P"]].drop_duplicates()
        strata = [
            group["_document_key"].to_numpy()
            for _, group in document_labels.groupby(["dataset", "label_P"], sort=True, dropna=False)
        ]
        row_document_keys = paired["_document_key"].to_numpy()
        bootstrap_deltas = np.empty(n_bootstrap, dtype=float)
        for iteration in range(n_bootstrap):
            draw_counts: Counter[str] = Counter()
            for stratum in strata:
                draws = random.choice(stratum, size=len(stratum), replace=True)
                draw_counts.update(draws.tolist())
            weights = np.fromiter(
                (draw_counts.get(key, 0) for key in row_document_keys), dtype=float, count=len(row_document_keys)
            )
            bootstrap_deltas[iteration] = _macro_f1(labels, ps_predictions, weights) - _macro_f1(
                labels, p_predictions, weights
            )
        summary = dict(zip(MODEL_CONFIGURATION_COLUMNS, configuration))
        summary.update(
            {
                "evaluation_document_count": int(document_labels["_document_key"].nunique()),
                "evaluation_window_count": int(len(paired)),
                "macro_f1_P": point_p,
                "macro_f1_P_plus_S": point_ps,
                "delta_macro_f1": point_ps - point_p,
                "ci_lower": float(np.quantile(bootstrap_deltas, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_deltas, 0.975)),
                "bootstrap_replicates": n_bootstrap,
            }
        )
        summaries.append(summary)
    return pd.DataFrame(summaries).sort_values(list(MODEL_CONFIGURATION_COLUMNS), kind="stable").reset_index(drop=True)
