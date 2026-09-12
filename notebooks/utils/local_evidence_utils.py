from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from notebooks.utils.classification_utils import evaluate_scores, fit_single_feature_threshold, limit_parent_texts, split_cohort
from notebooks.utils.logprob_metrics_distributions_utils import token_prefix_length_for_words, validate_word_limits
from notebooks.utils.token_candidate_similarity_heatmaps_utils import (
    split_dir,
    validated_text_summary,
)


LOCAL_METRICS = (
    "observed_logprob",
    "mean_candidate_logprob_delta",
    "mean_candidate_similarity",
    "top20_coverage",
)
WINDOW_STATISTICS = ("mean", "std", "q10", "q50", "q90", "min", "max", "slope", "finite_fraction")
LOCAL_FEATURE_COLUMNS = tuple(f"{metric}__{statistic}" for metric in LOCAL_METRICS for statistic in WINDOW_STATISTICS)
DEFAULT_WINDOW_LENGTHS = (32, 64, 96, 128, 192, 256)
LOCAL_EVIDENCE_CACHE_SCHEMA_VERSION = 1


def _window_starts(token_count: int, window_length: int, stride: int) -> list[int]:
    if token_count < window_length:
        return []
    starts = list(range(0, token_count - window_length + 1, stride))
    tail_start = token_count - window_length
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _pool_feature(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    finite_mask = np.isfinite(array)
    finite = array[finite_mask]
    statistics = {statistic: float("nan") for statistic in WINDOW_STATISTICS}
    statistics["finite_fraction"] = float(finite.size / array.size)
    if not finite.size:
        return statistics
    statistics.update(
        mean=float(finite.mean()),
        std=float(finite.std()),
        q10=float(np.quantile(finite, 0.1)),
        q50=float(np.quantile(finite, 0.5)),
        q90=float(np.quantile(finite, 0.9)),
        min=float(finite.min()),
        max=float(finite.max()),
    )
    if finite.size >= 2:
        statistics["slope"] = float(np.polyfit(np.flatnonzero(finite_mask), finite, 1)[0])
    return statistics


def _iter_parquet_rows(paths: list[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            yield batch.to_pylist()[0]


def _iter_paired_processed_rows(
    processed_root: Path,
    dataset: str,
    model: str,
    split: str,
    text_ids: tuple[int, ...] | None,
) -> tuple[Iterator[tuple[dict[str, object], dict[str, object]]], int]:
    data_paths = sorted((split_dir(processed_root, dataset, model, split) / "data").glob("*.parquet"))
    similarity_paths = sorted((split_dir(processed_root, dataset, model, split) / "token_similarities").glob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No stage-1 data parquet files for {dataset=} {model=} {split=}")
    if not similarity_paths:
        raise FileNotFoundError(f"No token similarity parquet files for {dataset=} {model=} {split=}")
    selected_ids = None if text_ids is None else set(text_ids)
    total = len(selected_ids) if selected_ids is not None else sum(
        pq.ParquetFile(path).metadata.num_rows for path in similarity_paths
    )

    def rows() -> Iterator[tuple[dict[str, object], dict[str, object]]]:
        sentinel = object()
        for logprob_row, similarity_row in zip_longest(
            _iter_parquet_rows(data_paths), _iter_parquet_rows(similarity_paths), fillvalue=sentinel
        ):
            if logprob_row is sentinel:
                raise KeyError(f"Unexpected token similarity row for {dataset=} {model=} {split=}")
            text_id = int(logprob_row["id"])
            if similarity_row is sentinel or int(similarity_row["id"]) != text_id:
                raise KeyError(f"No token similarity row for text id {text_id}")
            if selected_ids is None or text_id in selected_ids:
                yield logprob_row, similarity_row

    return rows(), total


def build_local_evidence_windows(
    processed_root: Path,
    dataset: str,
    model: str,
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
    splits: tuple[str, ...] = ("HG", "MG"),
    show_progress: bool = True,
    text_ids: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not window_lengths or any(
        isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length <= 0
        for length in window_lengths
    ):
        raise ValueError("window_lengths must contain positive integers")

    identity_columns = [
        "dataset",
        "model",
        "split",
        "label",
        "text_id",
        "window_length",
        "start_token",
        "stop_token",
        "num_tokens",
        "window_text",
    ]
    rows = []
    for split in splits:
        paired_rows, total = _iter_paired_processed_rows(processed_root, dataset, model, split, text_ids)
        if show_progress:
            paired_rows = tqdm(
                paired_rows,
                total=total,
                desc=f"{split}: local evidence",
                unit="text",
            )
        for logprob_row, similarity_row in paired_rows:
            text_id = int(logprob_row["id"])
            summary, token_labels = validated_text_summary(logprob_row, similarity_row, top_k=top_k)
            for window_length in window_lengths:
                for start in _window_starts(len(token_labels), window_length, stride):
                    stop = start + window_length
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "split": split,
                        "label": 0 if split == "HG" else 1,
                        "text_id": text_id,
                        "window_length": window_length,
                        "start_token": start,
                        "stop_token": stop,
                        "num_tokens": stop - start,
                        "window_text": "".join(token_labels[start:stop]),
                    }
                    for metric in LOCAL_METRICS:
                        for statistic, value in _pool_feature(summary[metric][start:stop]).items():
                            row[f"{metric}__{statistic}"] = value
                    rows.append(row)
    return pd.DataFrame(rows, columns=[*identity_columns, *LOCAL_FEATURE_COLUMNS])


def default_local_evidence_path(processed_root: Path, dataset: str, model: str) -> Path:
    return split_dir(processed_root, dataset, model, "HG").parent / "local_evidence" / "windows.parquet"


def _local_evidence_cache_metadata(
    *,
    dataset: str,
    model: str,
    splits: tuple[str, ...],
    window_lengths: tuple[int, ...],
    stride: int,
    top_k: int,
) -> dict[str, object]:
    return {
        "schema_version": LOCAL_EVIDENCE_CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "model": model,
        "splits": sorted(set(splits)),
        "window_lengths": list(window_lengths),
        "stride": stride,
        "top_k": top_k,
        "feature_columns": list(LOCAL_FEATURE_COLUMNS),
    }


def save_local_evidence_windows(
    windows: pd.DataFrame,
    path: Path,
    *,
    dataset: str,
    model: str,
    splits: tuple[str, ...],
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(windows, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"local_evidence_cache"] = json.dumps(
        _local_evidence_cache_metadata(
            dataset=dataset,
            model=model,
            splits=splits,
            window_lengths=window_lengths,
            stride=stride,
            top_k=top_k,
        ),
        sort_keys=True,
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


def load_local_evidence_windows(
    path: Path,
    *,
    dataset: str,
    model: str,
    splits: tuple[str, ...],
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
    text_ids: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    try:
        metadata = pq.read_schema(path).metadata or {}
        actual = json.loads(metadata[b"local_evidence_cache"].decode("utf-8"))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("local-evidence cache configuration is incompatible") from error
    expected = _local_evidence_cache_metadata(
        dataset=dataset,
        model=model,
        splits=splits,
        window_lengths=window_lengths,
        stride=stride,
        top_k=top_k,
    )
    if actual != expected:
        raise ValueError("local-evidence cache configuration is incompatible")
    requested = None
    if text_ids is not None:
        required = {"split", "text_id"}
        missing = required.difference(text_ids.columns)
        if missing:
            raise ValueError(f"missing cohort columns: {', '.join(sorted(missing))}")
        requested = text_ids[["split", "text_id"]].copy()
        requested["text_id"] = requested["text_id"].astype("int64")
        if requested.duplicated().any():
            raise ValueError("cohort text ids must be unique")

    read_columns = None if columns is None else list(dict.fromkeys([*columns, "split", "text_id"]))
    read_kwargs = (
        {}
        if requested is None
        else {"filters": [("text_id", "in", sorted(requested["text_id"].unique()))]}
    )
    windows = pd.read_parquet(path, columns=read_columns, **read_kwargs)
    if requested is not None:
        windows = windows.merge(requested, on=["split", "text_id"], how="inner", validate="many_to_one")
    if columns is not None:
        windows = windows.loc[:, list(columns)]
    return windows


def load_or_build_local_evidence_windows(
    processed_root: Path,
    dataset: str,
    model: str,
    *,
    path: Path | None = None,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
    splits: tuple[str, ...] = ("HG", "MG"),
    recompute: bool = False,
    text_ids: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = default_local_evidence_path(processed_root, dataset, model) if path is None else path
    normalized_splits = tuple(sorted(set(splits)))
    if not recompute:
        try:
            return load_local_evidence_windows(
                path,
                dataset=dataset,
                model=model,
                splits=normalized_splits,
                window_lengths=window_lengths,
                stride=stride,
                top_k=top_k,
                text_ids=text_ids,
                columns=columns,
            )
        except (FileNotFoundError, ValueError):
            pass
    windows = build_local_evidence_windows(
        processed_root,
        dataset,
        model,
        window_lengths=window_lengths,
        stride=stride,
        top_k=top_k,
        splits=normalized_splits,
        text_ids=None if text_ids is None else tuple(sorted(text_ids["text_id"].unique())),
    )
    save_local_evidence_windows(
        windows,
        path,
        dataset=dataset,
        model=model,
        splits=normalized_splits,
        window_lengths=window_lengths,
        stride=stride,
        top_k=top_k,
    )
    if text_ids is not None or columns is not None:
        del windows
        return load_local_evidence_windows(
            path,
            dataset=dataset,
            model=model,
            splits=normalized_splits,
            window_lengths=window_lengths,
            stride=stride,
            top_k=top_k,
            text_ids=text_ids,
            columns=columns,
        )
    return windows


def load_local_evidence_parent_metadata(
    processed_root: Path,
    dataset: str,
    model: str,
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
    splits: tuple[str, ...] = ("HG", "MG"),
) -> pd.DataFrame:
    """Load the distinct parent identities needed to build a shared cohort."""
    windows = load_or_build_local_evidence_windows(
        processed_root,
        dataset,
        model,
        window_lengths=window_lengths,
        stride=stride,
        top_k=top_k,
        splits=splits,
        columns=("split", "text_id", "label"),
    )
    return windows.drop_duplicates().reset_index(drop=True)


def load_local_evidence_cohort(
    processed_root: Path,
    dataset: str,
    model: str,
    text_ids: pd.DataFrame,
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    stride: int = 16,
    top_k: int = 20,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Load a requested parent-text cohort from the complete local-evidence cache."""
    required = {"split", "text_id"}
    missing = required.difference(text_ids.columns)
    if missing:
        raise ValueError(f"missing cohort columns: {', '.join(sorted(missing))}")
    requested = text_ids[["split", "text_id"]].copy()
    requested["text_id"] = requested["text_id"].astype("int64")
    if requested.duplicated().any():
        raise ValueError("cohort text ids must be unique")
    splits = tuple(sorted(requested["split"].astype(str).unique()))
    if not splits:
        return pd.DataFrame()
    windows = load_or_build_local_evidence_windows(
        processed_root,
        dataset,
        model,
        window_lengths=window_lengths,
        stride=stride,
        top_k=top_k,
        splits=splits,
        text_ids=requested,
    )
    available = windows[["split", "text_id"]].drop_duplicates()
    missing_ids = requested.merge(available, on=["split", "text_id"], how="left", indicator=True)
    if missing_ids["_merge"].eq("left_only").any():
        absent = missing_ids.loc[missing_ids["_merge"].eq("left_only"), ["split", "text_id"]]
        raise KeyError(f"requested local-evidence texts are unavailable: {absent.to_dict('records')}")
    return windows.merge(requested.assign(_requested=True), on=["split", "text_id"], how="inner", validate="many_to_one").drop(
        columns="_requested"
    )


def limit_local_evidence_windows_by_words(
    windows: pd.DataFrame,
    stage1_rows: Mapping[tuple[str, str], Mapping[str, object]],
    word_limit: int,
) -> pd.DataFrame:
    validate_word_limits((word_limit,))
    boundaries: dict[tuple[str, str], int] = {}
    for key, row in stage1_rows.items():
        boundaries[key] = token_prefix_length_for_words(
            str(row.get("text") or ""), list(row.get("tokens", [])), word_limit
        )
    keys = list(zip(windows["split"].astype(str), windows["text_id"].astype(str), strict=True))
    limits = pd.Series([boundaries.get(key, 0) for key in keys], index=windows.index)
    return windows.loc[windows["stop_token"].le(limits)].reset_index(drop=True)


def split_local_evidence_texts(
    windows: pd.DataFrame,
    *,
    seed: int = 42,
    max_per_split: int | None = None,
) -> dict[str, pd.DataFrame]:
    parent_columns = ["split", "text_id", "label"]
    parents = windows[parent_columns].drop_duplicates().reset_index(drop=True)
    label_counts = parents.groupby(["split", "text_id"])["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("each parent text must have exactly one label")
    if max_per_split is not None:
        parents = limit_parent_texts(parents, id_column="text_id", max_per_split=max_per_split)
    if parents["label"].nunique() == 2:
        return split_cohort(parents, seed=seed)
    if parents.empty or not (parents["split"].eq("HG") & parents["label"].eq(0)).all():
        raise ValueError("one-class cohort must use the HG split with label 0")

    development, test = train_test_split(parents, test_size=0.2, random_state=seed)
    train, validation = train_test_split(development, test_size=0.25, random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def aggregate_maximum_window(scored_windows: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["split", "text_id", "label", "window_length"]
    selected = (
        scored_windows.sort_values(
            [*group_columns, "mg_score", "start_token"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(group_columns, as_index=False, sort=False)
        .head(1)
        .copy()
    )
    return selected.rename(columns={"start_token": "selected_start_token", "stop_token": "selected_stop_token"})


def aggregate_evidence_fraction(scored_windows: pd.DataFrame, window_threshold: float) -> pd.DataFrame:
    group_columns = ["split", "text_id", "label", "window_length"]
    maximum = aggregate_maximum_window(scored_windows)
    fractions = (
        scored_windows.assign(is_evidence=scored_windows["mg_score"] >= window_threshold)
        .groupby(group_columns, as_index=False)["is_evidence"]
        .mean()
        .rename(columns={"is_evidence": "mg_score"})
    )
    return maximum.drop(columns="mg_score").merge(fractions, on=group_columns, validate="one_to_one")


def plot_local_evidence_timeline(scored_windows: pd.DataFrame, prediction: pd.Series) -> plt.Figure:
    windows = scored_windows.loc[
        scored_windows["split"].eq(prediction["split"])
        & scored_windows["text_id"].eq(prediction["text_id"])
        & scored_windows["window_length"].eq(prediction["window_length"])
    ]
    midpoints = (windows["start_token"] + windows["stop_token"]) / 2
    figure, axis = plt.subplots()
    axis.plot(midpoints, windows["mg_score"], marker="o")

    for window in windows.itertuples():
        score = float(window.mg_score)
        window_threshold = float(window.window_threshold)
        if np.isfinite(score) and np.isfinite(window_threshold) and score >= window_threshold:
            axis.axvspan(window.start_token, window.stop_token, alpha=0.2, color="tab:red")

    document_threshold = float(prediction["document_threshold"])
    if np.isfinite(document_threshold):
        axis.axhline(document_threshold, color="tab:gray", linestyle="--")
    axis.set_xlabel("Token position")
    axis.set_ylabel("MG score")
    axis.set_title(
        f"{prediction['split']} text {prediction['text_id']} | "
        f"label={prediction['label']}, prediction={prediction['prediction']} | "
        f"score={prediction['mg_score']:.3f}, threshold={document_threshold:.3f}"
    )
    return figure


def _partition_windows(windows: pd.DataFrame, parents: pd.DataFrame, partition: str) -> pd.DataFrame:
    parent_labels = parents[["split", "text_id", "label"]].rename(columns={"label": "parent_label"})
    filtered = windows.merge(parent_labels, on=["split", "text_id"], how="inner", validate="many_to_one")
    if not filtered["label"].eq(filtered["parent_label"]).all():
        raise ValueError(f"{partition} partition has parent labels inconsistent with its windows")
    filtered = filtered.drop(columns="parent_label")
    if set(filtered["label"]) != {0, 1}:
        raise ValueError(f"{partition} partition must have window rows for both labels")
    return filtered.assign(partition=partition)


def _window_contributions(model: Pipeline, windows: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    scored = windows.copy()
    transformed = model.named_steps["scale"].transform(model.named_steps["impute"].transform(scored[list(feature_columns)]))
    contribution_columns = [f"contribution__{feature}" for feature in feature_columns]
    scored.loc[:, contribution_columns] = transformed * model.named_steps["model"].coef_[0]
    contributions = scored[contribution_columns].to_numpy(dtype="float64")
    scored["top_positive_feature"] = np.asarray(feature_columns)[np.argmax(contributions, axis=1)]
    scored["top_negative_feature"] = np.asarray(feature_columns)[np.argmin(contributions, axis=1)]
    scored["intercept"] = float(model.named_steps["model"].intercept_[0])
    scored["mg_score"] = model.predict_proba(scored[list(feature_columns)])[:, 1]
    return scored


def _decision_scores(values: pd.Series, direction: str) -> pd.Series:
    scores = pd.to_numeric(values, errors="raise").astype("float64")
    if direction == "higher_is_mg":
        return scores
    if direction == "lower_is_mg":
        return -scores
    raise ValueError(f"unknown score direction: {direction!r}")


def _fit_document_threshold(y_true: pd.Series, mg_score: pd.Series) -> dict[str, float | str]:
    fitted = fit_single_feature_threshold(y_true, mg_score)
    direction = str(fitted["direction"])
    decision_scores = _decision_scores(mg_score, direction)
    labels = pd.Series(y_true).to_numpy(dtype="int64")
    candidates = [
        {
            "threshold": float(threshold),
            "macro_f1": float(f1_score(labels, decision_scores >= threshold, average="macro", zero_division=0)),
        }
        for threshold in np.unique(decision_scores)
    ]
    return {"direction": direction, **max(candidates, key=lambda candidate: (candidate["macro_f1"], candidate["threshold"]))}


def _select_evidence_parameters(candidates: list[dict[str, float | str]]) -> dict[str, float | str]:
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["macro_f1"]),
            float(candidate["window_threshold"]),
            float(candidate["threshold"]),
        ),
    )


def fit_local_evidence_model(
    windows: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    window_length: int,
    feature_columns: tuple[str, ...],
    seed: int,
) -> tuple[Pipeline, tuple[str, ...]]:
    feature_columns = tuple(feature_columns)
    missing_features = [feature for feature in feature_columns if feature not in windows.columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    train = _partition_windows(windows, partitions["train"], "train")
    train = train.loc[train["window_length"] == window_length]
    if set(train["label"]) != {0, 1}:
        raise ValueError(f"window_length={window_length} must have rows for both labels in the train partition")
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(train[list(feature_columns)], train["label"])
    return model, feature_columns


def print_local_evidence_model_coefficients(
    model: Pipeline,
    feature_columns: tuple[str, ...],
    window_length: int,
) -> None:
    coefficients = np.asarray(model.named_steps["model"].coef_[0], dtype="float64")
    if len(coefficients) != len(feature_columns):
        raise ValueError("coefficient vector length does not match feature columns")

    print(f"Local evidence model coefficients (window_length={window_length})")
    print("Positive coefficients increase the MG logit and negative coefficients decrease it.")
    for feature, coefficient in zip(feature_columns, coefficients):
        print(f"{feature}: coefficient={coefficient:.6f}")

    grouped: dict[str, list[tuple[str, float]]] = {}
    for feature, coefficient in zip(feature_columns, coefficients):
        if "__" in feature:
            metric, statistic = feature.split("__", 1)
        else:
            metric, statistic = feature, "no statistic"
        grouped.setdefault(metric, []).append((statistic, float(coefficient)))
    for metric, entries in grouped.items():
        values = np.asarray([coefficient for _, coefficient in entries], dtype="float64")
        maximum_index = int(np.argmax(np.abs(values)))
        print(
            f"{metric}: mean_signed={values.mean():.6f} mean_abs={np.abs(values).mean():.6f} "
            f"l1={np.abs(values).sum():.6f} max_abs={np.abs(values[maximum_index]):.6f} "
            f"({entries[maximum_index][0]})"
        )


def score_local_evidence_windows(
    model: Pipeline,
    windows: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    feature_columns = tuple(feature_columns)
    missing_features = [feature for feature in feature_columns if feature not in windows.columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    return _window_contributions(model, windows, feature_columns)


def run_local_evidence_experiments(
    windows: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    *,
    feature_columns: tuple[str, ...] = LOCAL_FEATURE_COLUMNS,
    seed: int = 42,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_features = [feature for feature in feature_columns if feature not in windows.columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    partitioned = {
        partition: _partition_windows(windows, partitions[partition], partition)
        for partition in ("train", "validation", "test")
    }
    summaries: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []

    window_lengths = sorted(windows["window_length"].unique())
    if show_progress:
        window_lengths = tqdm(window_lengths, desc="Local evidence window length", unit="length")
    for window_length in window_lengths:
        train = partitioned["train"].loc[partitioned["train"]["window_length"] == window_length]
        validation = partitioned["validation"].loc[partitioned["validation"]["window_length"] == window_length]
        test = partitioned["test"].loc[partitioned["test"]["window_length"] == window_length]
        if any(set(frame["label"]) != {0, 1} for frame in (train, validation, test)):
            raise ValueError(f"window_length={window_length} must have rows for both labels in every partition")
        eligibility = {}
        for partition, frame in (("train", train), ("validation", validation), ("test", test)):
            parents = frame[["split", "text_id", "label"]].drop_duplicates()
            eligibility[f"{partition}_eligible_parent_texts"] = len(parents)
            eligibility[f"{partition}_hg_support"] = int(parents["label"].eq(0).sum())
            eligibility[f"{partition}_mg_support"] = int(parents["label"].eq(1).sum())

        model, normalized_features = fit_local_evidence_model(
            windows,
            partitions,
            window_length,
            feature_columns,
            seed,
        )
        print_local_evidence_model_coefficients(model, normalized_features, window_length)
        validation_scored = score_local_evidence_windows(model, validation, normalized_features)
        test_scored = score_local_evidence_windows(model, test, normalized_features)
        scored_frames.extend([validation_scored, test_scored])

        validation_maximum = aggregate_maximum_window(validation_scored)
        maximum_parameters = _fit_document_threshold(validation_maximum["label"], validation_maximum["mg_score"])
        if maximum_parameters["direction"] != "higher_is_mg":
            raise ValueError("maximum-window scores must increase toward MG")
        test_maximum = aggregate_maximum_window(test_scored)

        candidates: list[dict[str, float | str]] = []
        for window_threshold in np.percentile(validation_scored["mg_score"], np.arange(5, 100, 5)):
            validation_fraction = aggregate_evidence_fraction(validation_scored, float(window_threshold))
            document_parameters = _fit_document_threshold(validation_fraction["label"], validation_fraction["mg_score"])
            candidates.append({"window_threshold": float(window_threshold), **document_parameters})
        evidence_parameters = _select_evidence_parameters(candidates)
        test_fraction = aggregate_evidence_fraction(test_scored, float(evidence_parameters["window_threshold"]))

        for aggregation, document_scores, parameters in (
            ("maximum_window", test_maximum, {"window_threshold": np.nan, **maximum_parameters}),
            ("evidence_fraction", test_fraction, evidence_parameters),
        ):
            threshold = float(parameters["threshold"])
            predictions = document_scores.copy()
            predictions["aggregation"] = aggregation
            predictions["window_threshold"] = parameters["window_threshold"]
            predictions["document_threshold"] = threshold
            predictions["score_direction"] = parameters["direction"]
            decision_scores = _decision_scores(predictions["mg_score"], str(parameters["direction"]))
            predictions["prediction"] = (decision_scores >= threshold).astype("int64")
            metrics = evaluate_scores(predictions["label"], decision_scores, threshold)
            summaries.append(
                {
                    "window_length": window_length,
                    "aggregation": aggregation,
                    "window_threshold": parameters["window_threshold"],
                    "document_threshold": threshold,
                    "score_direction": parameters["direction"],
                    "validation_macro_f1": float(parameters["macro_f1"]),
                    "available_parent_texts": sum(
                        eligibility[f"{partition}_eligible_parent_texts"]
                        for partition in ("train", "validation", "test")
                    ),
                    **eligibility,
                    **metrics,
                }
            )
            prediction_frames.append(predictions)

    return pd.DataFrame(summaries), pd.concat(prediction_frames, ignore_index=True), pd.concat(scored_frames, ignore_index=True)
