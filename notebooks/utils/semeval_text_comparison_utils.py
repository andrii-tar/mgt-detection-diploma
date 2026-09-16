from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.utils.local_evidence_comparison_utils import expand_model_scores_to_tokens


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


def filter_semeval_configuration(
    frame: pd.DataFrame,
    *,
    scorer_llm: str,
    feature_set: str,
    model_family: str = "logistic_regression",
    evaluation_dataset: str | None = None,
    semeval_split: str | None = None,
) -> pd.DataFrame:
    _require(frame, ("scorer_llm", "feature_set", "text_id", "window_length"))
    result = frame.loc[frame["scorer_llm"].eq(scorer_llm) & frame["feature_set"].eq(feature_set)].copy()
    if "model_family" in result:
        result = result.loc[result["model_family"].eq(model_family)]
    if "model" in result:
        result = result.loc[result["model"].eq(model_family)]
    if evaluation_dataset is not None:
        _require(result, ("evaluation_dataset",))
        result = result.loc[result["evaluation_dataset"].eq(evaluation_dataset)]
    if semeval_split is not None:
        _require(result, ("semeval_split",))
        result = result.loc[result["semeval_split"].eq(semeval_split)]
    return result.reset_index(drop=True)


def select_semeval_comparison_texts(
    evaluations: pd.DataFrame,
    *,
    scorer_llm: str = "littlelamb",
    feature_set: str = "extended",
    model_family: str = "logistic_regression",
    evaluation_dataset: str | None = None,
    semeval_split: str | None = None,
    count: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    filtered = filter_semeval_configuration(
        evaluations,
        scorer_llm=scorer_llm,
        feature_set=feature_set,
        model_family=model_family,
        evaluation_dataset=evaluation_dataset,
        semeval_split=semeval_split,
    )
    _require(filtered, ("mg_f1",))
    if filtered.empty:
        return pd.DataFrame(columns=["group", "text_id", "aggregate_mg_f1"])
    filtered["mg_f1"] = pd.to_numeric(filtered["mg_f1"], errors="coerce")
    filtered["window_length"] = pd.to_numeric(filtered["window_length"], errors="coerce").astype("Int64")
    pivot = filtered.pivot_table(index="text_id", columns="window_length", values="mg_f1", aggfunc="mean")
    pivot["aggregate_mg_f1"] = pivot.mean(axis=1, skipna=True)
    pivot = pivot.dropna(subset=["aggregate_mg_f1"]).sort_index()
    windows = sorted(column for column in pivot.columns if column != "aggregate_mg_f1")
    trace = pivot.rename(columns={window: f"mg_f1_window_{window}" for window in windows})
    base = trace.reset_index()
    best = base.sort_values(["aggregate_mg_f1", "text_id"], ascending=[False, True]).head(int(count)).assign(group="best")
    worst = base.sort_values(["aggregate_mg_f1", "text_id"], ascending=[True, True]).head(int(count)).assign(group="worst")
    random = base.sample(n=min(int(count), len(base)), random_state=int(seed)).assign(group="random")
    selected = pd.concat([best, worst, random], ignore_index=True)
    selected.insert(0, "feature_set", feature_set)
    selected.insert(0, "scorer_llm", scorer_llm)
    return selected[["scorer_llm", "feature_set", "group", "text_id", "aggregate_mg_f1", *[f"mg_f1_window_{window}" for window in windows]]]


def _text_rows(frame: pd.DataFrame, text_id: int, required: Iterable[str]) -> pd.DataFrame:
    _require(frame, required)
    rows = frame.loc[frame["text_id"].eq(text_id)].copy()
    if rows.empty:
        raise ValueError(f"no rows found for text id {text_id}")
    return rows


def plot_boundary_and_regions(predictions: pd.DataFrame, *, text_id: int, selection_group: str, title_prefix: str = "SemEval") -> plt.Figure:
    rows = _text_rows(predictions, text_id, ("text_id", "window_length", "gold_token_boundary", "predicted_token_boundary"))
    windows = sorted(rows["window_length"].unique())
    token_count = int(rows.get("token_count", pd.Series(dtype=float)).max()) if "token_count" in rows else 0
    if not token_count:
        token_count = int(pd.concat([rows["gold_token_boundary"], rows["predicted_token_boundary"]]).max())
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for window in windows:
        current = rows.loc[rows["window_length"].eq(window)]
        row = current.iloc[0]
        signal = np.zeros(token_count + 1, dtype="float64")
        starts = current.get("predicted_mg_region_start", pd.Series(dtype=float)).dropna()
        stops = current.get("predicted_mg_region_stop", pd.Series(dtype=float)).dropna()
        if starts.empty and {"start_token", "stop_token", "label"}.issubset(current.columns):
            mg_rows = current.loc[current["label"].astype(str).eq("MG")]
            starts = mg_rows["start_token"]
            stops = mg_rows["stop_token"]
        if not starts.empty and len(starts) == len(stops):
            for start, stop in zip(starts, stops):
                signal[max(0, int(start)):min(token_count, int(stop)) + 1] = 1.0
        axis.step(np.arange(token_count + 1), signal, where="post", label=f"window={window}")
        for predicted in current["predicted_token_boundary"].dropna():
            axis.axvline(float(predicted), color="tab:red", linestyle=":", alpha=.35)
    gold_boundary = float(rows["gold_token_boundary"].dropna().iloc[0])
    axis.axvspan(gold_boundary, token_count, color="tab:green", alpha=.12, label="gold MG region")
    axis.axvline(gold_boundary, color="tab:green", linestyle="--", label="gold boundary")
    axis.set_xlim(0, token_count)
    axis.set_ylim(-.05, 1.05)
    axis.set_xlabel("Token position")
    axis.set_ylabel("Predicted MG region")
    axis.set_title(f"{title_prefix}: text {text_id} ({selection_group}) boundary comparison")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=min(4, len(windows) + 2))
    return figure


def plot_window_level_scores(scored_windows: pd.DataFrame, *, text_id: int, gold_boundary: int, selection_group: str, title_prefix: str = "SemEval") -> plt.Figure:
    rows = _text_rows(scored_windows, text_id, ("text_id", "window_length", "start_token", "stop_token", "mg_score"))
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for window, current in rows.groupby("window_length", sort=True):
        current = current.sort_values("start_token")
        midpoint = (current["start_token"] + current["stop_token"]) / 2
        axis.plot(midpoint, current["mg_score"], marker="o", label=f"window={window}")
    axis.axvline(gold_boundary, color="tab:green", linestyle="--", label="gold boundary")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Token position")
    axis.set_ylabel("Window MG score")
    axis.set_title(f"{title_prefix}: text {text_id} ({selection_group}) window-level MG scores")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)
    return figure


def plot_reconstructed_token_scores(scored_windows: pd.DataFrame, *, text_id: int, token_count: int, gold_boundary: int, selection_group: str, title_prefix: str = "SemEval") -> plt.Figure:
    rows = _text_rows(scored_windows, text_id, ("model", "window_length", "start_token", "stop_token", "mg_score"))
    windows = sorted(rows["window_length"].unique())
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for window in windows:
        current = rows.loc[rows["window_length"].eq(window)]
        token_scores = expand_model_scores_to_tokens(current, token_count)
        scores = token_scores["mg_score"].to_numpy(dtype="float64")
        positions = token_scores["token_position"].to_numpy()
        axis.plot(positions, scores, label=f"window={window}")
    axis.axvspan(gold_boundary, token_count, color="tab:green", alpha=.12, label="gold MG region")
    axis.axvline(gold_boundary, color="tab:green", linestyle="--", label="gold boundary")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Token position")
    axis.set_ylabel("MG score")
    axis.set_title(f"{title_prefix}: text {text_id} ({selection_group}) token-level MG score comparison")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=min(4, len(windows) + 2))
    return figure
