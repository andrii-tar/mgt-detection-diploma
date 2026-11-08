from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def permute_feature_by_parent(frame: pd.DataFrame, *, feature: str, seed: int) -> pd.DataFrame:
    required = {"split", "text_id", feature}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    parent_columns = [column for column in ("dataset", "split", "text_id") if column in frame]
    parent_values = frame.loc[:, parent_columns].drop_duplicates()
    values = frame.drop_duplicates(parent_columns).loc[:, [*parent_columns, feature]]
    shuffled = np.random.default_rng(seed).permutation(values[feature].to_numpy())
    shuffled_values = values.loc[:, parent_columns].copy()
    shuffled_values[feature] = shuffled
    result = frame.copy()
    permuted = result.drop(columns=feature).merge(
        shuffled_values,
        on=parent_columns,
        how="left",
        validate="many_to_one",
    )
    return permuted.loc[:, frame.columns]


def metric_degradation(*, metric: str, baseline: float, permuted: float) -> float:
    if metric in {"macro_f1", "mg_region_f1"}:
        return baseline - permuted
    if metric == "boundary_mae":
        return permuted - baseline
    raise ValueError(f"unsupported metric: {metric}")


def importance_run_path(data_root: Path | str, *, run_id: str) -> Path:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    return Path(data_root) / "experiment-results" / "classification" / "feature-importance" / run_id


def summarize_importance(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"configuration_id", "metric", "feature", "degradation"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations are missing required columns: {sorted(missing)}")
    summary = (
        observations.groupby(["configuration_id", "metric", "feature"], as_index=False)["degradation"]
        .agg(
            mean_degradation="mean",
            lower_95=lambda values: values.quantile(0.025),
            upper_95=lambda values: values.quantile(0.975),
            repeat_count="count",
        )
    )
    summary["rank"] = summary.groupby(["configuration_id", "metric"])["mean_degradation"].rank(
        method="min", ascending=False,
    ).astype("int64")
    return summary
