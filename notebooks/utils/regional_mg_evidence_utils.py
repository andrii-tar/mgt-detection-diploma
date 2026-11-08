"""Regional evidence aggregation for local MG/HG scoring experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


REQUIRED_WINDOW_COLUMNS = {"start_token", "stop_token", "window_length"}


@dataclass
class _ProbabilityCalibrator:
    model: LogisticRegression

    def predict(self, raw_scores: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(values)[:, 1]


def fit_probability_calibrator(
    raw_scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> _ProbabilityCalibrator:
    """Fit a monotonic logistic calibration map on held-out window scores."""

    scores = np.asarray(raw_scores, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or targets.ndim != 1 or len(scores) != len(targets):
        raise ValueError("raw_scores and labels must be one-dimensional and equally sized")
    if len(np.unique(targets)) != 2:
        raise ValueError("calibration requires both HG and MG labels")

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), targets)
    return _ProbabilityCalibrator(model)


def _validate_windows(windows: pd.DataFrame, score_column: str) -> None:
    missing = (REQUIRED_WINDOW_COLUMNS | {score_column}) - set(windows.columns)
    if missing:
        raise ValueError(f"window scores are missing columns: {sorted(missing)}")
    if (windows["stop_token"] <= windows["start_token"]).any():
        raise ValueError("window stop_token must be greater than start_token")


def _project_windows(
    windows: pd.DataFrame,
    token_count: int,
    score_column: str,
) -> pd.DataFrame:
    _validate_windows(windows, score_column)
    if token_count < 0:
        raise ValueError("token_count must be non-negative")

    values: list[list[float]] = [[] for _ in range(token_count)]
    lengths: list[list[int]] = [[] for _ in range(token_count)]
    for row in windows.itertuples(index=False):
        start = max(0, int(row.start_token))
        stop = min(token_count, int(row.stop_token))
        for position in range(start, stop):
            values[position].append(float(getattr(row, score_column)))
            lengths[position].append(int(row.window_length))

    records = []
    for position, token_values in enumerate(values):
        scores = np.asarray(token_values, dtype=float)
        records.append(
            {
                "token_position": position,
                "mean_score": float(np.mean(scores)) if len(scores) else np.nan,
                "max_score": float(np.max(scores)) if len(scores) else np.nan,
                "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                "support_count": len(scores),
                "window_lengths": tuple(sorted(set(lengths[position]))),
            }
        )
    return pd.DataFrame.from_records(records)


def build_multiscale_token_evidence(
    windows: pd.DataFrame,
    token_count: int,
    *,
    score_column: str = "mg_score",
    window_threshold: float = 0.5,
) -> pd.DataFrame:
    """Project overlapping windows to tokens and quantify scale agreement."""

    evidence = _project_windows(windows, token_count, score_column)
    per_length_scores: list[list[float]] = [[] for _ in range(token_count)]
    per_length_high: list[list[bool]] = [[] for _ in range(token_count)]
    for row in windows.itertuples(index=False):
        start = max(0, int(row.start_token))
        stop = min(token_count, int(row.stop_token))
        for position in range(start, stop):
            per_length_scores[position].append(float(getattr(row, score_column)))
            per_length_high[position].append(float(getattr(row, score_column)) >= window_threshold)

    length_agreement = []
    evidence_fraction = []
    robust_score = []
    for position in range(token_count):
        scores = per_length_scores[position]
        high = per_length_high[position]
        if not scores:
            length_agreement.append(np.nan)
            evidence_fraction.append(np.nan)
            robust_score.append(np.nan)
            continue

        length_means = windows.loc[
            windows["start_token"].le(position) & windows["stop_token"].gt(position)
        ].groupby("window_length")[score_column].mean()
        agreement = float((length_means >= window_threshold).mean())
        mean_score = float(np.mean(scores))
        length_agreement.append(agreement)
        evidence_fraction.append(float(np.mean(high)))
        robust_score.append(mean_score * agreement)

    evidence["length_agreement"] = length_agreement
    evidence["evidence_fraction"] = evidence_fraction
    evidence["robust_score"] = robust_score
    return evidence


def extract_hysteresis_regions(
    evidence: pd.DataFrame,
    *,
    score_column: str = "robust_score",
    start_threshold: float = 0.75,
    continue_threshold: float = 0.55,
    minimum_length: int = 3,
    merge_gap: int = 1,
) -> pd.DataFrame:
    """Extract contiguous evidence regions using separate start/continue thresholds."""

    if score_column not in evidence or "token_position" not in evidence:
        raise ValueError(f"evidence must contain token_position and {score_column}")
    if minimum_length < 1 or merge_gap < 0:
        raise ValueError("minimum_length must be positive and merge_gap must be non-negative")

    scores = evidence.sort_values("token_position").reset_index(drop=True)
    candidates: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for row in scores.itertuples(index=False):
        position = int(row.token_position)
        value = float(getattr(row, score_column)) if pd.notna(getattr(row, score_column)) else -np.inf
        if start is None:
            if value >= start_threshold:
                start = position
                last_active = position
            continue
        if value >= continue_threshold:
            last_active = position
        elif position - int(last_active) > merge_gap:
            candidates.append((start, int(last_active) + 1))
            start = position if value >= start_threshold else None
            last_active = position if start is not None else None
    if start is not None:
        candidates.append((start, int(last_active) + 1))

    records = []
    for region_start, region_stop in candidates:
        region = evidence[
            evidence["token_position"].between(region_start, region_stop - 1)
        ]
        if region_stop - region_start < minimum_length:
            continue
        records.append(
            {
                "start_token": region_start,
                "stop_token": region_stop,
                "token_count": region_stop - region_start,
                "mean_score": float(region[score_column].mean()),
                "max_score": float(region[score_column].max()),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["start_token", "stop_token", "token_count", "mean_score", "max_score"],
    )


def summarize_window_contributions(
    scored_windows: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Summarize signed model contributions for explaining suspicious windows."""

    contribution_columns = [
        column for column in scored_windows.columns if column.startswith("contribution__")
    ]
    records = []
    for column in contribution_columns:
        feature = column.removeprefix("contribution__")
        records.append(
            {
                "feature": feature,
                "mean_signed_contribution": float(scored_windows[column].mean()),
                "mean_absolute_contribution": float(scored_windows[column].abs().mean()),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values("mean_signed_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
