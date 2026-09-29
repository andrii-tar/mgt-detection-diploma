from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


SCORED_WINDOW_KEY_COLUMNS = (
    "dataset",
    "split",
    "text_id",
    "model",
    "window_length",
    "start_token",
    "stop_token",
)


def validate_scored_window_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"scored-window frame is missing required columns: {missing}")


def merge_scored_window_exports(
    existing: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    key_columns: Sequence[str] = SCORED_WINDOW_KEY_COLUMNS,
) -> pd.DataFrame:
    validate_scored_window_columns(existing, key_columns)
    validate_scored_window_columns(generated, key_columns)
    if set(existing.columns) != set(generated.columns):
        missing = sorted(set(existing.columns).difference(generated.columns))
        extra = sorted(set(generated.columns).difference(existing.columns))
        raise ValueError(
            "scored-window frame columns do not match: "
            f"missing={missing}, extra={extra}"
        )

    ordered_columns = list(existing.columns)
    generated = generated.loc[:, ordered_columns].copy()
    if generated.empty:
        return existing.loc[:, ordered_columns].copy()

    generated_keys = generated.loc[:, list(key_columns)].drop_duplicates()
    retained = existing.merge(
        generated_keys.assign(_generated_key=True),
        on=list(key_columns),
        how="left",
        sort=False,
    )
    retained = retained.loc[retained["_generated_key"].isna(), ordered_columns]
    merged = pd.concat([retained, generated], ignore_index=True)
    merged = merged.drop_duplicates(subset=list(key_columns), keep="last")
    return merged.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
