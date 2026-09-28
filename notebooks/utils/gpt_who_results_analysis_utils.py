from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_gpt_who_metrics(
    results_root: Path,
    *,
    run_prefix: str | None = None,
) -> pd.DataFrame:
    """Load available GPT-who metric exports, returning an empty frame if absent."""
    frames = []
    for path in sorted(results_root.glob("*/metrics.parquet")):
        if run_prefix is not None and not path.parent.name.startswith(run_prefix):
            continue
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_gpt_who_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment_name", "scorer_llm", "macro_f1"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"GPT-who metrics are missing columns: {missing}")
    return (
        metrics.groupby(["experiment_name", "scorer_llm"], as_index=False)
        .agg(mean_macro_f1=("macro_f1", "mean"), runs=("macro_f1", "size"))
        .sort_values("mean_macro_f1", ascending=False, ignore_index=True)
    )
