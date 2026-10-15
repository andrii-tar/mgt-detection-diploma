from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportOutputPaths:
    root: Path
    figures: Path
    tables: Path


def report_output_paths(data_root: Path | str) -> ReportOutputPaths:
    root = Path(data_root) / "experiment-results" / "report"
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return ReportOutputPaths(root=root, figures=figures, tables=tables)


def save_report_table(frame: pd.DataFrame, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def save_report_figure(figure: plt.Figure, path: Path | str) -> tuple[Path, Path, Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(destination.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    for figure_path in paths:
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    return paths


def available_metric_rows(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in metrics.columns:
        raise ValueError(f"metrics are missing required column: {metric}")
    result = metrics.copy()
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result.dropna(subset=[metric]).reset_index(drop=True)


def normalized_confusion_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"tn", "fp", "fn", "tp"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics are missing confusion columns: {missing}")
    counts = metrics.loc[:, sorted(required)].apply(pd.to_numeric, errors="coerce").sum()
    matrix = pd.DataFrame(
        [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
        index=["HG", "MG"],
        columns=["HG", "MG"],
        dtype=float,
    )
    totals = matrix.sum(axis=1).replace(0, pd.NA)
    return matrix.div(totals, axis=0).fillna(0.0)
