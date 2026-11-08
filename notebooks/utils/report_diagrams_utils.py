from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportOutputPaths:
    root: Path
    figures: Path
    tables: Path


_OUTPUT_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _output_run_id(run_id: str) -> str:
    if not _OUTPUT_RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("output run ID must contain only letters, numbers, '.', '_' or '-'")
    return run_id


def report_output_paths(
    data_root: Path | str,
    report_run_id: str | None = None,
) -> ReportOutputPaths:
    root = Path(data_root) / "experiment-results" / "report"
    if report_run_id is not None:
        root /= _output_run_id(report_run_id)
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return ReportOutputPaths(root=root, figures=figures, tables=tables)


def plot_output_root(
    data_root: Path | str,
    plot_name: str,
    *,
    plot_run_id: str | None = None,
) -> Path:
    root = Path(data_root) / "plots"
    if plot_run_id is not None:
        root /= _output_run_id(plot_run_id)
    output = root / plot_name
    output.mkdir(parents=True, exist_ok=True)
    return output


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


def save_png_figure(figure: plt.Figure, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    return destination


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
