from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from notebooks.utils.report_diagrams_utils import (
    available_metric_rows,
    normalized_confusion_matrix,
    plot_output_root,
    report_output_paths,
    save_report_figure,
    save_report_table,
)


def test_report_output_paths_are_created_below_data_root(tmp_path) -> None:
    paths = report_output_paths(tmp_path)

    assert paths.root == tmp_path / "experiment-results" / "report"
    assert paths.figures.is_dir()
    assert paths.tables.is_dir()


def test_report_output_paths_support_a_separate_run_root(tmp_path) -> None:
    paths = report_output_paths(tmp_path, report_run_id="report-run-2")

    assert paths.root == tmp_path / "experiment-results" / "report" / "report-run-2"
    assert paths.figures.is_dir()
    assert paths.tables.is_dir()


def test_plot_output_root_supports_a_separate_run_root(tmp_path) -> None:
    output = plot_output_root(tmp_path, "classifier-results", plot_run_id="plots-run-2")

    assert output == tmp_path / "plots" / "plots-run-2" / "classifier-results"
    assert output.is_dir()


def test_report_exports_table_and_three_figure_formats(tmp_path) -> None:
    paths = report_output_paths(tmp_path)
    table_path = save_report_table(pd.DataFrame({"score": [0.5]}), paths.tables / "scores.csv")
    figure = plt.figure()
    figure_paths = save_report_figure(figure, paths.figures / "scores")
    plt.close(figure)

    assert table_path.read_text(encoding="utf-8").splitlines() == ["score", "0.5"]
    assert {path.suffix for path in figure_paths} == {".png", ".svg", ".pdf"}
    assert all(path.is_file() for path in figure_paths)


def test_available_metric_rows_excludes_missing_and_non_numeric_values() -> None:
    metrics = pd.DataFrame({"macro_f1": [0.8, None, "not-available", 0.6]})

    result = available_metric_rows(metrics, "macro_f1")

    assert result["macro_f1"].tolist() == [0.8, 0.6]


def test_normalized_confusion_matrix_normalizes_each_true_class() -> None:
    metrics = pd.DataFrame({"tn": [8], "fp": [2], "fn": [1], "tp": [9]})

    result = normalized_confusion_matrix(metrics)

    assert result.index.tolist() == ["HG", "MG"]
    assert result.columns.tolist() == ["HG", "MG"]
    assert result.loc["HG"].tolist() == [0.8, 0.2]
    assert result.loc["MG"].tolist() == [0.1, 0.9]
