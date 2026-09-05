from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "processing_monitoring.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_monitoring_notebook_is_output_free() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_monitoring_notebook_contains_scan_and_display_contract() -> None:
    source = _source()
    for fragment in (
        "from scripts.experiment_config import DATASETS, MODEL_IDS, build_datasets",
        "from scripts.data_root import resolve_data_root",
        "scan_processing_counts(",
        "TEXT_DETECTION_DATA",
        "DATA_ROOT",
        "DATA_ROOT = resolve_data_root()",
        "build_datasets(DATA_ROOT)",
        "processing_counts",
        "stage_matrix",
        "pivot_table(",
        "display(",
    ):
        assert fragment in source


def test_monitoring_notebook_does_not_process_model_outputs() -> None:
    source = _source()
    for entry_point in (
        "process_pipeline",
        "token_similarity_stage2",
        "load_local_model",
        "score_texts_with_model",
    ):
        assert entry_point not in source


def test_monitoring_notebook_preserves_split_counts_in_progress_plots() -> None:
    source = _source()
    for fragment in (
        'processing_counts.groupby(["dataset", "model", "split", "stage"]',
            "sns.catplot(",
            'hue="split"',
            "stage2_counts",
            'x="dataset"',
    ):
        assert fragment in source
    assert ".sum()" not in source
    assert ".mean()" not in source


def test_progress_plot_uses_a_four_column_grid_per_model() -> None:
    source = _source()
    progress_start = source.index("progress_plot = sns.catplot(")
    progress_end = source.index("stage2_counts =", progress_start)
    progress_source = source[progress_start:progress_end]
    assert 'col="dataset"' in progress_source
    assert "col_wrap=4" in progress_source


def test_monitoring_notebook_renders_progress_charts_sequentially() -> None:
    source = _source()
    assert "for model in sorted(split_stage_counts[\"model\"].unique())" in source
    assert "plt.close(progress_plot.fig)" in source
    assert "for model in sorted(stage2_counts[\"model\"].unique())" in source


def test_progress_charts_are_grouped_in_four_column_model_grids() -> None:
    source = _source()
    assert "for model in sorted(split_stage_counts[\"model\"].unique())" in source
    assert 'col="dataset"' in source
    assert "col_wrap=4" in source


def test_stage2_summary_charts_are_grouped_by_model() -> None:
    source = _source()
    assert "for model in sorted(stage2_counts[\"model\"].unique())" in source
    assert 'x="dataset"' in source
    assert 'y="processed_texts"' in source
    assert 'hue="split"' in source
