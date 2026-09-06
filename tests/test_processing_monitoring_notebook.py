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
        "from scripts.data_root import resolve_data_root",
        "scan_postprocessing_readiness(",
        "DATA_ROOT",
        "DATA_ROOT = resolve_data_root()",
        "PROCESSED_ROOT",
        "readiness",
        "summary",
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
        "sns.catplot(",
        'hue="split"',
        "artifact_columns",
        "local_evidence_rows",
    ):
        assert fragment in source
    assert "stage1_rows" in source
    assert "stage2_rows" in source


def test_monitoring_notebook_contains_processed_text_heatmap() -> None:
    source = _source()
    for fragment in (
        'values="stage1_rows"',
        "processed_texts = readiness.pivot_table",
        'index=["dataset", "split"]',
        'columns="model"',
        "annot=processed_texts",
        "Processed text count by dataset/split and model",
    ):
        assert fragment in source
