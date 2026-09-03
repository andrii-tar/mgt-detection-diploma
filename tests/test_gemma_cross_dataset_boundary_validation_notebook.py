from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "gemma_cross_dataset_boundary_validation.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_output_free() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_configures_gemma_cross_dataset_and_partial_semeval_contract() -> None:
    source = _source()
    for fragment in (
        'MODEL = "gemma_3_1b_it"',
        "kaggle_train",
        "mage_train",
        "outfox_chatgpt_train",
        "SEMEVAL_TEXT_IDS = tuple(range(270))",
        "SEMEVAL_TOTAL_TEXTS = 505",
        "270/505 partial coverage",
        "logistic_regression",
        "macro_f1",
        "mean_absolute_token_error",
        "exact_accuracy",
        "within_5_accuracy",
    ):
        assert fragment in source


def test_notebook_does_not_run_pipeline_processing() -> None:
    source = _source()
    for entry_point in (
        "process_pipeline",
        "token_similarity_stage2",
        "load_local_model",
        "run_stage_2",
    ):
        assert entry_point not in source


def test_notebook_loads_or_builds_persisted_local_evidence_windows() -> None:
    source = _source()

    for fragment in (
        "load_or_build_local_evidence_windows",
        "LOCAL_EVIDENCE_CACHE_ROOT",
        'drop(columns="window_text")',
        "recompute=False",
    ):
        assert fragment in source


def test_notebook_shows_dataset_loading_progress() -> None:
    source = _source()

    for fragment in (
        "from tqdm.auto import tqdm",
        'desc="Loading source datasets"',
    ):
        assert fragment in source
