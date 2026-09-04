from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "cross_dataset_classification_poc.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_poc_notebook_is_output_free() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_poc_notebook_contains_cross_dataset_matrix_contract() -> None:
    source = _source()
    for fragment in (
        "smollm2_360m", "opt_350m", "kaggle_train", "outfox_chatgpt_train", "mage_train",
        "LogisticRegression", "load_metrics(", "train_test_split", '"Kaggle"', '"Outfox"',
        '"MAGE"', '"Mixed"', "macro_f1", "sns.heatmap", "matrix.isna()", "mask=matrix.isna()",
    ):
        assert fragment in source


def test_poc_notebook_does_not_process_new_llm_outputs() -> None:
    source = _source()
    for entry_point in (
        "process_pipeline",
        "token_similarity_stage2",
        "load_local_model",
        "score_texts_with_model",
        "run_stage_2",
    ):
        assert entry_point not in source


def test_poc_notebook_uses_mixed_local_evidence_for_both_processing_models() -> None:
    source = _source()
    for fragment in (
        '"gemma_3_1b_it"',
        '"smollm2_360m"',
        "load_or_build_local_evidence_windows(",
        "split_local_evidence_texts(",
        "run_local_evidence_model_comparison(",
        "training_input_fingerprint(",
        "logistic_regression",
        "gaussian_nb",
        "gaussian_mixture_nb",
        "pymc_bayesian_logistic",
        '"Mixed"',
        "pd.concat(",
        'split="test"',
    ):
        assert fragment in source


def test_poc_notebook_reports_dataset_specific_transfer_results() -> None:
    source = _source()
    for fragment in (
        '"evaluation_dataset"',
        '"processing_model"',
        '"window_length"',
        '"macro_f1"',
        'pivot(index=["classifier", "window_length"]',
        "sns.heatmap",
    ):
        assert fragment in source
