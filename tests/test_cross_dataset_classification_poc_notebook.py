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
        'DATASETS = {"Kaggle": "kaggle_train", "Outfox": "outfox_chatgpt_train"}',
        'LLM_MODELS = ["smollm2_360m"]',
        '"Kaggle": ("Kaggle",)', '"Outfox": ("Outfox",)',
        '"Mixed": ("Kaggle", "Outfox")',
        "EVALUATION_REGIMES = TRAINING_REGIMES",
        'CLASSIFIER_NAMES = ("logistic_regression",)',
        "WINDOW_LENGTHS = (32, 64, 128)", "STRIDE = 16", "TOP_K = 20",
        "SEED = 42", "RECOMPUTE_WINDOWS = False",
        "load_or_build_local_evidence_windows(", "split_local_evidence_texts(",
        '"Kaggle"', '"Outfox"', '"Mixed"', "macro_f1", "sns.heatmap",
        "matrix.isna()", "mask=matrix.isna()",
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


def test_poc_notebook_uses_only_supported_local_evidence_apis() -> None:
    source = _source()
    for fragment in (
        "load_or_build_local_evidence_windows(",
        "split_local_evidence_texts(",
        "train_local_evidence_models(",
        "evaluate_local_evidence_datasets(",
        "combine_local_evidence_dataset_results(",
        "evaluate_local_evidence_semeval(",
        "logistic_regression",
        "pd.concat(",
        '"training_regime"',
        '"evaluation_regime"',
    ):
        assert fragment in source


def test_poc_notebook_reports_dataset_specific_transfer_results() -> None:
    source = _source()
    for fragment in (
        '"evaluation_dataset"',
        '"llm"',
        '"training_regime"',
        '"evaluation_regime"',
        '"window_length"',
        '"macro_f1"',
        'pivot(index=["training_regime", "classifier", "window_length"]',
        "sns.heatmap",
    ):
        assert fragment in source


def test_poc_notebook_evaluates_each_training_regime_on_semeval() -> None:
    source = _source()
    for fragment in (
        '"semeval2024_task8_subtask_c_dev"',
        "evaluate_local_evidence_semeval(",
        "semeval_region_summary",
        "semeval_region_evaluations",
        '"training_regime"',
    ):
        assert fragment in source


def test_poc_notebook_excludes_removed_models_settings_and_processing_entry_points() -> None:
    source = _source()
    for fragment in (
        "gemma_3_1b_it", "mage_train", "MAGE", "BayesianConfig",
        "GaussianMixtureNBConfig", "pymc", "REUSE_FITTED_MODELS",
        "SAVE_FITTED_MODELS", "TRUST_FITTED_MODEL_CACHE", "PYMC_AVAILABLE",
        "BAYESIAN_CONFIG", "GAUSSIAN_MIXTURE_CONFIG", "process_pipeline",
        "token_similarity_stage2", "load_local_model", "score_texts_with_model",
        "run_stage_2",
    ):
        assert fragment not in source
