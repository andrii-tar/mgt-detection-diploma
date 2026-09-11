from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "prepare_and_train_classifiers.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_prepare_and_train_notebook_is_four_cell_output_free_without_definitions() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert notebook["nbformat"] >= 4
    assert len(code_cells) == 4
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        tree = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )


def test_prepare_and_train_notebook_declares_complete_experiment_configuration() -> None:
    source = _source()

    for fragment in (
        "SCORER_LLMS =",
        "DATASETS =",
        "SOURCE_SPLITS =",
        "TRAINING_REGIME =",
        "WINDOW_CONFIGURATIONS =",
        "FEATURE_SETS =",
        "MODEL_FAMILIES =",
        '"gemma_3_1b_it"',
        '"gemma_3_270m_it"',
        '"littlelamb"',
        '"smollm2_360m"',
        '"Kaggle": "kaggle_train"',
        '"MAGE": "mage_train"',
        '"Outfox": "outfox_chatgpt_train"',
        '"BBC": "bbc_news_content"',
        '"BBC": ("HG",)',
        '"Mixed": ("Kaggle", "MAGE", "Outfox")',
        '"Mixed+BBC": ("Kaggle", "MAGE", "Outfox", "BBC")',
        '"logistic_regression"',
    ):
        assert fragment in source


def test_prepare_and_train_notebook_uses_resumable_disk_backed_workflow() -> None:
    source = _source()

    for fragment in (
        "ensure_prepared_partition(",
        "prepared_partition_path(",
        "path.exists()",
        "build_feature_parent_cohorts(",
        "load_training_episode(",
        "fit_feature_vector_model(",
        "save_model_artifact(",
        "export_result_frames(",
        "availability_row(",
        "release_frames(",
        "to_csv(",
    ):
        assert fragment in source
    assert "TRAINING_PARTITIONS = (\"train\", \"validation\")" in source
    assert 'for partition in ("train", "validation", "test")' not in source


def test_prepare_and_train_notebook_records_failures_and_never_runs_processing_or_test_evaluation() -> None:
    source = _source()

    for fragment in (
        "try:",
        "except (FileNotFoundError, KeyError, ValueError, OSError) as error:",
        '"status": "unavailable"',
        "continue",
        "run_summary",
        "training_metrics",
        "evaluation_metrics",
        "evaluation_predictions",
        "evaluation_coverage",
    ):
        assert fragment in source
    for forbidden in (
        "process_pipeline",
        "token_similarity_stage2",
        "load_local_model",
        "run_stage_2",
        "evaluate_feature_vector_model(",
        'partition="test"',
        'partition = "test"',
    ):
        assert forbidden not in source
