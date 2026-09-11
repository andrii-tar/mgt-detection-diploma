from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOKS_DIRECTORY = Path(__file__).parents[1] / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIRECTORY / "prepare_and_train_classifiers.ipynb"


def _notebook(name: str = "prepare_and_train_classifiers.ipynb") -> dict:
    return json.loads((NOTEBOOKS_DIRECTORY / name).read_text(encoding="utf-8"))


def _source(name: str = "prepare_and_train_classifiers.ipynb") -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook(name)["cells"]
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


def test_evaluate_notebook_is_output_free_and_has_no_definitions() -> None:
    notebook = _notebook("evaluate_classifiers.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert notebook["nbformat"] >= 4
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        tree = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )


def test_evaluate_notebook_uses_metadata_driven_trusted_transfer_workflow() -> None:
    source = _source("evaluate_classifiers.ipynb")

    for fragment in (
        "discover_model_artifacts(",
        "load_model_metadata(",
        "load_model_artifact(",
        "trust_artifact=True",
        "metadata[\"scorer_llm\"]",
        "metadata[\"feature_columns\"]",
        "evaluation_window",
        "TRAINING_REGIMES",
        "EVALUATION_REGIMES",
        'for partition in ("validation", "test")',
        "evaluate_feature_vector_model(",
        "release_frames(",
        "export_result_frames(",
        "evaluation_metrics",
        "evaluation_predictions",
        "evaluation_coverage",
        "run_summary",
    ):
        assert fragment in source


def test_evaluate_notebook_records_expected_failures_and_reraises_unexpected_errors() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert "except (FileNotFoundError, KeyError, ValueError, OSError) as error:" in source
    assert '"status": "unavailable"' in source
    assert "continue" in source
    assert "except Exception:" in source
    assert "raise" in source
    for forbidden in ("fit_feature_vector_model(", "save_model_artifact(", "process_pipeline", "token_similarity_stage2"):
        assert forbidden not in source
