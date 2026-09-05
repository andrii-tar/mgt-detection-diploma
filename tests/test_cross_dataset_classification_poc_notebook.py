from __future__ import annotations

import ast
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


def _tree() -> ast.Module:
    return ast.parse(_source())


def _calls(tree: ast.AST, function_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _constant_tuple(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    if not all(isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts):
        return None
    return tuple(element.value for element in node.elts)


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


def test_poc_notebook_splits_each_source_dataset_deterministically() -> None:
    tree = _tree()
    dataset_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "DATASETS"
    ]

    assert len(dataset_loops) == 1
    split_calls = _calls(dataset_loops[0], "split_local_evidence_texts")
    assert len(split_calls) == 1
    assert any(
        keyword.arg == "seed"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "SEED"
        for keyword in split_calls[0].keywords
    )


def test_poc_notebook_executes_all_training_and_evaluation_regimes() -> None:
    tree = _tree()
    training_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "TRAINING_REGIMES"
    ]
    evaluation_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "EVALUATION_REGIMES"
    ]

    assert len(training_loops) == 1
    assert len(evaluation_loops) == 1
    assert _calls(evaluation_loops[0], "evaluate_local_evidence_datasets")
    assert _calls(training_loops[0], "combine_local_evidence_dataset_results")
    assert any(
        isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.Eq) for operator in node.ops)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "Mixed"
            for comparator in node.comparators
        )
        for node in ast.walk(evaluation_loops[0])
    )


def test_poc_notebook_propagates_classifiers_and_labels_every_result_frame() -> None:
    tree = _tree()
    training_calls = _calls(tree, "train_local_evidence_models")
    assert len(training_calls) == 1
    assert any(
        keyword.arg == "classifier_names"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "CLASSIFIER_NAMES"
        for keyword in training_calls[0].keywords
    )

    result_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and _constant_tuple(node.iter)
        == ("metrics", "predictions", "scored_windows", "coverage")
    ]
    assert len(result_loops) == 2
    for result_loop in result_loops:
        assigned_labels = {
            node.targets[0].slice.value
            for node in ast.walk(result_loop)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and isinstance(node.targets[0].slice.value, str)
        }
        assert {"llm", "training_regime", "evaluation_regime"} <= assigned_labels


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
