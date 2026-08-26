from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "bayesian_model_comparison.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK_PATH.is_file()
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_valid_output_free_nbformat(notebook: dict):
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_code_cells_have_no_function_definitions(notebook: dict):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )


def test_notebook_contains_comparison_contract(source: str):
    required_fragments = (
        "logistic_regression",
        "gaussian_nb",
        "gaussian_mixture_nb",
        "pymc_bayesian_logistic",
        "GaussianMixtureNBConfig",
        "GAUSSIAN_MIXTURE_COMPONENTS",
        "gaussian_mixture_config=GAUSSIAN_MIXTURE_CONFIG",
        "run_local_evidence_model_comparison",
        "accuracy",
        "macro_f1",
        "visualize_text_similarities",
        "plot_model_score_heatmap",
        "plot_normalized_token_metrics_with_model_evidence",
        "SELECTED_TEXTS",
        "PYMC_AVAILABLE",
        "uv sync --extra bayesian",
        "groupby([\"model\", \"window_length\", \"aggregation\"])",
        "np.isfinite",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_notebook_inspection_uses_test_partition(source: str):
    assert 'partition"].eq("test")' in source
    assert 'partition == "test"' in source or "partition == 'test'" in source


def test_notebook_does_not_import_pymc_unconditionally(source: str):
    tree = ast.parse(source)
    pymc_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name == "pymc" for alias in node.names)
    ]
    assert len(pymc_imports) == 1
    guarded_try = next(
        parent
        for parent in ast.walk(tree)
        if isinstance(parent, ast.Try) and pymc_imports[0] in parent.body
    )
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "ModuleNotFoundError"
        for handler in guarded_try.handlers
    )


def test_notebook_runs_one_comparison_with_mandatory_deterministic_models(source: str):
    tree = ast.parse(source)
    comparison_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_local_evidence_model_comparison"
    ]
    assert len(comparison_calls) == 1
    assert 'MODEL_NAMES = ("logistic_regression", "gaussian_nb", "gaussian_mixture_nb", "pymc_bayesian_logistic")' in source
    assert "AVAILABLE_MODELS" in source


def test_notebook_groups_overall_metrics_and_covers_configured_lengths(source: str):
    assert 'groupby(["model", "window_length", "aggregation"])' in source
    assert "for window_length in WINDOW_LENGTHS" in source
    assert "drop_duplicates([\"split\", \"text_id\"])" in source


def test_notebook_uses_only_multi_model_plot_for_model_scores(source: str):
    assert "plot_normalized_token_metrics_with_local_evidence(" not in source
