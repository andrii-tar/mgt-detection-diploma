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
        "pymc_bayesian_logistic",
        "run_local_evidence_model_comparison",
        "accuracy",
        "macro_f1",
        "visualize_text_similarities",
        "plot_model_score_heatmap",
        "plot_normalized_token_metrics_with_model_evidence",
        "SELECTED_TEXTS",
        "PYMC_AVAILABLE",
        "uv sync --extra bayesian",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_notebook_inspection_uses_test_partition(source: str):
    assert 'partition"].eq("test")' in source
    assert 'partition == "test"' in source or "partition == 'test'" in source


def test_notebook_does_not_import_pymc_unconditionally(source: str):
    tree = ast.parse(source)
    unconditional_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name == "pymc" for alias in node.names)
    ] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "pymc"
    ]
    assert not unconditional_imports
