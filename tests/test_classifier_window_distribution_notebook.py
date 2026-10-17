from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "classifier_window_distributions.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_classifier_window_distribution_notebook_contract() -> None:
    notebook = _notebook()
    source = _source()

    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))

    for fragment in (
        "SCORER_LLM",
        "DATASET",
        "MIN_TOKEN_COUNT",
        "NUM_TEXTS",
        "['token_count'] >= MIN_TOKEN_COUNT",
        "logistic_regression",
        "random_forest",
        "aggregate_window_classifications",
        "mg_window_count",
        "mg_window_percent",
        "selected_metadata",
        "HG",
        "MG",
        "plot_window_distribution",
        "DATASETS",
        "COMBINED_NUM_TEXTS",
        "combined_window_metrics",
        "dataset_colors",
        "split_styles",
        "sns.kdeplot",
        "name_that_llm_train_MG.csv",
        "source_metadata",
        "source_llm",
        "EXPECTED_SOURCE_LLM_COUNT",
        "source_llm_window_metrics",
        "source_llm_colors",
        "Name-That-LLM source LLMs",
    ):
        assert fragment in source
