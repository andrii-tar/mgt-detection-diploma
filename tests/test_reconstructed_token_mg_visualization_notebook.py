import json
import ast
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "reconstructed_token_mg_score_visualizations.ipynb"
CHAPTER3_NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "report" / "07_chapter3_figures.ipynb"


def test_reconstructed_token_visualization_notebook_has_clean_outputs_and_fixed_configuration():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert {cell["id"] for cell in code_cells} >= {
        "configuration",
        "load-and-score",
        "select-and-render",
        "dispersion-plots",
        "export-figures",
    }
    assert all(cell["outputs"] == [] for cell in code_cells)
    assert all(cell["execution_count"] is None for cell in code_cells)
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert 'SCORER_LLM = "littlelamb"' in source
    assert "WINDOW_CONFIGURATION = 64" in source
    assert "logistic_regression" in source and "random_forest" in source
    assert "reconstructed_token_mg_score_visualization_utils" in source
    assert "feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100" in source
    assert "TRAINING_REGIME = 'Kaggle+MAGE+Outfox'" in source
    assert "semeval_boundary_errors_from_scored_windows" in source


def test_reconstructed_token_visualization_notebook_adds_dispersion_for_every_selected_example():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    source = "".join(cells["dispersion-plots"]["source"])

    assert "for example in selection.to_dict(orient='records')" in source
    assert "reconstruct_token_score_dispersion" in source
    assert "plot_passage_score_dispersion" in source


def test_reconstructed_token_visualization_notebook_adds_configured_window_boundaries_for_semeval():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    code_source = "\n".join("".join(cell["source"]) for cell in cells.values() if cell["cell_type"] == "code")
    boundary_source = "".join(cells["semeval-window-boundaries"]["source"])

    assert "WINDOW_BOUNDARY_THRESHOLD = 0.5" in code_source
    assert "WINDOW_BOUNDARY_RUN_LENGTH = 3" in code_source
    assert "for example in selection.loc[selection['dataset'].eq('semeval')].to_dict(orient='records')" in boundary_source
    assert "detect_window_run_boundaries" in boundary_source
    assert "plot_window_boundary_profile" in boundary_source
    assert "absolute_token_error" in boundary_source


def test_reconstructed_token_visualization_notebook_documents_dataset_boundary_difference():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")

    assert "gold" in markdown.lower() and "name that llm" in markdown.lower()
    assert "overlap" in markdown.lower()


def test_reconstructed_token_visualization_notebook_code_cells_are_valid_python():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])


def test_chapter3_figures_notebook_has_no_saved_code_outputs():
    notebook = json.loads(CHAPTER3_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
