import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "report" / "complete_classifier_analysis.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_complete_analysis_notebook_is_output_free():
    notebook = _notebook()

    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_complete_analysis_notebook_declares_resumable_smoke_pipeline():
    source = _source()

    for required in (
        "SMOKE_MODE = True",
        "SMOKE_MODEL_FAMILY",
        "SMOKE_TRAINING_WINDOW",
        "select_smoke_artifacts(",
        "load_prediction_artifacts(",
        "INTERMEDIATE_ROOT",
        "OUTPUT_ROOT",
        "run_step(",
        "artifact_inventory",
        "training_composition",
        "binary_metrics",
        "calibration",
        "confusion_matrices",
        "score_curves",
        "threshold_sweep",
        "bounded_prefix",
        "robustness",
        "semeval_localization",
        "run_summary.json",
    ):
        assert required in source


def test_complete_analysis_notebook_reports_progress_for_long_operations():
    source = _source()

    assert "from tqdm.auto import tqdm" in source
    assert "progress_description=\"Loading prediction artifacts\"" in source
    assert "tqdm(np.linspace(0.05, 0.95, 19)" in source
    assert "for step_id, table_name, output_name in tqdm((" in source


def test_complete_analysis_notebook_does_not_train_or_score_models():
    source = _source()

    for forbidden in (
        "fit_feature_vector_model(",
        "score_texts(",
        "AutoModelForCausalLM.from_pretrained(",
        "save_model_artifact(",
    ):
        assert forbidden not in source
