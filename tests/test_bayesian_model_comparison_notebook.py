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
        "LR, GaussianNB, and GaussianMixtureNB",
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


def test_notebook_configures_fitted_model_cache(source: str):
    assert "SAVE_FITTED_MODELS = False" in source
    assert "REUSE_FITTED_MODELS = True" in source
    assert "TRUST_FITTED_MODEL_CACHE = False" in source
    assert 'MODEL_CACHE_ROOT = PROCESSED_ROOT / f"{DATASET}__{MODEL}" / "local_evidence" / "models"' in source
    assert "MODEL_CACHE_MANIFEST" in source
    for field in (
        '"dataset": DATASET',
        '"model": MODEL',
        '"features": LOCAL_FEATURE_COLUMNS',
        '"seed": SEED',
        '"window_lengths": WINDOW_LENGTHS',
        '"model_names": AVAILABLE_MODELS',
        '"bayesian_config": asdict(BAYESIAN_CONFIG)',
        '"gaussian_mixture_config": asdict(GAUSSIAN_MIXTURE_CONFIG)',
        'MODEL_CACHE_MANIFEST["training_input_fingerprint"] = training_input_fingerprint(windows, partitions)',
    ):
        assert field in source


def test_notebook_wires_cache_configuration_into_comparison(source: str):
    tree = ast.parse(source)
    comparison_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_local_evidence_model_comparison"
    ]
    assert len(comparison_calls) == 1
    keyword_values = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in comparison_calls[0].keywords
    }
    assert {
        "cache_root": "MODEL_CACHE_ROOT",
        "cache_manifest": "MODEL_CACHE_MANIFEST",
        "reuse_fitted_models": "REUSE_FITTED_MODELS",
        "save_fitted_models": "SAVE_FITTED_MODELS",
        "trust_fitted_models": "TRUST_FITTED_MODEL_CACHE",
    }.items() <= keyword_values.items()


def test_notebook_groups_overall_metrics_and_covers_configured_lengths(source: str):
    assert 'groupby(["model", "window_length", "aggregation"])' in source
    assert "for window_length in WINDOW_LENGTHS" in source
    assert "drop_duplicates([\"split\", \"text_id\"])" in source


def test_notebook_preserves_local_evidence_plot_contract(source: str):
    assert "plot_normalized_token_metrics_with_local_evidence" in source
    assert "display(plot_normalized_token_metrics_with_local_evidence(" in source


def test_notebook_passes_one_labeled_model_to_legacy_local_evidence_plot(source: str):
    assert 'LOCAL_EVIDENCE_MODEL = "logistic_regression"' in source
    assert 'local_evidence_windows = text_windows.loc[' in source
    assert 'text_windows["model"].eq(LOCAL_EVIDENCE_MODEL)' in source
    assert 'display(f"Local evidence model: {LOCAL_EVIDENCE_MODEL}")' in source
    assert 'token_summary, local_evidence_windows' in source


def test_notebook_calls_focused_plot_with_lr_filtered_windows(source: str):
    tree = ast.parse(source)
    focused_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "token_candidate_similarity_heatmaps_utils"
        and any(
            alias.name == "plot_focused_token_metrics_with_local_evidence"
            for alias in node.names
        )
    ]
    assert focused_imports

    focused_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plot_focused_token_metrics_with_local_evidence"
    ]
    assert len(focused_calls) == 1
    focused_call = focused_calls[0]
    assert [ast.unparse(argument) for argument in focused_call.args] == [
        "token_summary",
        "local_evidence_windows",
        "len(token_labels)",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in focused_call.keywords
    } == {"window": "SMOOTHING_WINDOW"}

    visualization_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "visualize_text_similarities"
            for child in ast.walk(node)
        )
    )
    window_loop = next(
        node
        for node in ast.walk(visualization_loop)
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "local_evidence_windows"
                for target in child.targets
            )
            for child in node.body
        )
    )
    filter_index = next(
        index
        for index, child in enumerate(window_loop.body)
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "local_evidence_windows"
            for target in child.targets
        )
    )
    call_index = next(
        index
        for index, child in enumerate(window_loop.body)
        if focused_call in ast.walk(child)
    )
    assert call_index > filter_index


def test_notebook_validates_kaggle_trained_models_on_semeval_boundaries(source: str):
    required_fragments = (
        'SEMEVAL_DATASET = "semeval2024_task8_subtask_c_dev"',
        "SEMEVAL_PREPARED_PATH",
        "SEMEVAL_MG_SCORE_THRESHOLD = 0.5",
        "fit_local_evidence_classifier",
        "score_local_evidence_classifier",
        "predict_mg_regions",
        "build_gold_mg_regions",
        "evaluate_mg_regions",
        "summarize_mg_region_evaluations",
        "boundary_word_index",
        "mg_f1",
        "predicted_mg_region_count",
        "fitted_models = {}",
        "fitted_models=fitted_models",
        "load_token_labels",
        "show_progress=True",
        "Scoring SemEval with trained models",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_notebook_configures_multi_region_detection(source: str):
    assert "SEMEVAL_SMOOTHING_WINDOW = 5" in source
    assert "SEMEVAL_MINIMUM_REGION_LENGTH = 3" in source

    tree = ast.parse(source)
    prediction_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "predict_mg_regions"
    ]
    assert len(prediction_calls) == 1
    keyword_values = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in prediction_calls[0].keywords
    }
    assert {
        "smoothing_window": "SEMEVAL_SMOOTHING_WINDOW",
        "minimum_region_length": "SEMEVAL_MINIMUM_REGION_LENGTH",
    }.items() <= keyword_values.items()


def test_notebook_aligns_gold_annotations_to_available_processed_tokens(source: str):
    assert 'semeval_annotations["id"].isin(semeval_token_labels)' in source
    assert "available_semeval_annotations" in source


def test_notebook_visualizes_gold_and_predicted_text_zones(source: str):
    for fragment in (
        "token_labels[prediction.text_id]",
        "gold_zone_axis",
        "predicted_zone_axis",
        'gold_zone_axis.set_ylabel("Gold")',
        'predicted_zone_axis.set_ylabel("Predicted")',
        "HG-to-MG",
        "MG-to-HG",
    ):
        assert fragment in source


def test_notebook_plots_raw_and_smoothed_token_mg_scores(source: str):
    assert 'token_scores["mg_score"]' in source
    assert 'label="raw token MG score"' in source
    assert 'label="smoothed token MG score"' in source
    assert 'marker="."' in source
    assert 'zorder=3' in source
