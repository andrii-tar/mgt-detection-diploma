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


def test_notebook_is_valid_output_free_json_without_code_definitions() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 3
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        tree = ast.parse("".join(cell["source"]))
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))


def test_notebook_exposes_three_matrix_blocks_and_exact_configuration() -> None:
    source = _source()
    for fragment in (
        "BLOCK 1: WHOLE-TEXT SCALAR METRICS",
        "BLOCK 2: LOCAL EVIDENCE",
        "BLOCK 3: SEMEVAL BOUNDARIES",
        "gemma_3_1b_it",
        "gemma_3_270m_it",
        "littlelamb",
        "smollm2_360m",
        "kaggle_train",
        "mage_train",
        "outfox_chatgpt_train",
        "bbc_news_content",
        "Mixed",
        "split_local_evidence_texts(",
        "max_per_split=500",
        "train_whole_text_models(",
        "evaluate_whole_text_datasets(",
        "train_local_evidence_models(",
        "evaluate_local_evidence_datasets(",
        "evaluate_local_evidence_semeval(",
        "semeval2024_task8_subtask_c_train",
        "semeval2024_task8_subtask_c_dev",
        "sustained_length=3",
        "whole_text_metrics",
        "whole_text_predictions",
        "whole_text_coverage",
        "local_evidence_metrics",
        "local_evidence_predictions",
        "local_evidence_coverage",
        "semeval_region_summary",
        "semeval_boundary_summary",
        "combine_whole_text_dataset_results",
        "mixed_boundary",
        "accuracy_tolerance",
        'axis.axvspan(row[\"gold_token_boundary\"], row[\"token_count\"]',
        'axis.axvspan(row[\"predicted_token_boundary\"], row[\"token_count\"]',
        "boundary_predictions",
        "boundary_exclusions",
        "sns.heatmap",
        "BBC one-class evaluation",
    ):
        assert fragment in source


def test_notebook_does_not_run_processing_entrypoints() -> None:
    source = _source()
    for entry_point in ("process_pipeline", "token_similarity_stage2", "load_local_model", "run_stage_2"):
        assert entry_point not in source


def test_notebook_reuses_shared_manifests_and_local_bundles_for_semeval() -> None:
    notebook = _notebook()
    source = _source()
    assert "build_shared_parent_manifest(" in source
    assert "local_model_bundles" in source
    assert "semeval_predicted_regions" in source
    assert "semeval_region_evaluations" in source
    assert "source_coverage_row(" in source
    assert "semeval_predicted_regions" in source
    assert "boundary_predictions" in source

    semeval_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "block-three-semeval")
    semeval_source = "".join(semeval_cell["source"])
    assert semeval_source.count("train_local_evidence_models(") == 0
    assert "local_model_bundles[(llm, training_regime)]" in semeval_source

    block_two = next(cell for cell in notebook["cells"] if cell.get("id") == "block-two-local-evidence")
    assert "train_local_evidence_models(" in "".join(block_two["source"])


def test_notebook_evaluates_train_validation_and_test_partitions() -> None:
    notebook = _notebook()
    for cell_id in ("block-one-whole-text", "block-two-local-evidence"):
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
        assert 'for partition in ("train", "validation", "test")' in "".join(cell["source"])


def test_notebook_appends_visualization_cell_without_replacing_result_tables() -> None:
    notebook = _notebook()
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert cell_ids.index("matrix-displays") < cell_ids.index("matrix-visualizations")
    visualization = next(cell for cell in notebook["cells"] if cell.get("id") == "matrix-visualizations")
    assert visualization["cell_type"] == "code"
    source = "".join(visualization["source"])
    for fragment in (
        "whole_text_metrics",
        "local_evidence_metrics",
        "train",
        "validation",
        "test",
        "macro_f1",
        "sns.heatmap",
        "configuration_columns",
        'aggfunc=\"mean\"',
        "semeval_boundary_summary",
        "exact_accuracy",
        "within_1_accuracy",
        "within_3_accuracy",
        "within_5_accuracy",
        "boundary_predictions",
        "gold_token_boundary",
        "predicted_token_boundary",
        "drop_duplicates([\"llm\", \"training_regime\", \"evaluation_dataset\", \"text_id\"])",
        "groupby([\"llm\", \"training_regime\", \"evaluation_dataset\"]",
        "{training_regime}",
    ):
        assert fragment in source


def test_notebook_evaluates_only_individual_configured_dataset_targets() -> None:
    tree = ast.parse(_source())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EVALUATION_REGIMES" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Dict)
    keys = [key.value if isinstance(key, ast.Constant) else None for key in assignment.value.keys]
    assert keys == ["Kaggle", "MAGE", "Outfox", "BBC", "Mixed"]
    assert all(
        isinstance(value, (ast.Tuple, ast.List))
        and len(value.elts) in (1, 3)
        and all(isinstance(element, ast.Constant) for element in value.elts)
        for value in assignment.value.values
    )


def test_notebook_includes_mixed_evaluation_bbc_partitions_and_llm_comparison() -> None:
    source = _source()
    for fragment in (
        '"Mixed": ("Kaggle", "MAGE", "Outfox")',
        '"BBC": ("BBC",)',
        'for partition in ("train", "validation", "test")',
        'frame["training_regime"].eq("Mixed")',
        "SCORER_LLMS",
        "whole_text_metrics",
        "local_evidence_metrics",
        "semeval_boundary_summary",
    ):
        assert fragment in source


def test_notebook_comments_out_single_feature_threshold_models_and_plots() -> None:
    source = _source()
    assert "single_feature_threshold" not in source
    assert "fit_single_feature_threshold" not in source
