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


def _named_assignment(tree: ast.AST, name: str) -> ast.Assign:
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(assignments) == 1
    return assignments[0]


def _keyword_names(node: ast.Call) -> set[str]:
    return {keyword.arg for keyword in node.keywords if keyword.arg is not None}


def _keyword(node: ast.Call, name: str) -> ast.keyword:
    return next(keyword for keyword in node.keywords if keyword.arg == name)


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
    training_assignment = _named_assignment(tree, "TRAINING_REGIMES")
    assert isinstance(training_assignment.value, ast.Dict)
    assert [key.value for key in training_assignment.value.keys] == ["Kaggle", "Outfox", "Mixed"]
    assert [_constant_tuple(value) for value in training_assignment.value.values] == [
        ("Kaggle",),
        ("Outfox",),
        ("Kaggle", "Outfox"),
    ]
    evaluation_assignment = _named_assignment(tree, "EVALUATION_REGIMES")
    assert isinstance(evaluation_assignment.value, ast.Name)
    assert evaluation_assignment.value.id == "TRAINING_REGIMES"

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

    coverage_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "coverage_frames"
    ]
    assert len(coverage_assignments) == 3
    source_coverage = next(
        node for node in coverage_assignments
        if any(
            isinstance(child, ast.Constant) and child.value == "source"
            for child in ast.walk(node)
        )
    )
    source_assign = next(
        child for child in ast.walk(source_coverage)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "assign"
    )
    assert {"llm", "source_dataset", "training_regime", "evaluation_regime"} <= _keyword_names(source_assign)

    semeval_frames = {
        "semeval_summary_frames": "summary",
        "semeval_evaluation_frames": "evaluations",
        "semeval_region_frames": "predicted_regions",
        "semeval_scored_window_frames": "scored_windows",
    }
    for frame_list, result_name in semeval_frames.items():
        append = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == frame_list
        )
        assign = next(
            child for child in ast.walk(append)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "assign"
        )
        assert isinstance(assign.func.value, ast.Attribute)
        assert assign.func.value.attr == result_name
        assert {"llm", "training_regime", "evaluation_regime"} <= _keyword_names(assign)


def test_poc_notebook_passes_training_aliases_from_dataset_mapping() -> None:
    tree = _tree()
    training_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "TRAINING_REGIMES"
    )
    alias_assign = next(
        node for node in ast.walk(training_loop)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "training_aliases"
    )
    assert isinstance(alias_assign.value, ast.Call)
    assert isinstance(alias_assign.value.func, ast.Name)
    assert alias_assign.value.func.id == "tuple"
    assert len(alias_assign.value.args) == 1
    aliases = alias_assign.value.args[0]
    assert isinstance(aliases, ast.GeneratorExp)
    assert isinstance(aliases.elt, ast.Subscript)
    assert isinstance(aliases.elt.value, ast.Name)
    assert aliases.elt.value.id == "DATASETS"
    assert isinstance(aliases.elt.slice, ast.Name)
    assert aliases.elt.slice.id == "name"
    assert len(aliases.generators) == 1
    generator = aliases.generators[0]
    assert isinstance(generator.target, ast.Name)
    assert generator.target.id == "name"
    assert isinstance(generator.iter, ast.Name)
    assert generator.iter.id == "training_names"

    training_call = _calls(training_loop, "train_local_evidence_models")
    assert len(training_call) == 1
    assert len(training_call[0].args) >= 2
    assert isinstance(training_call[0].args[1], ast.Name)
    assert training_call[0].args[1].id == "training_aliases"


def test_poc_notebook_passes_dataset_alias_and_matching_test_cohort() -> None:
    tree = _tree()
    evaluation_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "EVALUATION_REGIMES"
    )
    evaluation_call = _calls(evaluation_loop, "evaluate_local_evidence_datasets")[0]
    assert len(evaluation_call.args) == 4
    datasets_arg = evaluation_call.args[2]
    assert isinstance(datasets_arg, ast.Tuple)
    assert len(datasets_arg.elts) == 1
    dataset_alias = datasets_arg.elts[0]
    assert isinstance(dataset_alias, ast.Subscript)
    assert isinstance(dataset_alias.value, ast.Name)
    assert dataset_alias.value.id == "DATASETS"
    assert isinstance(dataset_alias.slice, ast.Name)
    assert dataset_alias.slice.id == "evaluation_name"

    cohorts_arg = evaluation_call.args[3]
    assert isinstance(cohorts_arg, ast.Dict)
    assert len(cohorts_arg.keys) == len(cohorts_arg.values) == 1
    cohort_key = cohorts_arg.keys[0]
    assert isinstance(cohort_key, ast.Subscript)
    assert isinstance(cohort_key.value, ast.Name)
    assert cohort_key.value.id == "DATASETS"
    assert isinstance(cohort_key.slice, ast.Name)
    assert cohort_key.slice.id == "evaluation_name"
    cohort_value = cohorts_arg.values[0]
    assert isinstance(cohort_value, ast.Subscript)
    assert isinstance(cohort_value.value, ast.Name)
    assert cohort_value.value.id == "test_ids"
    assert isinstance(cohort_value.slice, ast.Name)
    assert cohort_value.slice.id == "evaluation_name"


def test_poc_notebook_combines_results_as_mixed() -> None:
    calls = _calls(_tree(), "combine_local_evidence_dataset_results")
    assert len(calls) == 1
    evaluation_regime = _keyword(calls[0], "evaluation_regime")
    assert isinstance(evaluation_regime.value, ast.Constant)
    assert evaluation_regime.value.value == "Mixed"


def test_poc_notebook_labels_per_dataset_and_mixed_output_frames() -> None:
    tree = _tree()
    result_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and _constant_tuple(node.iter)
        == ("metrics", "predictions", "scored_windows", "coverage")
    ]
    assert len(result_loops) == 2
    labels = []
    for result_loop in result_loops:
        assignment = next(
            node for node in ast.walk(result_loop)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "evaluation_dataset"
        )
        labels.append(assignment.value)

    assert any(
        isinstance(label, ast.Subscript)
        and isinstance(label.value, ast.Name)
        and label.value.id == "DATASETS"
        and isinstance(label.slice, ast.Name)
        and label.slice.id == "evaluation_name"
        for label in labels
    )
    assert any(
        isinstance(label, ast.Constant) and label.value == "Mixed"
        for label in labels
    )


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
