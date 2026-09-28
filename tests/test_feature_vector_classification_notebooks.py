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


def test_generalization_comparison_notebook_declares_parameters_and_plots() -> None:
    source = _source("compare_classifier_generalization.ipynb")
    assert source.index("MIN_TEXT_TOKENS =") < source.index("load_classification_run(")
    for fragment in (
        "All texts",
        ">= {MIN_TEXT_TOKENS}",
        "held_out_binary_metrics(",
        "comparison_matrix(",
        "plot_matrix_facets(",
        "insufficient_binary_holdout",
    ):
        assert fragment in source


def test_generalization_notebook_uses_matrix_facets_instead_of_flat_plots() -> None:
    source = _source("compare_classifier_generalization.ipynb")
    assert "comparison_matrix(" in source
    assert "plot_matrix_facets(" in source
    assert "configuration_index(" in source
    for forbidden in (
        "plot_generalization_leaderboard(",
        "plot_dataset_heatmap(",
        "plot_f1_mae_frontier(",
        "plot_parameter_effects(",
        "plot_safety_frontier(",
    ):
        assert forbidden not in source


def test_feature_importance_notebook_declares_isolated_permutation_experiment() -> None:
    notebook = _notebook("explore_extended_feature_importance_limited.ipynb")
    source = _source("explore_extended_feature_importance_limited.ipynb")

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    for required in (
        "IMPORTANCE_RUN_ID =",
        "SOURCE_RUN_ID =",
        "PERMUTATION_REPEATS = 5",
        "feature-importance",
        "permute_feature_by_parent",
        "summarize_importance",
    ):
        assert required in source
    assert 'PERMUTATION_TRAINING_WINDOW = "32"' in source
    assert 'binary_candidates = binary_candidates.loc[binary_candidates["training_window"].eq(PERMUTATION_TRAINING_WINDOW)].copy()' in source
    assert 'semeval_ranking = semeval_ranking.loc[semeval_ranking["training_window"].eq(PERMUTATION_TRAINING_WINDOW)].copy()' in source
    assert "save_model_artifact" not in source
    assert "fit_feature_vector_model" not in source


def test_feature_importance_notebook_requires_complete_validation_binary_artifacts() -> None:
    source = _source("explore_extended_feature_importance_limited.ipynb")

    assert 'partition="validation"' in source
    assert 'binary_candidates["dataset_count"].eq(3)' in source
    assert 'raise RuntimeError("No binary configurations have complete prediction coverage.")' in source
    assert '"binary": "mean validation macro-F1"' in source


def test_feature_importance_notebook_reports_permutation_progress() -> None:
    source = _source("explore_extended_feature_importance_limited.ipynb")

    assert 'desc=f"Binary {dataset} permutations"' in source
    assert 'desc=f"SemEval {training_window} permutations"' in source
    assert 'leave=False' in source
    assert 'unit="permutation"' in source


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
        "TRAINING_COMPOSITIONS =",
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
        'MODEL_FAMILIES = ("logistic_regression", "random_forest")',
    ):
        assert fragment in source


def test_prepare_and_train_notebook_iterates_all_dataset_subsets() -> None:
    source = _source()

    for fragment in (
        "dataset_subsets(",
        "TRAINING_COMPOSITIONS =",
        "for training_composition in TRAINING_COMPOSITIONS:",
        "composition_id(training_composition)",
        "composition_has_both_classes(training_composition, SOURCE_SPLITS)",
    ):
        assert fragment in source
    assert 'TRAINING_REGIME = "Mixed+BBC"' not in source


def test_prepare_and_train_notebook_persists_composition_aware_metadata() -> None:
    source = _source()

    for fragment in (
        '"training_composition": training_composition_id',
        '"evaluation_composition": ""',
        '"training_composition": training_composition_id',
        '"dataset": training_composition_id',
    ):
        assert fragment in source


def test_prepare_and_train_notebook_declares_all_scalar_and_similarity_feature_sets() -> None:
    source = _source()

    assert 'RUN_ID = "feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100"' in source
    for name in ("scalar_all", "scalar_compact", "simple_similarity", "extended"):
        assert f'"{name}": FeatureSet("{name}"' in source
    for column in (
        "mean_candidate_similarity__mean",
        "mean_candidate_similarity__std",
        "top20_coverage__mean",
        "top20_coverage__std",
        "observed_logprob__mean",
        "observed_logprob__std",
        "mean_candidate_logprob_delta__mean",
        "mean_candidate_logprob_delta__std",
    ):
        assert f'"{column}"' in source


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
        "ensure_result_exports(",
        "append_result_frame(",
        "discover_model_artifacts(",
        "load_model_artifact(",
        "availability_row(",
        "release_frames(",
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


def test_prepare_and_train_notebook_bounds_source_frame_memory_and_handles_episode_failures() -> None:
    source = _source()

    assert "release_frames(*source_frames)" in source
    assert "except (FileNotFoundError, KeyError, ValueError, OSError) as error:" in source
    assert "fit_feature_vector_model(" in source
    assert "save_model_artifact(" in source
    assert "run_summary.append" in source
    assert "continue" in source


def test_prepare_and_train_notebook_limits_the_new_run_to_fast_model_families() -> None:
    source = _source()

    assert 'RUN_ID = "feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100"' in source
    assert 'MODEL_FAMILIES = ("logistic_regression", "random_forest")' in source


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
        "TRAINING_COMPOSITIONS",
        "EVALUATION_COMPOSITIONS",
        "composition_has_both_classes",
        "composition_id",
        'for partition in ("validation", "test")',
        "EVALUATION_BATCH_SIZE =",
        "iter_prepared_batches(",
        "write_prediction_artifact(",
        "aggregate_prediction_artifact(",
        "release_frames(",
        "ensure_result_exports(",
        "successful_result_keys(",
        "completed_evaluation_keys",
        "evaluation_metrics",
        "evaluation_coverage",
        "run_summary",
        "build_feature_parent_cohorts(",
        "validation_cohort",
        "test_cohort",
        "selected_cohort",
        "evaluate_local_evidence_semeval(",
        "completed_semeval_keys",
        "ensure_prepared_partition(",
        "semeval_prepared_partitions[(scorer_llm, semeval_dataset, feature_set_name, semeval_window)]",
        'semeval_prepared["path"]',
        'semeval_prepared["expected"]',
        "release_frames(prepared_windows)",
        "windows_override=prepared_windows",
        "SEMEVAL_DATASETS =",
        "semeval_summary",
        "semeval_boundary_predictions",
        '"training_datasets": "|".join(metadata["training_datasets"])',
        '"training_composition": training_composition',
        '"evaluation_composition": evaluation_composition_id',
        '"train_row_count": metadata["train_row_count"]',
        '"validation_row_count": metadata["validation_row_count"]',
        '"artifact_created_at": metadata["created_at"]',
        '"stage": "semeval"',
    ):
        assert fragment in source

    assert 'for partition in ("validation", "test")' in source
    assert source.index("ensure_prepared_partition(") < source.index("for artifact_path in tqdm(")
    assert source.index("semeval_prepared = semeval_prepared_partitions[") > source.index("for artifact_path in tqdm(")
    assert source.index("windows_override=prepared_windows") > source.index("for artifact_path in tqdm(")
    assert 'training_window != "whole_text"' in source
    assert "load_or_build_local_evidence_windows(" not in source
    assert "load_episode_partitions(DATA_ROOT" not in source
    assert 'append_result_frame(exports["evaluation_predictions"]' not in source


def test_evaluate_notebook_has_one_configuration_and_does_not_overwrite_streamed_results() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert source.count('RUN_ID = "') == 1
    assert source.count("Preparing SemEval feature partitions") == 1
    assert '"evaluation_predictions": pd.DataFrame()' not in source
    assert source.count("ensure_result_exports(") == 1
    assert "append_result_frame(exports[\"run_summary\"], pd.DataFrame([run_summary[-1]]))" in source
    assert "release_frames(metrics, coverage)" in source


def test_evaluate_notebook_regenerates_missing_prediction_artifacts_on_resume() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert 'RUN_ID = "feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100"' in source
    assert "prediction_artifact_is_complete(prediction_path, expected=artifact_key)" in source
    assert '"evaluation_window": evaluation_window_value' in source
    assert "if not partition_keys <= completed_evaluation_keys or not all(" in source


def test_evaluate_notebook_uses_reduced_composition_matrix_and_windows() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert 'SINGLE_DATASETS = ("Kaggle", "MAGE", "Outfox")' in source
    assert 'MIXED_COMPOSITIONS = (("Kaggle", "MAGE", "Outfox"), ("Kaggle", "MAGE", "Outfox", "BBC"))' in source
    assert 'TRAINING_COMPOSITIONS = tuple((name,) for name in SINGLE_DATASETS) + MIXED_COMPOSITIONS' in source
    assert 'EVALUATION_COMPOSITIONS = TRAINING_COMPOSITIONS' in source
    assert 'EVALUATION_WINDOWS = ("whole_text", 32, 64)' in source
    assert 'str(artifact_metadata.get(artifact_path, {}).get("training_window", "")) in {"whole_text", "32", "64"}' in source
    assert 'MIXED_COMPOSITION_ID = composition_id(MIXED_COMPOSITIONS[0])' in source
    assert 'MIXED_BBC_COMPOSITION_ID = composition_id(MIXED_COMPOSITIONS[1])' in source
    assert 'allowed_evaluation_compositions = {' in source
    assert 'MIXED_BBC_COMPOSITION_ID: {MIXED_COMPOSITION_ID, MIXED_BBC_COMPOSITION_ID}' in source
    assert "build_pooled_feature_parent_cohorts" in source
    assert "MIXED_MAX_PER_SPLIT = 1000" in source
    assert "is_mixed_evaluation = len(source_names) > 1" in source
    assert "dataset = evaluation_composition_id" in source
    assert "build_pooled_feature_parent_cohorts(" in source
    assert "aggregate_prediction_artifact(prediction_path" in source
    assert "for evaluation_composition in EVALUATION_COMPOSITIONS:" in source
    assert "composition_has_both_classes(evaluation_composition, SOURCE_SPLITS)" in source
    assert "evaluation_composition_id = composition_id(evaluation_composition)" in source
    assert 'if evaluation_composition_id not in allowed_evaluation_compositions[training_composition_id]: continue' in source
    assert "128" not in source[source.index('EVALUATION_WINDOWS ='):source.index('EVALUATION_BATCH_SIZE =')]


def test_evaluate_notebook_uses_semeval_train_only() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert 'SEMEVAL_DATASETS = {"train":' in source
    assert '"dev":' not in source[source.index("SEMEVAL_DATASETS ="):]


def test_evaluate_notebook_shows_semeval_preparation_progress() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert 'for (scorer_llm, feature_set_name, semeval_window), feature_columns in tqdm(semeval_configurations.items(), desc="Preparing SemEval feature partitions", unit="configuration"):' in source
    assert 'for semeval_split, (semeval_dataset, semeval_path) in tqdm(SEMEVAL_DATASETS.items(), desc=f"{scorer_llm}/{feature_set_name}/{semeval_window}", unit="split", leave=False):' in source


def test_analysis_notebook_is_output_free_single_run_and_uses_analysis_utilities() -> None:
    notebook = _notebook("analyze_classifier_results.ipynb")
    source = _source("analyze_classifier_results.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert notebook["nbformat"] >= 4
    assert source.count('RUN_ID = "') == 1
    assert all("execution" not in cell["metadata"] for cell in notebook["cells"])
    for fragment in (
        "load_classification_run(", "document_metrics(", "configuration_ranking(",
        "aggregate_by_llm(", "aggregate_by_dataset(", "mixed_bbc_comparison(",
        "bbc_safety_metrics(", "semeval_region_ranking(", "semeval_boundary_ranking(",
        "plot_feature_set_heatmap(", "plot_window_heatmap(", "plot_regime_deltas(",
    ):
        assert fragment in source
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        assert "execution" not in cell["metadata"]
        tree = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )


def test_analysis_notebook_selects_only_sem_eval_texts_with_at_least_200_tokens() -> None:
    source = _source("analyze_classifier_results.ipynb")

    assert 'semeval_boundary_configuration["token_count"].ge(200)' in source
    assert "eligible_text_ids" in source
    assert "eligible_text_ids=eligible_text_ids" in source


def test_analysis_notebook_declares_dimension_plot_matrix_and_availability_warning() -> None:
    source = _source("analyze_classifier_results.ipynb")

    assert "ANALYSIS_DIMENSIONS = (" in source
    for dimension in (
        "scorer_llm",
        "model_family",
        "feature_set",
        "training_regime",
        "training_window",
        "evaluation_window",
    ):
        assert f'"{dimension}"' in source
    assert "plot_dimension_stability_profiles(" in source
    assert "plot_dimension_semeval_tradeoff(" in source
    assert "for dimension in ANALYSIS_DIMENSIONS:" in source
    assert "Availability warning" in source


def test_analysis_notebook_checks_stability_and_semeval_dimensions_independently() -> None:
    source = _source("analyze_classifier_results.ipynb")
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]

    stability_loops = [
        node for node in loops
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "plot_dimension_stability_profiles"
            for child in ast.walk(node)
        )
    ]
    semeval_loops = [
        node for node in loops
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "plot_dimension_semeval_tradeoff"
            for child in ast.walk(node)
        )
    ]

    assert len(stability_loops) == 1
    assert len(semeval_loops) == 1
    assert stability_loops[0] is not semeval_loops[0]
    stability_source = ast.get_source_segment(source, stability_loops[0]) or ""
    semeval_source = ast.get_source_segment(source, semeval_loops[0]) or ""
    assert "usable_transfer" in stability_source
    assert "usable_region" not in stability_source
    assert "usable_region" in semeval_source
    assert "usable_transfer" not in semeval_source


def test_analysis_notebook_preserves_existing_outputs_and_adds_transfer_analysis() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _notebook("analyze_classifier_results.ipynb")["cells"]
    )

    for fragment in (
        "## Document Classification", "## Feature Sets And Classification Parameters",
        "## Mixed Versus Mixed+BBC Training", "## SemEval Region And Boundary Localization",
        "transfer_scores(", "universal_ranking(", "best_universal_by_llm(",
        "plot_transfer_heatmap(", "plot_universal_frontier(", "plot_stability_profiles(",
        "plot_transfer_asymmetry(", "plot_semeval_tradeoff(",
    ):
        assert fragment in source


def test_analysis_notebook_compares_requested_semeval_windows() -> None:
    source = _source("analyze_classifier_results.ipynb")

    assert "SEMEVAL_COMPARISON_WINDOWS = (32, 64, 128)" in source
    assert 'window_length"].isin(SEMEVAL_COMPARISON_WINDOWS)' in source


def test_evaluate_notebook_records_expected_failures_and_reraises_unexpected_errors() -> None:
    source = _source("evaluate_classifiers.ipynb")

    assert "except (FileNotFoundError, KeyError, ValueError, OSError) as error:" in source
    assert '"status": "unavailable"' in source
    assert "continue" in source
    assert "except Exception:" in source
    assert "raise" in source
    for forbidden in ("fit_feature_vector_model(", "save_model_artifact(", "process_pipeline", "token_similarity_stage2"):
        assert forbidden not in source


def test_bounded_prefix_evaluation_notebook_contract() -> None:
    notebook = _notebook("report/evaluate_bounded_prefix_classifiers.ipynb")
    source = _source("report/evaluate_bounded_prefix_classifiers.ipynb")

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            tree = ast.parse("".join(cell["source"]))
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in ast.walk(tree)
            )
    for fragment in (
        'RUN_ID = "bounded-prefix-classifier-evaluation-window32-2026-09-25"',
        'SOURCE_RUN_ID = "feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100"',
        "TOKEN_BOUNDS =",
        "MIN_MACRO_F1 =",
        "MIN_BALANCED_ACCURACY =",
        "MIN_CLASS_SUPPORT =",
        'MODEL_FAMILIES = ("logistic_regression", "random_forest")',
        "ensure_bounded_prefix_result_exports(",
        "iter_bounded_prefix_feature_batches(",
        "collect_source_parent_identities(",
        "prefix_window_label(",
        'progress_description=f"Collecting parents:',
        'progress_description=f"Scoring prefix:',
        "bounded_prefix_prediction_key(",
        "prediction_artifact_is_complete(",
        "aggregate_prediction_artifact(",
        'read(columns=["split", "text_id"])',
        '.drop_duplicates(["split", "text_id"])',
        "original_token_count",
        "evaluated_token_count",
        "No qualifying length",
    ):
        assert fragment in source
    for forbidden in ("fit_feature_vector_model(", "save_model_artifact(", "process_pipeline", "token_similarity_stage2"):
        assert forbidden not in source


def test_semeval_random_forest_visualization_notebook_contract() -> None:
    notebook = _notebook("visualize_semeval_random_forest_token_profiles.ipynb")
    source = _source("visualize_semeval_random_forest_token_profiles.ipynb")

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            tree = ast.parse("".join(cell["source"]))
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in ast.walk(tree)
            )
    for fragment in (
        'SCORER_LLM = "gemma_3_270m_it"',
        'MODEL_FAMILY = "random_forest"',
        'FEATURE_SET = "simple_similarity"',
        'TRAINING_REGIME = "Mixed"',
        'WINDOW_LENGTHS = (32, 64, 128)',
        "TEXT_COUNT = 20",
        "SELECTION_SEED = 42",
        "load_visualization_models(",
        "resolve_prepared_inputs(",
        "select_eligible_text_ids(",
        "score_visualization_windows(",
        "build_gold_boundaries(",
        "write_visualization_export(",
        "plot_window_level_scores(",
        "plot_reconstructed_token_scores(",
    ):
        assert fragment in source
    for forbidden in (
        "ensure_prepared_partition(",
        "fit_feature_vector_model(",
        "save_model_artifact(",
        "process_pipeline",
        "token_similarity_stage2",
    ):
        assert forbidden not in source
