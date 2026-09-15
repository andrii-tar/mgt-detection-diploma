from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from notebooks.utils.classification_results_analysis_utils import (
    ClassificationRunResults,
    aggregate_by_dataset,
    aggregate_by_llm,
    available_table,
    best_universal_by_llm,
    configuration_ranking,
    document_metrics,
    load_classification_run,
    mixed_bbc_comparison,
    plot_aggregate_bars,
    plot_semeval_tradeoff,
    plot_stability_profiles,
    plot_transfer_asymmetry,
    plot_transfer_heatmap,
    plot_universal_frontier,
    pareto_universal_configurations,
    semeval_boundary_ranking,
    semeval_region_ranking,
    transfer_matrix,
    transfer_scores,
    universal_ranking,
)


def _run_root(tmp_path: Path, run_id: str = "run-1") -> Path:
    root = tmp_path / "experiment-results" / "classification" / "evaluations" / run_id
    root.mkdir(parents=True)
    return root


def test_load_classification_run_filters_headered_rows_and_records_provenance(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    pd.DataFrame([
        {"artifact_run_id": "run-1", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.8},
        {"artifact_run_id": "old-run", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.1},
    ]).to_csv(root / "evaluation_metrics.csv", index=False)

    results = load_classification_run(tmp_path, run_id="run-1")

    assert available_table(results, "evaluation_metrics")["artifact_run_id"].tolist() == ["run-1"]
    assert results.provenance.to_dict("records") == [
        {"table": "evaluation_metrics", "source": "headered", "rows": 1},
    ]


def test_load_classification_run_recovers_legacy_evaluation_metrics(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    placeholder_header = "model_family,scorer_llm,feature_set,training_regime,training_window,evaluation_regime,evaluation_window,partition\n"
    row = "1,42,2026-09-13T00:00:00+00:00,extended,logistic_regression,run-1,littlelamb,16,10,10,2,20,Mixed,64,4,4,1,8,64,Mixed,kaggle_train,test,0.8,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8,4,0.8,0.8,0.8,4,3,1,1,3,0.25,0.5\n"
    (root / "evaluation_metrics.csv").write_text(placeholder_header + row, encoding="utf-8")

    results = load_classification_run(tmp_path, run_id="run-1")

    metrics = available_table(results, "evaluation_metrics")
    assert metrics.loc[0, "artifact_run_id"] == "run-1"
    assert metrics.loc[0, "macro_f1"] == 0.8
    assert results.provenance.loc[0, "source"] == "legacy positional recovery"


def test_load_classification_run_recovers_legacy_evaluation_coverage(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    placeholder_header = "dataset,split,class,parent_count,row_count,availability_status\n"
    row = "1,42,2026-09-13T00:00:00+00:00,extended,logistic_regression,run-1,littlelamb,16,10,10,2,20,Mixed,64,4,4,1,8,64,Mixed,kaggle_train,test,HG+MG,binary,2,2,available,2,1,1\n"
    (root / "evaluation_coverage.csv").write_text(placeholder_header + row, encoding="utf-8")

    results = load_classification_run(tmp_path, run_id="run-1")

    coverage = available_table(results, "evaluation_coverage")
    assert coverage.loc[0, "availability_status"] == "available"
    assert coverage.loc[0, "dataset"] == "kaggle_train"


def test_load_classification_run_recovers_headerless_semeval_summary(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    row = "logistic_regression,64,10,0.8,0.7,0.6,0.65,12,0.1,1.0,0.0,run-1,littlelamb,extended,Mixed,64,logistic_regression,path,run-1,2026-09-13T00:00:00+00:00,kaggle_train|mage_train|outfox_chatgpt_train,42,16,30,10,3,1,15,15,5,5,SemEval,semeval,train\n"
    (root / "semeval_summary.csv").write_text(row, encoding="utf-8")

    results = load_classification_run(tmp_path, run_id="run-1")

    summary = available_table(results, "semeval_summary")
    assert summary.loc[0, "mg_f1"] == 0.65
    assert summary.loc[0, "training_regime"] == "Mixed"
    assert results.provenance.loc[0, "source"] == "legacy positional recovery"


def test_load_classification_run_recovers_headerless_semeval_evaluation_and_boundary_tables(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    dimensions = "run-1,littlelamb,extended,Mixed,64,logistic_regression,path,run-1,2026-09-13T00:00:00+00:00,kaggle_train|mage_train|outfox_chatgpt_train,42,16,30,10,3,1,15,15,5,5,SemEval,semeval,train"
    evaluation = f"logistic_regression,64,1,100,0.8,0.7,0.6,0.65,1,0,0,True,40,42,2,{dimensions}\n"
    boundary = f"logistic_regression,64,1,12,10,0.1,0.2,0.3,0.4,0.5,0.25,0.1,{dimensions}\n"
    (root / "semeval_evaluations.csv").write_text(evaluation, encoding="utf-8")
    (root / "semeval_boundary_summary.csv").write_text(boundary, encoding="utf-8")

    results = load_classification_run(tmp_path, run_id="run-1")

    assert available_table(results, "semeval_evaluations").loc[0, "mg_f1"] == 0.65
    assert available_table(results, "semeval_boundary_summary").loc[0, "mean_absolute_token_error"] == 12


def test_load_classification_run_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="result directory does not exist"):
        load_classification_run(tmp_path, run_id="missing")


def test_load_classification_run_rejects_unknown_legacy_row_width(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    (root / "evaluation_metrics.csv").write_text("a,b\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported legacy row width"):
        load_classification_run(tmp_path, run_id="run-1")


def test_document_metrics_excludes_hg_only_bbc_from_macro_f1_ranking() -> None:
    results = ClassificationRunResults(
        "run-1", Path("."),
        {"evaluation_metrics": pd.DataFrame([
            {"dataset": "kaggle_train", "partition": "test", "macro_f1": 0.8, "scorer_llm": "a"},
            {"dataset": "bbc_news_content", "partition": "test", "macro_f1": 1.0, "scorer_llm": "a"},
        ])},
        pd.DataFrame(),
    )

    selected = document_metrics(results)

    assert selected["dataset"].tolist() == ["kaggle_train"]


def test_configuration_and_aggregate_rankings_sort_by_mean_macro_f1() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "extended", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": 0.6},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "extended", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": 0.8},
    ])

    assert configuration_ranking(metrics).loc[0, "scorer_llm"] == "b"
    assert aggregate_by_llm(metrics).loc[0, "scorer_llm"] == "b"
    assert aggregate_by_dataset(metrics).loc[0, "dataset"] == "kaggle_train"


def test_transfer_scores_exclude_only_single_source_same_dataset_cells() -> None:
    metrics = pd.DataFrame([
        {"training_regime": "Kaggle", "dataset": "kaggle_train", "macro_f1": .99},
        {"training_regime": "Kaggle", "dataset": "mage_train", "macro_f1": .60},
        {"training_regime": "Mixed", "dataset": "kaggle_train", "macro_f1": .90},
    ])

    scores = transfer_scores(metrics)

    assert scores["is_eligible_transfer"].tolist() == [False, True, True]


def test_transfer_scores_keep_mixed_bbc_same_dataset_cells_eligible() -> None:
    metrics = pd.DataFrame([
        {"training_regime": "Mixed+BBC", "dataset": "kaggle_train", "macro_f1": .90},
    ])

    scores = transfer_scores(metrics)

    assert scores["is_eligible_transfer"].tolist() == [True]


def test_universal_ranking_sorts_worst_source_before_mean() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .95},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .70},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .80},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .75},
    ])

    assert universal_ranking(metrics).loc[0, "scorer_llm"] == "b"


def test_universal_ranking_excludes_high_single_source_same_dataset_score() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Kaggle", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .99},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Kaggle", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .80},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .80},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .81},
    ])

    ranking = universal_ranking(metrics)

    assert ranking.loc[0, "scorer_llm"] == "b"
    assert ranking.loc[ranking["scorer_llm"].eq("a"), "mean_source_macro_f1"].item() == .80


def test_best_universal_by_llm_selects_each_llms_top_ranked_configuration() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .90},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .80},
        {"scorer_llm": "a", "model_family": "tree", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .75},
        {"scorer_llm": "a", "model_family": "tree", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .75},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .70},
        {"scorer_llm": "b", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .70},
    ])

    best = best_universal_by_llm(metrics)

    assert best[["scorer_llm", "model_family"]].to_dict("records") == [
        {"scorer_llm": "a", "model_family": "lr"},
        {"scorer_llm": "b", "model_family": "lr"},
    ]


def test_transfer_matrix_keeps_unavailable_cells_missing() -> None:
    metrics = pd.DataFrame([
        {"training_regime": "Kaggle", "dataset": "kaggle_train", "macro_f1": .9},
        {"training_regime": "Mixed", "dataset": "mage_train", "macro_f1": .8},
    ])

    matrix = transfer_matrix(metrics, {})

    assert pd.isna(matrix.loc["Kaggle", "mage_train"])


def test_mixed_bbc_comparison_matches_all_non_regime_dimensions() -> None:
    metrics = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "extended", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.70},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "extended", "training_regime": "Mixed+BBC", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.75},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "extended", "training_regime": "Mixed+BBC", "training_window": "32", "evaluation_window": "64", "dataset": "kaggle_train", "partition": "test", "macro_f1": 0.99},
    ])

    comparison = mixed_bbc_comparison(metrics)

    assert comparison[["mixed_macro_f1", "mixed_bbc_macro_f1", "macro_f1_delta"]].iloc[0].tolist() == pytest.approx(
        [0.70, 0.75, 0.05]
    )


def test_semeval_rankings_keep_region_and_boundary_objectives_separate() -> None:
    results = ClassificationRunResults(
        "run-1", Path("."),
        {
            "semeval_summary": pd.DataFrame([
                {"scorer_llm": "a", "feature_set": "x", "mg_f1": 0.8},
                {"scorer_llm": "b", "feature_set": "x", "mg_f1": 0.7},
            ]),
            "semeval_boundary_summary": pd.DataFrame([
                {"scorer_llm": "a", "feature_set": "x", "mean_absolute_token_error": 40.0},
                {"scorer_llm": "b", "feature_set": "x", "mean_absolute_token_error": 20.0},
            ]),
        },
        pd.DataFrame(),
    )

    assert semeval_region_ranking(results).loc[0, "scorer_llm"] == "a"
    assert semeval_boundary_ranking(results).loc[0, "scorer_llm"] == "b"


def test_plot_aggregate_bars_returns_figure() -> None:
    figure = plot_aggregate_bars(
        pd.DataFrame({"scorer_llm": ["a"], "mean": [0.8]}),
        category="scorer_llm",
        title="Run run-1",
    )

    assert figure.axes[0].get_ylabel() == "Mean test macro F1"


def test_pareto_universal_configurations_removes_dominated_rows() -> None:
    ranking = pd.DataFrame([
        {"scorer_llm": "a", "worst_source_macro_f1": .8, "mean_source_macro_f1": .8},
        {"scorer_llm": "b", "worst_source_macro_f1": .7, "mean_source_macro_f1": .9},
        {"scorer_llm": "c", "worst_source_macro_f1": .7, "mean_source_macro_f1": .7},
    ])

    assert pareto_universal_configurations(ranking)["scorer_llm"].tolist() == ["a", "b"]


def test_transfer_heatmap_returns_figure() -> None:
    figure = plot_transfer_heatmap(pd.DataFrame({"kaggle_train": [.8]}, index=["Mixed"]), title="transfer")

    assert figure.axes[0].get_title() == "transfer"
    plt.close(figure)


def test_universal_frontier_labels_pareto_configurations() -> None:
    ranking = pd.DataFrame([
        {"scorer_llm": "a", "worst_source_macro_f1": .8, "mean_source_macro_f1": .8},
        {"scorer_llm": "b", "worst_source_macro_f1": .7, "mean_source_macro_f1": .9},
    ])

    figure = plot_universal_frontier(ranking, title="frontier")

    assert figure.axes[0].get_xlabel() == "Mean eligible test macro F1"
    assert {text.get_text() for text in figure.axes[0].texts} == {"a", "b"}
    plt.close(figure)


def test_stability_profiles_highlight_best_llm_configurations() -> None:
    scores = pd.DataFrame([
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "kaggle_train", "macro_f1": .8},
        {"scorer_llm": "a", "model_family": "lr", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "evaluation_window": "64", "dataset": "mage_train", "macro_f1": .7},
    ])
    best = scores.drop(columns=["dataset", "macro_f1"])

    figure = plot_stability_profiles(scores, best, title="stability")

    assert figure.axes[0].get_ylabel() == "Test macro F1"
    assert figure.axes[0].get_title() == "stability"
    plt.close(figure)


def test_transfer_asymmetry_returns_directed_source_figure() -> None:
    matrix = pd.DataFrame(
        [[.9, .7, .6], [.8, .9, .65], [.75, .7, .9]],
        index=["Kaggle", "MAGE", "Outfox"],
        columns=["kaggle_train", "mage_train", "outfox_chatgpt_train"],
    )

    figure = plot_transfer_asymmetry(matrix, title="asymmetry")

    assert figure.axes[0].get_title() == "asymmetry"
    assert len(figure.axes[0].patches) == 6
    plt.close(figure)


def test_semeval_tradeoff_joins_matching_configuration_dimensions() -> None:
    region = pd.DataFrame([
        {"scorer_llm": "a", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "model_family": "lr", "evaluation_window": "64", "mg_f1": .8},
        {"scorer_llm": "b", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "model_family": "lr", "evaluation_window": "64", "mg_f1": .9},
    ])
    boundary = pd.DataFrame([
        {"scorer_llm": "a", "feature_set": "x", "training_regime": "Mixed", "training_window": "64", "model_family": "lr", "evaluation_window": "64", "mean_absolute_token_error": 10.0},
    ])

    figure = plot_semeval_tradeoff(region, boundary, title="trade-off")

    assert figure.axes[0].get_xlabel() == "Mean absolute token error"
    assert len(figure.axes[0].collections[0].get_offsets()) == 1
    plt.close(figure)
