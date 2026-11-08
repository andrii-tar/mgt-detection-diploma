from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")


REPORT_TABLES = (
    "T06_window_metrics_long.csv",
    "T07_document_mean_window.csv",
    "T08_document_length_metrics.csv",
    "T08_window_vs_whole_text.csv",
    "T09_external_document_results.csv",
)


def test_load_report_tables_reads_required_csv_files(tmp_path):
    from notebooks.utils.chapter3_figures_utils import load_report_tables

    (tmp_path / "tables").mkdir()
    for name in REPORT_TABLES:
        pd.DataFrame({"marker": [name]}).to_csv(tmp_path / "tables" / name, index=False)

    tables = load_report_tables(tmp_path)

    assert set(tables) == {name.removesuffix(".csv") for name in REPORT_TABLES}
    assert tables["T06_window_metrics_long"].marker.tolist() == ["T06_window_metrics_long.csv"]


def test_load_report_tables_names_the_first_missing_file(tmp_path):
    from notebooks.utils.chapter3_figures_utils import load_report_tables

    with pytest.raises(FileNotFoundError, match="T06_window_metrics_long.csv"):
        load_report_tables(tmp_path)


def test_save_figure_pair_writes_png_svg_and_closes_figure(tmp_path):
    import matplotlib.pyplot as plt

    from notebooks.utils.chapter3_figures_utils import save_figure_pair

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    png_path, svg_path = save_figure_pair(figure, tmp_path, "F01_smoke")

    assert png_path.name == "F01_smoke.png"
    assert svg_path.name == "F01_smoke.svg"
    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0
    assert plt.fignum_exists(figure.number) is False


def test_f01_matrix_preserves_missing_cross_window_cells():
    from notebooks.utils.chapter3_figures_utils import window_transfer_matrix

    rows = pd.DataFrame({
        "block": ["W", "W", "W"],
        "scorer_llm": ["gemma", "gemma", "gemma"],
        "model_family": ["lr", "lr", "lr"],
        "evaluation_regime": ["Mixed"] * 3,
        "training_window": [32, 32, 64],
        "evaluation_window": [32, 64, 64],
        "macro_f1": [0.7, 0.6, 0.8],
    })

    result = window_transfer_matrix(
        rows, scorer_llm="gemma", model_family="lr", evaluation_regime="Mixed",
    )

    assert result.loc[32, 32] == pytest.approx(0.7)
    assert result.loc[32, 64] == pytest.approx(0.6)
    assert pd.isna(result.loc[64, 32])
    assert result.loc[64, 64] == pytest.approx(0.8)


def test_f02_confusion_normalization_is_by_true_class():
    from notebooks.utils.chapter3_figures_utils import normalize_confusion_counts

    normalized = normalize_confusion_counts(tn=8, fp=2, fn=1, tp=9)

    np.testing.assert_allclose(normalized, [[0.8, 0.2], [0.1, 0.9]])


def test_f04_filters_bins_below_minimum_support_per_class():
    from notebooks.utils.chapter3_figures_utils import eligible_length_metrics

    rows = pd.DataFrame({
        "hg_support": [20, 19, 30], "mg_support": [25, 40, 10],
        "macro_f1": [0.8, 0.9, 0.7],
    })

    result = eligible_length_metrics(rows, minimum_class_support=20)

    assert result.index.tolist() == [0]


def test_f06_source_recall_aggregates_positive_predictions_by_generator():
    from notebooks.utils.chapter3_figures_utils import source_llm_recall

    rows = pd.DataFrame({
        "source_llm": ["alpha", "alpha", "beta"],
        "prediction": [1, 0, 1],
        "scorer_llm": ["gemma", "gemma", "littlelamb"],
        "model_family": ["lr"] * 3,
        "training_window": [32] * 3,
        "evaluation_window": [32] * 3,
    })

    result = source_llm_recall(rows)

    assert result.set_index(["scorer_llm", "source_llm"]).mg_recall.to_dict() == {
        ("gemma", "alpha"): 0.5, ("littlelamb", "beta"): 1.0,
    }


def test_external_document_scores_must_match_t09_baseline_support():
    from notebooks.utils.chapter3_figures_utils import validate_external_score_counts

    scores = pd.DataFrame({
        "dataset_label": ["BBC-only", "BBC-only"], "scorer_llm": ["gemma"] * 2,
        "model_family": ["lr"] * 2, "training_window": [32] * 2,
        "evaluation_window": [32] * 2, "text_id": [1, 2],
    })
    summary = pd.DataFrame({
        "dataset": ["BBC-only"], "evaluation_unit": ["document"],
        "model": ["mean_score_baseline"], "scorer_llm": ["gemma"],
        "model_family": ["lr"], "training_window": [32],
        "evaluation_window": [32], "document_count": [2],
    })

    validate_external_score_counts(scores, summary)
    summary.loc[0, "document_count"] = 3
    with pytest.raises(ValueError, match="does not match T09 support"):
        validate_external_score_counts(scores, summary)


def test_external_score_builder_aggregates_cached_windows_and_attaches_source_metadata(tmp_path, monkeypatch):
    from notebooks.utils import chapter3_figures_utils as figures_utils

    metadata_path = tmp_path / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
    metadata_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": ["1", "2"], "source_llm": ["alpha", "beta"]}).to_csv(metadata_path, index=False)

    def fake_score(data_root, *, dataset, split, **kwargs):
        return pd.DataFrame({
            "source_dataset": [dataset, dataset, dataset, dataset], "split": [split] * 4,
            "text_id": [1, 1, 2, 2], "label": [1 if split == "MG" else 0] * 4,
            "start_token": [0, 10, 0, 10], "stop_token": [10, 20, 10, 20],
            "mg_score": [.2, .8, .2, .3],
        })

    monkeypatch.setattr(figures_utils, "score_prepared_window_partition", fake_score)
    monkeypatch.setattr(
        figures_utils, "load_processed_token_counts",
        lambda *args, **kwargs: pd.DataFrame({"id": ["1", "2"], "token_count": [300, 400]}),
    )

    scores, availability = figures_utils.build_external_document_scores(
        tmp_path, run_id="cached-run", scorer_llms=("gemma",), model_families=("lr",),
        training_windows=(32,), evaluation_windows=(32,), n2llm_limit=2, min_token_count=250,
    )

    assert len(availability.loc[availability.status.eq("available")]) == 2
    assert scores.groupby("dataset").size().to_dict() == {"bbc_news_content": 2, "name_that_llm_train": 2}
    assert scores.loc[scores.text_id.astype(str).eq("1"), "document_score"].eq(.5).all()
    assert scores.loc[scores.dataset.eq("name_that_llm_train"), "source_llm"].tolist() == ["alpha", "beta"]


def test_external_document_score_builder_uses_one_shared_n2llm_id_cap(tmp_path, monkeypatch):
    from notebooks.utils import chapter3_figures_utils as figures_utils

    metadata_path = tmp_path / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
    metadata_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "source_llm": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
    }).to_csv(metadata_path, index=False)
    available_ids = {
        "gemma": [1, 2, 4, 5],
        "littlelamb": [2, 3, 5, 6],
    }

    def fake_score(data_root, *, dataset, split, scorer_llm, **kwargs):
        ids = available_ids[scorer_llm]
        return pd.DataFrame({
            "source_dataset": [dataset] * len(ids),
            "split": [split] * len(ids),
            "text_id": ids,
            "label": [1 if split == "MG" else 0] * len(ids),
            "start_token": [0] * len(ids),
            "stop_token": [10] * len(ids),
            "mg_score": [0.75] * len(ids),
        })

    def fake_token_counts(data_root, *, scorer_llm, **kwargs):
        counts = {
            "gemma": [300, 300, 300, 300, 300, 300],
            "littlelamb": [300, 300, 300, 300, 200, 300],
        }[scorer_llm]
        return pd.DataFrame({"id": ["1", "2", "3", "4", "5", "6"], "token_count": counts})

    monkeypatch.setattr(figures_utils, "score_prepared_window_partition", fake_score)
    monkeypatch.setattr(figures_utils, "load_processed_token_counts", fake_token_counts)

    scores, availability = figures_utils.build_external_document_scores(
        tmp_path,
        run_id="cached-run",
        scorer_llms=("gemma", "littlelamb"),
        model_families=("lr",),
        training_windows=(32,),
        evaluation_windows=(32,),
        n2llm_limit=2,
        min_token_count=250,
    )

    n2llm_scores = scores.loc[scores.dataset.eq("name_that_llm_train")]
    ids_by_scorer = (
        n2llm_scores.groupby("scorer_llm").text_id.apply(lambda ids: sorted(ids.astype(str)))
    ).to_dict()
    assert ids_by_scorer == {"gemma": ["5"], "littlelamb": ["6"]}
    assert len(availability.loc[availability.status.eq("available")]) == 4


def test_source_llm_join_normalizes_ids_and_rejects_duplicate_metadata():
    from notebooks.utils.chapter3_figures_utils import attach_source_llm_metadata

    documents = pd.DataFrame({"text_id": [1, 2], "document_score": [0.2, 0.8]})
    metadata = pd.DataFrame({"id": ["1", "2"], "source_llm": ["alpha", "beta"]})

    result = attach_source_llm_metadata(documents, metadata)

    assert result.source_llm.tolist() == ["alpha", "beta"]
    with pytest.raises(ValueError, match="unique document IDs"):
        attach_source_llm_metadata(documents, pd.concat([metadata, metadata.iloc[[0]]]))


def test_semeval_example_selection_is_deterministic_and_excludes_nonfinite_errors():
    from notebooks.utils.chapter3_figures_utils import select_semeval_profile_examples

    evaluations = pd.DataFrame({
        "text_id": [9, 4, 2, 8],
        "absolute_token_error": [2.0, 2.0, 100.0, float("nan")],
    })

    correct_id, correct_rule = select_semeval_profile_examples(evaluations, correct=True)
    failure_id, failure_rule = select_semeval_profile_examples(evaluations, correct=False)

    assert (correct_id, correct_rule) == (4, "lowest finite boundary error; tie-break by lowest text_id")
    assert (failure_id, failure_rule) == (2, "highest finite boundary error; tie-break by lowest text_id")


def test_semeval_example_selection_requires_all_requested_windows():
    from notebooks.utils.chapter3_figures_utils import select_semeval_profile_examples

    evaluations = pd.DataFrame({
        "text_id": [1, 2, 2], "window_length": [32, 32, 64],
        "absolute_token_error": [0.0, 5.0, 7.0],
    })

    text_id, rule = select_semeval_profile_examples(
        evaluations, correct=True, required_window_lengths=(32, 64),
    )

    assert text_id == 2
    assert "requires window lengths 32,64" in rule


def test_f07_and_f08_semeval_profiles_export_png_svg_pairs(tmp_path):
    from notebooks.utils.chapter3_figures_utils import plot_f07_semeval_profile, plot_f08_semeval_failure

    rows = pd.DataFrame({
        "text_id": [7, 7], "window_length": [32, 64], "training_window": [32, 64],
        "start_token": [0, 0], "stop_token": [32, 64], "mg_score": [.2, .3],
        "gold_token_boundary": [20, 20], "predicted_token_boundary": [21, 25],
        "token_count": [100, 100],
    })

    f07 = plot_f07_semeval_profile(rows, tmp_path, text_id=7, selection_rule="lowest error")
    f08 = plot_f08_semeval_failure(rows, tmp_path, text_id=7, selection_rule="highest error")

    assert [p.suffix for p in f07] == [".png", ".svg"]
    assert [p.suffix for p in f08] == [".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in [*f07, *f08])


def test_semeval_labeled_windows_receive_gold_boundary_from_exclusion_summary():
    from notebooks.utils.chapter3_figures_utils import attach_semeval_gold_boundaries

    labeled = pd.DataFrame({"text_id": [7, 7], "start_token": [0, 32], "gold_label": [0, 1]})
    exclusions = pd.DataFrame({"text_id": [7, 8], "gold_token_boundary": [123, 456]})

    result = attach_semeval_gold_boundaries(labeled, exclusions)

    assert result.gold_token_boundary.tolist() == [123, 123]


def test_boundary_threshold_sweep_reports_mae_and_no_crossing_by_threshold():
    from notebooks.utils.chapter3_figures_utils import semeval_boundary_threshold_sweep

    scored = pd.DataFrame({
        "model": ["lr32"] * 4, "window_length": [32] * 4, "text_id": [7] * 4,
        "start_token": [0, 1, 2, 3], "stop_token": [1, 2, 3, 4],
        "mg_score": [.2, .6, .7, .9],
    })
    gold = pd.DataFrame({"text_id": [7], "gold_token_boundary": [1]})

    result = semeval_boundary_threshold_sweep(
        scored, {7: 4}, gold, thresholds=(.6, .8), smoothing_window=1, sustained_length=3,
    )

    at_point_six = result.loc[result.threshold.eq(.6)].iloc[0]
    at_point_eight = result.loc[result.threshold.eq(.8)].iloc[0]
    assert at_point_six.mean_absolute_token_error == pytest.approx(0.0)
    assert at_point_six.no_crossing_rate == pytest.approx(0.0)
    assert at_point_eight.mean_absolute_token_error == pytest.approx(1.0)
    assert at_point_eight.no_crossing_rate == pytest.approx(1.0)


def test_f09_boundary_threshold_plot_saves_png_svg(tmp_path):
    from notebooks.utils.chapter3_figures_utils import plot_f09_boundary_threshold_sweep

    metrics = pd.DataFrame({
        "model": ["lr32", "lr32", "lr64", "lr64"],
        "window_length": [32, 32, 64, 64], "threshold": [.4, .5, .4, .5],
        "mean_absolute_token_error": [5, 4, 7, 6], "no_crossing_rate": [0, .1, .1, .2],
    })

    paths = plot_f09_boundary_threshold_sweep(metrics, tmp_path)

    assert [path.suffix for path in paths] == [".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)


def test_semeval_window_error_metrics_reports_fraction_by_gold_region():
    from notebooks.utils.chapter3_figures_utils import semeval_window_error_metrics

    windows = pd.DataFrame({
        "scorer_llm": ["gemma", "gemma", "gemma"],
        "model_family": ["logistic_regression"] * 3,
        "training_window": [32] * 3,
        "evaluation_window": [32] * 3,
        "text_id": [1, 1, 2],
        "prediction": [1, 0, 0],
        "gold_label": [0, 0, 1],
    })

    result = semeval_window_error_metrics(windows)

    hg = result.loc[result.gold_label.eq(0)].iloc[0]
    mg = result.loc[result.gold_label.eq(1)].iloc[0]
    assert hg.incorrect_fraction == pytest.approx(.5)
    assert hg.incorrect_window_count == 1
    assert hg.window_count == 2
    assert hg.document_count == 1
    assert mg.incorrect_fraction == pytest.approx(1.0)
    assert mg.incorrect_window_count == 1
    assert mg.window_count == 1
    assert mg.document_count == 1


def test_semeval_token_error_metrics_thresholds_scores_and_coverage():
    from notebooks.utils.chapter3_figures_utils import semeval_token_error_metrics

    scored = pd.DataFrame({
        "scorer_llm": ["gemma", "gemma", "gemma", "gemma"],
        "model_family": ["logistic_regression"] * 2 + ["random_forest"] * 2,
        "training_window": [32] * 4,
        "evaluation_window": [32] * 4,
        "text_id": [1, 1, 2, 2],
        "start_token": [0, 1, 0, 3],
        "stop_token": [2, 3, 1, 4],
        "mg_score": [.2, .8, .1, .9],
    })
    token_counts = {("gemma", 1): 3, ("gemma", 2): 4}
    boundaries = pd.DataFrame({
        "scorer_llm": ["gemma", "gemma"], "text_id": [1, 2], "gold_token_boundary": [1, 2],
    })

    result = semeval_token_error_metrics(scored, token_counts, boundaries, thresholds=(.5, .8))

    full_coverage_mg = result.loc[
        result.model_family.eq("logistic_regression") & result.threshold.eq(.8) & result.gold_label.eq(1)
    ].iloc[0]
    partial_coverage_hg = result.loc[
        result.model_family.eq("random_forest") & result.threshold.eq(.5) & result.gold_label.eq(0)
    ].iloc[0]
    partial_coverage_mg = result.loc[
        result.model_family.eq("random_forest") & result.threshold.eq(.5) & result.gold_label.eq(1)
    ].iloc[0]
    assert full_coverage_mg.incorrect_fraction == pytest.approx(.5)
    assert full_coverage_mg.incorrect_token_count == 1
    assert full_coverage_mg.scored_token_count == 2
    assert full_coverage_mg.coverage == pytest.approx(1.0)
    assert partial_coverage_hg.scored_token_count == 1
    assert partial_coverage_hg.expected_token_count == 2
    assert partial_coverage_hg.coverage == pytest.approx(.5)
    assert partial_coverage_mg.scored_token_count == 1
    assert partial_coverage_mg.expected_token_count == 2
    assert partial_coverage_mg.coverage == pytest.approx(.5)


def test_build_semeval_local_error_inputs_uses_available_summary_configurations(tmp_path, monkeypatch):
    import notebooks.utils.chapter3_figures_utils as figures_utils

    summary_root = tmp_path / "experiment-results" / "classification" / "evaluations" / "run-1"
    summary_root.mkdir(parents=True)
    pd.DataFrame([
        {
            "run_id": "run-1", "scorer_llm": "alpha", "model_family": family,
            "training_window": 32, "window_length": 32, "feature_set": "simple_similarity",
            "training_regime": "Kaggle+MAGE+Outfox",
            "evaluation_dataset": "semeval2024_task8_subtask_c_train", "semeval_split": "train",
        }
        for family in ("logistic_regression", "random_forest")
    ] + [{
        "run_id": "run-1", "scorer_llm": "alpha", "model_family": "logistic_regression",
        "training_window": 32, "window_length": 32, "feature_set": "extended",
        "training_regime": "Kaggle+MAGE+Outfox",
        "evaluation_dataset": "semeval2024_task8_subtask_c_train", "semeval_split": "train",
    }]).to_csv(summary_root / "semeval_summary.csv", index=False)

    annotation_root = tmp_path / "raw" / "semevals_2024-task8" / "SemEval2024-task8" / "subtaskC"
    annotation_root.mkdir(parents=True)
    pd.DataFrame([{"id": 1, "text": "one two", "boundary_word_index": 1}]).to_csv(
        annotation_root / "SemEval2024_task8_C_train_MG.csv", index=False,
    )
    score_calls = []

    def fake_score_prepared_window_partition(data_root, **configuration):
        score_calls.append(configuration)
        return pd.DataFrame({
            "text_id": [1, 1], "start_token": [0, 1], "stop_token": [1, 2],
            "prediction": [0, 1], "mg_score": [.2, .8],
        })

    monkeypatch.setattr(figures_utils, "score_prepared_window_partition", fake_score_prepared_window_partition)
    monkeypatch.setattr(
        figures_utils, "load_semeval_token_labels",
        lambda data_root, *, scorer_llm: {1: ["one", " two"]},
    )

    scored, pure, token_counts, boundaries, availability = figures_utils.build_semeval_local_error_inputs(
        tmp_path, run_id="run-1",
    )

    assert len(score_calls) == 2
    assert {call["model_family"] for call in score_calls} == {"logistic_regression", "random_forest"}
    assert len(scored) == 4
    assert len(pure) == 4
    assert token_counts == {("alpha", 1): 2}
    assert boundaries[["scorer_llm", "text_id", "gold_token_boundary"]].to_dict("records") == [
        {"scorer_llm": "alpha", "text_id": 1, "gold_token_boundary": 1},
    ]
    assert availability.status.eq("available").all()


def test_f10_semeval_local_error_plot_saves_png_svg(tmp_path):
    from notebooks.utils.chapter3_figures_utils import plot_f10_semeval_local_errors

    configuration = {
        "scorer_llm": "gemma", "model_family": "logistic_regression",
        "training_window": 32, "evaluation_window": 32,
    }
    window_metrics = pd.DataFrame([
        {**configuration, "gold_label": 0, "incorrect_fraction": .2, "incorrect_window_count": 2, "window_count": 10, "document_count": 5},
        {**configuration, "gold_label": 1, "incorrect_fraction": .4, "incorrect_window_count": 4, "window_count": 10, "document_count": 5},
    ])
    token_metrics = pd.DataFrame([
        {
            **configuration, "gold_label": label, "threshold": threshold,
            "incorrect_fraction": fraction, "incorrect_token_count": 4,
            "scored_token_count": 10, "expected_token_count": 10,
            "coverage": 1.0, "document_count": 5,
        }
        for label, fractions in ((0, (.1, .2)), (1, (.5, .3)))
        for threshold, fraction in zip((.5, .8), fractions)
    ])

    paths = plot_f10_semeval_local_errors(window_metrics, token_metrics, tmp_path)

    assert [path.suffix for path in paths] == [".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)


def test_window_boundary_comparison_aligns_token_and_window_mae_for_32_64():
    from notebooks.utils.chapter3_figures_utils import build_semeval_window_boundary_comparison

    scored_rows = []
    for training_window, starts_by_text in (
        (32, {1: (0, 16, 32, 48, 64, 80), 2: (0, 16, 32, 48, 64, 80)}),
        (64, {1: (0, 32, 64, 96, 128, 160), 2: (0, 32, 64, 96, 128, 160)}),
    ):
        for text_id, starts in starts_by_text.items():
            scores = (0.1, 0.2, 0.3, 0.5, 0.8, 0.9) if text_id == 1 else (0.1,) * 6
            scored_rows.extend(
                {
                    "scorer_llm": "gemma",
                    "model_family": "logistic_regression",
                    "training_window": training_window,
                    "evaluation_window": training_window,
                    "text_id": text_id,
                    "start_token": start,
                    "stop_token": start + training_window,
                    "mg_score": score,
                }
                for start, score in zip(starts, scores, strict=True)
            )
    scored_windows = pd.DataFrame(scored_rows)
    boundary_rows = pd.DataFrame(
        [
            {
                "scorer_llm": "gemma",
                "model_family": "logistic_regression",
                "training_window": window,
                "window_length": window,
                "text_id": text_id,
                "token_count": 240,
                "gold_token_boundary": gold,
                "absolute_token_error": token_error,
                "no_crossing": no_crossing,
                "predicted_token_boundary": gold + token_error,
                "feature_set": "simple_similarity",
                "training_regime": "Kaggle+MAGE+Outfox",
                "evaluation_dataset": "semeval2024_task8_subtask_c_train",
                "semeval_split": "train",
            }
            for window, rows in (
                (32, ((1, 54, 4, False), (2, 230, 10, True))),
                (64, ((1, 110, 6, False), (2, 220, 20, True))),
            )
            for text_id, gold, token_error, no_crossing in rows
        ]
    )

    result = build_semeval_window_boundary_comparison(scored_windows, boundary_rows).iloc[0]

    assert result.n_32 == 2
    assert result.token_mae_32 == pytest.approx(117.0)
    assert result.window_mae_32 == pytest.approx(116.0)
    assert result.token_no_crossing_pct_32 == pytest.approx(50.0)
    assert result.window_no_transition_pct_32 == pytest.approx(50.0)
    assert result.n_window_detected_32 == 1
    assert result.window_detected_mae_32 == pytest.approx(2.0)
    assert result.window_detection_pct_32 == pytest.approx(50.0)
    assert result.n_64 == 2
    assert result.token_mae_64 == pytest.approx(113.0)
    assert result.window_mae_64 == pytest.approx(111.0)
    assert result.token_no_crossing_pct_64 == pytest.approx(50.0)
    assert result.window_no_transition_pct_64 == pytest.approx(50.0)
    assert result.n_window_detected_64 == 1
    assert result.window_detected_mae_64 == pytest.approx(2.0)
    assert result.window_detection_pct_64 == pytest.approx(50.0)


def test_window_boundary_comparison_detected_only_mae_is_nan_when_no_transition_is_found():
    from notebooks.utils.chapter3_figures_utils import build_semeval_window_boundary_comparison

    scored = pd.DataFrame(
        [
            {
                "scorer_llm": "gemma", "model_family": "logistic_regression",
                "training_window": 32, "evaluation_window": 32, "text_id": 7,
                "start_token": start, "stop_token": start + 32, "mg_score": 0.2,
            }
            for start in (0, 16, 32, 48, 64, 80)
        ]
    )
    token_rows = pd.DataFrame(
        [{
            "scorer_llm": "gemma", "model_family": "logistic_regression", "training_window": 32,
            "window_length": 32, "text_id": 7, "token_count": 128, "gold_token_boundary": 64,
            "absolute_token_error": 10, "no_crossing": False, "predicted_token_boundary": 54,
            "feature_set": "simple_similarity", "training_regime": "Kaggle+MAGE+Outfox",
            "evaluation_dataset": "semeval2024_task8_subtask_c_train", "semeval_split": "train",
        }]
    )

    result = build_semeval_window_boundary_comparison(scored, token_rows).iloc[0]

    assert result.n_window_detected_32 == 0
    assert pd.isna(result.window_detected_mae_32)
    assert result.window_detection_pct_32 == pytest.approx(0.0)


def test_window_boundary_comparison_rejects_eligible_text_without_scored_windows():
    from notebooks.utils.chapter3_figures_utils import build_semeval_window_boundary_comparison

    scored_windows = pd.DataFrame(
        [{
            "scorer_llm": "gemma", "model_family": "logistic_regression", "model": "logistic_regression",
            "training_window": 32, "evaluation_window": 32, "text_id": 1,
            "start_token": 0, "stop_token": 32, "mg_score": 0.2,
        }]
    )
    boundary_rows = pd.DataFrame(
        [{
            "scorer_llm": "gemma", "model_family": "logistic_regression", "training_window": 32,
            "window_length": 32, "text_id": 2, "token_count": 20, "gold_token_boundary": 8,
            "absolute_token_error": 2, "no_crossing": False, "predicted_token_boundary": 10,
            "feature_set": "simple_similarity", "training_regime": "Kaggle+MAGE+Outfox",
            "evaluation_dataset": "semeval2024_task8_subtask_c_train", "semeval_split": "train",
        }]
    )

    with pytest.raises(ValueError, match="missing scored windows"):
        build_semeval_window_boundary_comparison(scored_windows, boundary_rows)
