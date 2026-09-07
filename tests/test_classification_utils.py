from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import notebooks.classification_utils as classification_utils
import notebooks.local_evidence_utils as local_evidence_utils
from notebooks.classification_utils import (
    build_canonical_cohort,
    evaluate_scores,
    fit_single_feature_threshold,
    load_baseline_cohort,
    run_baseline_experiments,
    select_model_by_coverage,
    split_cohort,
)
from notebooks.local_evidence_utils import (
    aggregate_evidence_fraction,
    aggregate_maximum_window,
    fit_local_evidence_model,
    plot_local_evidence_timeline,
    run_local_evidence_experiments,
    score_local_evidence_windows,
    split_local_evidence_texts,
)


FEATURES = [
    "prompt_mu_score",
    "prompt_sigma_score",
    "perplexity",
    "d_score_top20",
    "d_score_top20_coverage",
    "text_len_words",
]
TEST_LOCAL_FEATURES = ("observed_logprob__mean", "top20_coverage__mean")


def metric_row(row_id: int, split: str, **overrides: object) -> dict[str, object]:
    return {
        "id": row_id,
        "split": split,
        "prompt_mu_score": float(row_id),
        "prompt_sigma_score": float(row_id) + 0.1,
        "perplexity": float(row_id) + 10.0,
        "d_score_top20": float(row_id) - 2.0,
        "d_score_top20_coverage": 0.5,
        "text_len_words": 100 + row_id,
        **overrides,
    }


def write_metrics(processed_root: Path, dataset: str, model: str, split: str, rows: list[dict[str, object]]) -> None:
    path = processed_root / f"{dataset}__{model}" / split / "metrics" / "metrics.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_select_model_by_coverage_prefers_largest_balanced_pair() -> None:
    frame = pd.DataFrame(
        [
            *[metric_row(index, "HG", model="small") for index in range(3)],
            *[metric_row(index, "MG", model="small") for index in range(3, 6)],
            *[metric_row(index, "HG", model="balanced") for index in range(6, 11)],
            *[metric_row(index, "MG", model="balanced") for index in range(11, 16)],
            *[metric_row(index, "HG", model="skewed") for index in range(16, 26)],
            *[metric_row(index, "MG", model="skewed") for index in range(26, 28)],
        ]
    )
    frame["dataset"] = "kaggle_train"

    selected, coverage = select_model_by_coverage(frame, "kaggle_train")

    assert selected == "balanced"
    assert coverage.set_index("model").loc["balanced", ["hg_rows", "mg_rows"]].to_dict() == {
        "hg_rows": 5,
        "mg_rows": 5,
    }


def test_select_model_by_coverage_uses_total_then_model_alias_for_ties() -> None:
    frame = pd.DataFrame(
        [
            *[metric_row(index, "HG", model="zeta") for index in range(4)],
            *[metric_row(index, "MG", model="zeta") for index in range(4, 7)],
            *[metric_row(index, "HG", model="alpha") for index in range(7, 10)],
            *[metric_row(index, "MG", model="alpha") for index in range(10, 14)],
        ]
    )
    frame["dataset"] = "kaggle_train"

    selected, _ = select_model_by_coverage(frame, "kaggle_train")

    assert selected == "alpha"


def test_build_canonical_cohort_maps_labels_and_removes_non_finite_rows() -> None:
    frame = pd.DataFrame(
        [
            metric_row(1, "HG"),
            metric_row(2, "MG"),
            metric_row(3, "HG", perplexity=float("nan")),
            metric_row(4, "other"),
        ]
    )

    cohort = build_canonical_cohort(frame, FEATURES)

    assert cohort[["id", "split", "label"]].to_dict("records") == [
        {"id": 1, "split": "HG", "label": 0},
        {"id": 2, "split": "MG", "label": 1},
    ]


def test_load_baseline_cohort_discovers_metrics_and_honors_model_override(tmp_path: Path) -> None:
    write_metrics(tmp_path, "kaggle_train", "model_a", "HG", [metric_row(1, "HG")])
    write_metrics(tmp_path, "kaggle_train", "model_a", "MG", [metric_row(2, "MG")])
    write_metrics(tmp_path, "kaggle_train", "model_b", "HG", [metric_row(3, "HG"), metric_row(4, "HG")])
    write_metrics(tmp_path, "kaggle_train", "model_b", "MG", [metric_row(5, "MG"), metric_row(6, "MG")])

    cohort, features, selected_model, coverage = load_baseline_cohort(
        tmp_path,
        dataset="kaggle_train",
        model="model_a",
    )

    assert selected_model == "model_a"
    assert features == FEATURES
    assert set(cohort["label"]) == {0, 1}
    assert set(coverage["model"]) == {"model_a", "model_b"}


def test_build_canonical_cohort_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        build_canonical_cohort(pd.DataFrame([metric_row(1, "HG")]), ["not_a_metric"])


def test_limit_parent_texts_selects_numeric_ids_per_split_and_preserves_provenance() -> None:
    frame = pd.DataFrame(
        [
            {"split": "MG", "text_id": 11, "label": 1, "dataset": "source", "window": "mg-11"},
            {"split": "HG", "text_id": 10, "label": 0, "dataset": "source", "window": "hg-10"},
            {"split": "HG", "text_id": 2, "label": 0, "dataset": "source", "window": "hg-2-first"},
            {"split": "MG", "text_id": 3, "label": 1, "dataset": "source", "window": "mg-3"},
            {"split": "HG", "text_id": 2, "label": 0, "dataset": "source", "window": "hg-2-second"},
            {"split": "MG", "text_id": 1, "label": 1, "dataset": "source", "window": "mg-1"},
        ]
    )

    limited = classification_utils.limit_parent_texts(frame, id_column="text_id", max_per_split=2)

    assert list(limited.columns) == list(frame.columns)
    assert limited.loc[limited["split"].eq("HG"), "text_id"].tolist() == [2, 10]
    assert limited.loc[limited["split"].eq("MG"), "text_id"].tolist() == [1, 3]
    assert limited.loc[limited["text_id"].eq(2), "window"].tolist() == ["hg-2-first"]
    assert limited["dataset"].eq("source").all()


def test_limit_parent_texts_rejects_non_positive_limits_and_conflicting_labels() -> None:
    frame = pd.DataFrame(
        [
            {"split": "HG", "id": 1, "label": 0},
            {"split": "HG", "id": 1, "label": 1},
        ]
    )

    with pytest.raises(ValueError, match="max_per_split must be greater than 0"):
        classification_utils.limit_parent_texts(frame, id_column="id", max_per_split=0)
    with pytest.raises(ValueError, match="each parent text must have exactly one label"):
        classification_utils.limit_parent_texts(frame, id_column="id")


def test_limit_parent_texts_preserves_existing_numeric_parent_id_provenance() -> None:
    frame = pd.DataFrame(
        {
            "split": ["HG", "HG"],
            "text_id": [10, 2],
            "label": [0, 0],
            "_numeric_parent_id": ["source-10", "source-2"],
        }
    )

    limited = classification_utils.limit_parent_texts(frame, id_column="text_id")

    assert limited["text_id"].tolist() == [2, 10]
    assert limited["_numeric_parent_id"].tolist() == ["source-2", "source-10"]


def test_limit_parent_texts_retains_numeric_parent_id_when_it_is_the_requested_id() -> None:
    frame = pd.DataFrame(
        {
            "split": ["MG", "MG"],
            "_numeric_parent_id": ["10", "2"],
            "label": [1, 1],
        }
    )

    limited = classification_utils.limit_parent_texts(frame, id_column="_numeric_parent_id")

    assert list(limited.columns) == list(frame.columns)
    assert limited["_numeric_parent_id"].tolist() == ["2", "10"]


def labeled_cohort(rows_per_label: int = 20) -> pd.DataFrame:
    rows = []
    for label, split in ((0, "HG"), (1, "MG")):
        for index in range(rows_per_label):
            row_id = label * rows_per_label + index
            rows.append(
                {
                    **metric_row(
                        row_id,
                        split,
                        prompt_mu_score=float(label * 10 + index / 100),
                        prompt_sigma_score=float(label) + 0.2,
                        perplexity=float(10 + label),
                        d_score_top20=float(label) + 0.1,
                        d_score_top20_coverage=0.4 + 0.1 * label,
                        text_len_words=100 + index,
                    ),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_split_cohort_is_reproducible_and_stratified() -> None:
    cohort = labeled_cohort()

    first = split_cohort(cohort, seed=42)
    second = split_cohort(cohort, seed=42)

    assert {name: len(partition) for name, partition in first.items()} == {
        "train": 24,
        "validation": 8,
        "test": 8,
    }
    assert all(set(partition["label"]) == {0, 1} for partition in first.values())
    assert first["test"]["id"].tolist() == second["test"]["id"].tolist()


@pytest.mark.parametrize(
    "fractions",
    [
        {"train_fraction": 0.0, "validation_fraction": 0.5, "test_fraction": 0.5},
        {"train_fraction": -0.1, "validation_fraction": 0.6, "test_fraction": 0.5},
        {"train_fraction": 0.5, "validation_fraction": 0.0, "test_fraction": 0.5},
        {"train_fraction": 0.6, "validation_fraction": -0.1, "test_fraction": 0.5},
        {"train_fraction": 0.5, "validation_fraction": 0.5, "test_fraction": 0.0},
        {"train_fraction": 0.6, "validation_fraction": 0.5, "test_fraction": -0.1},
    ],
)
def test_split_cohort_rejects_non_positive_fractions(fractions: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="fractions must be greater than 0"):
        split_cohort(labeled_cohort(), **fractions)


def test_fit_single_feature_threshold_chooses_the_mg_increasing_direction() -> None:
    validation = pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})

    threshold = fit_single_feature_threshold(validation["label"], validation["score"])

    assert threshold["direction"] == "higher_is_mg"
    assert threshold["macro_f1"] == pytest.approx(1.0)


def test_fit_single_feature_threshold_can_invert_the_feature_direction() -> None:
    validation = pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.9, 0.8, 0.2, 0.1]})

    threshold = fit_single_feature_threshold(validation["label"], validation["score"])

    assert threshold["direction"] == "lower_is_mg"
    assert threshold["macro_f1"] == pytest.approx(1.0)


def test_evaluate_scores_reports_required_binary_metrics() -> None:
    result = evaluate_scores(
        y_true=pd.Series([0, 0, 1, 1]),
        mg_score=pd.Series([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["hg_support"] == 2
    assert result["mg_support"] == 2
    assert result["tn"] == 2
    assert result["tp"] == 2
    assert result["false_positive_rate"] == pytest.approx(0.0)
    assert result["predicted_mg_rate"] == pytest.approx(0.5)


def test_evaluate_scores_reports_bbc_safe_metrics_for_hg_only_cohort() -> None:
    result = evaluate_scores(
        y_true=pd.Series([0, 0, 0]),
        mg_score=pd.Series([0.1, 0.8, 0.2]),
        threshold=0.5,
    )

    assert pd.isna(result["roc_auc"])
    assert pd.isna(result["pr_auc"])
    assert pd.isna(result["balanced_accuracy"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["hg_precision"] == pytest.approx(1.0)
    assert result["hg_recall"] == pytest.approx(2 / 3)
    assert result["hg_f1"] == pytest.approx(0.8)
    assert result["hg_support"] == 3
    assert result["mg_precision"] == pytest.approx(0.0)
    assert result["mg_recall"] == pytest.approx(0.0)
    assert result["mg_f1"] == pytest.approx(0.0)
    assert result["mg_support"] == 0
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (2, 1, 0, 0)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)
    assert result["predicted_mg_rate"] == pytest.approx(1 / 3)


def test_evaluate_scores_uses_all_binary_hg_rows_as_the_fpr_denominator() -> None:
    result = evaluate_scores(
        y_true=pd.Series([0, 0, 0, 1, 1]),
        mg_score=pd.Series([0.1, 0.7, 0.8, 0.2, 0.9]),
        threshold=0.5,
    )

    assert (result["tn"], result["fp"]) == (1, 2)
    assert result["false_positive_rate"] == pytest.approx(2 / 3)
    assert result["predicted_mg_rate"] == pytest.approx(3 / 5)


def test_run_baseline_experiments_returns_single_and_multi_feature_test_results() -> None:
    partitions = split_cohort(labeled_cohort())

    summary, predictions = run_baseline_experiments(partitions, FEATURES, seed=42)

    assert set(summary["experiment"]) == {*(f"threshold:{feature}" for feature in FEATURES), "logistic_regression:all"}
    assert set(summary["classifier"]) == {"single_feature_threshold", "logistic_regression"}
    assert {"roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "hg_precision", "mg_recall", "tn", "fp", "fn", "tp"} <= set(summary.columns)
    assert predictions.groupby("experiment").size().eq(len(partitions["test"])).all()
    assert set(predictions["label"]) == {0, 1}


def local_evidence_windows(texts_per_label: int = 10) -> pd.DataFrame:
    rows = []
    for label, split in ((0, "HG"), (1, "MG")):
        for text_id in range(texts_per_label):
            for window_length in (32, 64):
                for start_token in (0, 16, 32):
                    rows.append(
                        {
                            "split": split,
                            "text_id": text_id,
                            "label": label,
                            "window_length": window_length,
                            "start_token": start_token,
                            "stop_token": start_token + window_length,
                            TEST_LOCAL_FEATURES[0]: float(label * 10 + text_id / 100 + start_token / 1000),
                            TEST_LOCAL_FEATURES[1]: float(label * 5 + window_length / 100 + start_token / 1000),
                        }
                    )
    return pd.DataFrame(rows)


def test_split_local_evidence_texts_keeps_parent_texts_disjoint() -> None:
    windows = local_evidence_windows()

    partitions = split_local_evidence_texts(windows, seed=42)

    parent_sets = [{(row.split, row.text_id) for row in partition.itertuples()} for partition in partitions.values()]
    assert not (parent_sets[0] & parent_sets[1] or parent_sets[0] & parent_sets[2] or parent_sets[1] & parent_sets[2])
    assert all(set(partition["label"]) == {0, 1} for partition in partitions.values())


def test_split_local_evidence_texts_limits_binary_parent_cohorts_before_partitioning() -> None:
    windows = local_evidence_windows(texts_per_label=10)

    partitions = split_local_evidence_texts(windows, seed=42, max_per_split=5)

    assert {name: len(partition) for name, partition in partitions.items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    selected = pd.concat(partitions.values(), ignore_index=True)
    assert all(set(partition["label"]) == {0, 1} for partition in partitions.values())
    assert selected.groupby("split")["text_id"].apply(set).to_dict() == {
        "HG": {0, 1, 2, 3, 4},
        "MG": {0, 1, 2, 3, 4},
    }


def test_split_local_evidence_texts_reproducibly_partitions_hg_only_cohort() -> None:
    windows = local_evidence_windows().loc[lambda frame: frame["split"].eq("HG")]

    first = split_local_evidence_texts(windows, seed=42)
    second = split_local_evidence_texts(windows, seed=42)

    assert {name: len(partition) for name, partition in first.items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert all(partition["label"].eq(0).all() for partition in first.values())
    assert {
        name: partition["text_id"].tolist()
        for name, partition in first.items()
    } == {
        name: partition["text_id"].tolist()
        for name, partition in second.items()
    }


def test_split_local_evidence_texts_limits_hg_only_parent_cohort_before_partitioning() -> None:
    windows = local_evidence_windows().loc[lambda frame: frame["split"].eq("HG")]

    partitions = split_local_evidence_texts(windows, seed=42, max_per_split=5)

    assert {name: len(partition) for name, partition in partitions.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    selected = pd.concat(partitions.values(), ignore_index=True)
    assert set(selected["text_id"]) == {0, 1, 2, 3, 4}
    assert selected["split"].eq("HG").all()
    assert selected["label"].eq(0).all()


def test_split_local_evidence_texts_rejects_label_zero_with_non_hg_provenance() -> None:
    windows = local_evidence_windows().loc[lambda frame: frame["split"].eq("HG")].copy()
    windows["split"] = "bbc_news_content"

    with pytest.raises(ValueError, match="HG split with label 0"):
        split_local_evidence_texts(windows, seed=42)


def test_local_evidence_aggregation_uses_maximum_and_evidence_fraction() -> None:
    scored_windows = pd.DataFrame(
        {
            "split": ["MG"] * 3,
            "text_id": [7] * 3,
            "label": [1] * 3,
            "window_length": [32] * 3,
            "start_token": [0, 32, 64],
            "stop_token": [32, 64, 96],
            "mg_score": [0.1, 0.9, 0.4],
        }
    )

    maximum = aggregate_maximum_window(scored_windows)
    fraction = aggregate_evidence_fraction(scored_windows, window_threshold=0.8)

    assert maximum.iloc[0]["mg_score"] == pytest.approx(0.9)
    assert maximum.iloc[0]["selected_start_token"] == 32
    assert fraction.iloc[0]["mg_score"] == pytest.approx(1 / 3)
    assert fraction.iloc[0]["selected_stop_token"] == 64


def test_coefficient_reporting_prints_features_and_grouped_summaries(capsys: pytest.CaptureFixture[str]) -> None:
    feature_columns = (
        "observed_logprob__mean",
        "observed_logprob__std",
        "mean_candidate_logprob_delta__mean",
        "mean_candidate_logprob_delta__slope",
        "mean_candidate_similarity__mean",
        "mean_candidate_similarity__slope",
        "top20_coverage__mean",
        "top20_coverage__slope",
    )
    model = Pipeline([("model", LogisticRegression())])
    model.fit([[0] * len(feature_columns), [1] * len(feature_columns)], [0, 1])
    model.named_steps["model"].coef_ = [[1.0, -2.0, 0.5, -0.5, 0.25, -0.75, 2.0, -1.0]]

    local_evidence_utils.print_local_evidence_model_coefficients(model, feature_columns, window_length=32)

    output = capsys.readouterr().out
    assert "window_length=32" in output
    assert "Positive coefficients increase the MG logit and negative coefficients decrease it." in output
    assert "observed_logprob__mean: coefficient=1.000000" in output
    assert "top20_coverage__slope: coefficient=-1.000000" in output
    assert "observed_logprob: mean_signed=-0.500000 mean_abs=1.500000 l1=3.000000 max_abs=2.000000 (std)" in output
    assert "mean_candidate_logprob_delta: mean_signed=0.000000 mean_abs=0.500000 l1=1.000000 max_abs=0.500000 (mean)" in output
    assert "mean_candidate_similarity: mean_signed=-0.250000 mean_abs=0.500000 l1=1.000000 max_abs=0.750000 (slope)" in output
    assert "top20_coverage: mean_signed=0.500000 mean_abs=1.500000 l1=3.000000 max_abs=2.000000 (mean)" in output


def test_coefficient_reporting_prints_custom_feature_names(capsys: pytest.CaptureFixture[str]) -> None:
    feature_columns = ("feature_a", "feature_b")
    model = Pipeline([("model", LogisticRegression())])
    model.fit([[0, 0], [1, 1]], [0, 1])
    model.named_steps["model"].coef_ = [[1.0, -2.0]]

    local_evidence_utils.print_local_evidence_model_coefficients(model, feature_columns, window_length=32)

    output = capsys.readouterr().out
    assert "feature_a: mean_signed=1.000000 mean_abs=1.000000 l1=1.000000 max_abs=1.000000 (no statistic)" in output
    assert "feature_b: mean_signed=-2.000000 mean_abs=2.000000 l1=2.000000 max_abs=2.000000 (no statistic)" in output


def test_plot_local_evidence_timeline_marks_scores_and_suspicious_windows() -> None:
    windows = pd.DataFrame(
        {
            "split": ["MG", "MG"],
            "text_id": [7, 7],
            "window_length": [32, 32],
            "start_token": [0, 16],
            "stop_token": [32, 48],
            "mg_score": [0.2, 0.9],
            "window_threshold": [0.8, 0.8],
        }
    )
    prediction = pd.Series(
        {
            "split": "MG",
            "text_id": 7,
            "label": 1,
            "prediction": 1,
            "window_length": 32,
            "document_threshold": 0.5,
            "mg_score": 0.9,
        }
    )

    figure = plot_local_evidence_timeline(windows, prediction)

    axis = figure.axes[0]
    assert axis.lines[0].get_xdata().tolist() == [16.0, 32.0]
    assert len(axis.patches) == 1


def test_run_local_evidence_experiments_returns_text_scores_and_attributions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    windows = local_evidence_windows()
    for index, feature in enumerate(local_evidence_utils.LOCAL_FEATURE_COLUMNS):
        windows[feature] = windows[TEST_LOCAL_FEATURES[0]] + index / 1000
    partitions = split_local_evidence_texts(windows, seed=42)

    summary, text_predictions, scored_windows = run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=local_evidence_utils.LOCAL_FEATURE_COLUMNS,
        seed=42,
        show_progress=False,
    )

    assert set(summary["aggregation"]) == {"maximum_window", "evidence_fraction"}
    assert set(text_predictions["aggregation"]) == {"maximum_window", "evidence_fraction"}
    assert text_predictions.groupby(["window_length", "aggregation"]).size().eq(len(partitions["test"])).all()
    assert {"mg_score", "prediction", "selected_start_token", "selected_stop_token"} <= set(text_predictions.columns)
    assert {
        "contribution__observed_logprob__mean",
        "contribution__top20_coverage__finite_fraction",
        "top_positive_feature",
        "top_negative_feature",
    } <= set(scored_windows.columns)
    expected_summary = pd.DataFrame(
        [
            [32, "maximum_window", 0.75, 1.0, 1.0, 0.7333333333333334, 0.9959130155591355, 2, 0, 1, 1],
            [32, "evidence_fraction", 0.75, 0.75, 0.75, 0.7333333333333334, 0.3333333333333333, 2, 0, 1, 1],
            [64, "maximum_window", 0.75, 1.0, 1.0, 0.7333333333333334, 0.9959130155591355, 2, 0, 1, 1],
            [64, "evidence_fraction", 0.75, 0.75, 0.75, 0.7333333333333334, 0.3333333333333333, 2, 0, 1, 1],
        ],
        columns=[
            "window_length",
            "aggregation",
            "accuracy",
            "roc_auc",
            "pr_auc",
            "macro_f1",
            "document_threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    pd.testing.assert_frame_equal(
        summary[expected_summary.columns].reset_index(drop=True),
        expected_summary,
        check_exact=False,
        atol=1e-12,
    )
    expected_prediction_scores = [
        0.004148479255124898,
        0.00428662760806591,
        0.9958227703477759,
        0.9960013189343522,
        0.0,
        0.0,
        0.0,
        0.6666666666666666,
    ]
    actual_prediction_scores = text_predictions["mg_score"].tolist()
    assert actual_prediction_scores == pytest.approx(expected_prediction_scores * 2)
    assert text_predictions["prediction"].tolist() == [0, 0, 0, 1, 0, 0, 0, 1] * 2
    expected_scored_scores = [
        0.003919444445780661,
        0.003988541949585705,
        0.004058852638063911,
        0.0042308269314928725,
        0.0043053901982591,
        0.004381261771635264,
        0.9957676735649855,
        0.9958409767561681,
        0.9959130155591355,
        0.9959040791121612,
        0.9959750293314094,
        0.996044755420255,
        0.004006005066606809,
        0.004076622337865493,
        0.004148479255124898,
        0.004139428613357596,
        0.004212387908117312,
        0.00428662760806591,
        0.9956742329787093,
        0.9957491476366175,
        0.9958227703477759,
        0.995859104142103,
        0.995930830242042,
        0.9960013189343522,
    ]
    for window_length in (32, 64):
        actual_scores = scored_windows.loc[scored_windows["window_length"].eq(window_length), "mg_score"].tolist()
        assert actual_scores == pytest.approx(expected_scored_scores)
    output = capsys.readouterr().out
    reports = output.split("Local evidence model coefficients (window_length=")[1:]
    assert len(reports) == 2
    for window_length, report in zip((32, 64), reports):
        assert report.startswith(f"{window_length})\n")
        coefficient_lines = [line for line in report.splitlines() if ": coefficient=" in line]
        assert len(coefficient_lines) == 36
        assert all(
            any(line.startswith(f"{feature}: coefficient=") for line in coefficient_lines)
            for feature in local_evidence_utils.LOCAL_FEATURE_COLUMNS
        )
        for metric in local_evidence_utils.LOCAL_METRICS:
            assert f"{metric}: mean_signed=" in report


def test_reusable_local_evidence_model_scores_supplied_windows_with_contributions() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)

    model, feature_columns = fit_local_evidence_model(
        windows,
        partitions,
        window_length=32,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
    )
    supplied = windows.loc[windows["window_length"].eq(32)].copy()
    scored = score_local_evidence_windows(model, supplied, feature_columns)

    assert feature_columns == TEST_LOCAL_FEATURES
    assert len(scored) == len(supplied)
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert {
        "contribution__observed_logprob__mean",
        "contribution__top20_coverage__mean",
        "top_positive_feature",
        "top_negative_feature",
    } <= set(scored.columns)
    assert scored["contribution__observed_logprob__mean"].notna().all()


def test_fit_local_evidence_model_ignores_non_training_rows_and_features() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    train_parents = set(partitions["train"].set_index(["split", "text_id"]).index)
    non_training = ~pd.MultiIndex.from_frame(windows[["split", "text_id"]]).isin(train_parents)
    mutated_windows = windows.copy()
    mutated_windows.loc[non_training, TEST_LOCAL_FEATURES[0]] = 1_000_000.0
    mutated_windows.loc[non_training, TEST_LOCAL_FEATURES[1]] = -1_000_000.0

    original_model, feature_columns = fit_local_evidence_model(
        windows,
        partitions,
        window_length=32,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
    )
    mutated_model, _ = fit_local_evidence_model(
        mutated_windows,
        partitions,
        window_length=32,
        feature_columns=feature_columns,
        seed=42,
    )
    train_windows = windows.loc[
        windows["window_length"].eq(32) & pd.MultiIndex.from_frame(windows[["split", "text_id"]]).isin(train_parents)
    ]

    pd.testing.assert_frame_equal(
        pd.DataFrame(original_model.predict_proba(train_windows[list(feature_columns)])),
        pd.DataFrame(mutated_model.predict_proba(train_windows[list(feature_columns)])),
    )


def test_score_local_evidence_windows_rejects_missing_features() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    model, _ = fit_local_evidence_model(
        windows,
        partitions,
        window_length=32,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
    )

    with pytest.raises(ValueError, match="missing feature columns"):
        score_local_evidence_windows(model, windows.drop(columns=TEST_LOCAL_FEATURES[1]), TEST_LOCAL_FEATURES)


def test_run_local_evidence_experiments_shows_window_length_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_calls: list[dict[str, object]] = []

    def record_progress(iterator: object, **kwargs: object) -> object:
        progress_calls.append(kwargs)
        return iterator

    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    monkeypatch.setattr(local_evidence_utils, "tqdm", record_progress)

    run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
        show_progress=True,
    )

    assert progress_calls == [{"desc": "Local evidence window length", "unit": "length"}]


def test_run_local_evidence_experiments_can_disable_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    monkeypatch.setattr(local_evidence_utils, "tqdm", pytest.fail)

    run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
        show_progress=False,
    )


def test_local_evidence_preserves_all_nan_training_features() -> None:
    windows = local_evidence_windows()
    windows["all_nan_feature__mean"] = float("nan")
    partitions = split_local_evidence_texts(windows, seed=42)

    _, _, scored_windows = run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=(TEST_LOCAL_FEATURES[0], "all_nan_feature__mean"),
        seed=42,
        show_progress=False,
    )

    assert "contribution__all_nan_feature__mean" in scored_windows
    assert scored_windows["contribution__all_nan_feature__mean"].eq(0.0).all()


def test_local_evidence_summary_reports_current_length_partition_eligibility_and_support() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    ineligible_parents = pd.concat(
        [partition.groupby("label", group_keys=False).head(1) for partition in partitions.values()],
        ignore_index=True,
    )
    ineligible_index = ineligible_parents.set_index(["split", "text_id"]).index
    eligible_windows = windows.loc[
        ~(
            windows["window_length"].eq(64)
            & windows.set_index(["split", "text_id"]).index.isin(ineligible_index)
        )
    ].reset_index(drop=True)

    summary, _, _ = run_local_evidence_experiments(
        eligible_windows,
        partitions,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
        show_progress=False,
    )

    eligible_parent_index = eligible_windows.loc[
        eligible_windows["window_length"].eq(64), ["split", "text_id"]
    ].drop_duplicates().set_index(["split", "text_id"]).index
    expected = {
        partition: frame.loc[frame.set_index(["split", "text_id"]).index.isin(eligible_parent_index)]
        for partition, frame in partitions.items()
    }
    length_summary = summary.loc[summary["window_length"].eq(64)]
    assert length_summary["available_parent_texts"].eq(sum(len(frame) for frame in expected.values())).all()
    for partition, frame in expected.items():
        assert length_summary[f"{partition}_eligible_parent_texts"].eq(len(frame)).all()
        assert length_summary[f"{partition}_hg_support"].eq(frame["label"].eq(0).sum()).all()
        assert length_summary[f"{partition}_mg_support"].eq(frame["label"].eq(1).sum()).all()
    assert length_summary["hg_support"].eq(expected["test"]["label"].eq(0).sum()).all()
    assert length_summary["mg_support"].eq(expected["test"]["label"].eq(1).sum()).all()


def test_local_evidence_uses_normalized_scores_for_lower_evidence_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    calls = 0

    def fit_forced_direction(y_true: pd.Series, values: pd.Series) -> dict[str, float | str]:
        nonlocal calls
        calls += 1
        if (calls - 1) % 20 == 0:
            return {"direction": "higher_is_mg", "threshold": 0.5, "macro_f1": 0.5}
        return {"direction": "lower_is_mg", "threshold": -0.5, "macro_f1": 0.5}

    def aggregate_mixed_fractions(scored_windows: pd.DataFrame, window_threshold: float) -> pd.DataFrame:
        aggregated = aggregate_maximum_window(scored_windows)
        aggregated["mg_score"] = [0.25 if index % 2 == 0 else 0.75 for index in range(len(aggregated))]
        return aggregated

    monkeypatch.setattr(local_evidence_utils, "fit_single_feature_threshold", fit_forced_direction)
    monkeypatch.setattr(local_evidence_utils, "aggregate_evidence_fraction", aggregate_mixed_fractions)

    summary, text_predictions, _ = run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=TEST_LOCAL_FEATURES,
        seed=42,
        show_progress=False,
    )

    evidence_predictions = text_predictions[text_predictions["aggregation"] == "evidence_fraction"]
    decision_scores = -evidence_predictions["mg_score"]
    assert set(evidence_predictions["score_direction"]) == {"lower_is_mg"}
    assert evidence_predictions["mg_score"].between(0.0, 1.0).all()
    assert evidence_predictions["prediction"].tolist() == (
        decision_scores >= evidence_predictions["document_threshold"]
    ).astype("int64").tolist()
    expected_metrics = evaluate_scores(
        evidence_predictions["label"],
        decision_scores,
        float(evidence_predictions["document_threshold"].iloc[0]),
    )
    evidence_summary = summary[summary["aggregation"] == "evidence_fraction"]
    assert evidence_summary["macro_f1"].tolist() == pytest.approx([expected_metrics["macro_f1"]] * len(evidence_summary))


def test_local_evidence_threshold_ties_prefer_higher_window_then_document_threshold() -> None:
    document_parameters = local_evidence_utils._fit_document_threshold(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.0, 0.5, 0.5, 1.0]),
    )
    selected = local_evidence_utils._select_evidence_parameters(
        [
            {"window_threshold": 0.3, "threshold": 0.95, "macro_f1": 0.5, "direction": "higher_is_mg"},
            {"window_threshold": 0.8, "threshold": 0.2, "macro_f1": 0.5, "direction": "higher_is_mg"},
            {"window_threshold": 0.8, "threshold": 0.75, "macro_f1": 0.5, "direction": "higher_is_mg"},
        ]
    )

    assert document_parameters["threshold"] == pytest.approx(1.0)
    assert selected["window_threshold"] == pytest.approx(0.8)
    assert selected["threshold"] == pytest.approx(0.75)


def test_classification_notebook_is_thin_and_output_free() -> None:
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "text_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    assert "from classification_utils import (" in source
    assert "load_baseline_cohort(" in source
    assert "split_cohort(" in source
    assert "run_baseline_experiments(" in source
    assert "plot_confusion_matrices(" in source
    assert "plot_score_curves(" in source
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in cells
        if cell["cell_type"] == "code"
    )
    assert all(
        "def " not in "".join(cell.get("source", []))
        for cell in cells
        if cell["cell_type"] == "code"
    )


def test_local_evidence_classification_notebook_is_thin_and_output_free() -> None:
    import ast
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    import_tree = ast.parse("".join(cells[1]["source"]))
    imports = {
        alias.name
        for node in ast.walk(import_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"local_evidence_utils", "token_candidate_similarity_heatmaps_utils"}
        for alias in node.names
    }
    assert "plot_normalized_token_metrics_with_local_evidence" in imports
    local_import = next(
        node
        for node in ast.walk(import_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "local_evidence_utils"
    )
    assert all(alias.name != "plot_normalized_token_metrics_with_local_evidence" for alias in local_import.names)

    for required_source in (
        "WINDOW_LENGTHS = (32, 64, 128)",
        "STRIDE = 16",
        "RECOMPUTE_WINDOWS = False",
        "default_local_evidence_path(",
        "load_or_build_local_evidence_windows(",
        "split_local_evidence_texts(",
        "run_local_evidence_experiments(",
        "plot_local_evidence_timeline,",
        "MOST_SUSPICIOUS",
        "MOST_CONFIDENT_ERROR",
        "INSPECT_TEXT = None",
    ):
        assert required_source in source

    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) == 13
    for required_source in (
        "path=WINDOWS_PATH,",
        "recompute=RECOMPUTE_WINDOWS,",
        "partitions = split_local_evidence_texts(windows, seed=SEED)",
        "test_summary, test_predictions, scored_windows = run_local_evidence_experiments(",
        "curve_predictions = test_predictions.assign(",
        'frame["score_direction"].eq("higher_is_mg")',
        'frame["window_length"].astype(str) + " tokens: " + frame["aggregation"]',
        "plot_score_curves(curve_predictions)",
        "prediction_scores = test_predictions.assign(",
        "decision_score=lambda frame: frame[\"mg_score\"].where(",
        ").sort_values(\"decision_score\", ascending=False).reset_index(drop=True)",
        "MOST_SUSPICIOUS = prediction_scores.head(1)",
        "MOST_CONFIDENT_ERROR = prediction_scores.loc[",
        "if INSPECT_TEXT is not None",
        "plot_local_evidence_timeline(test_windows, prediction)",
        'token_range=(prediction["selected_start_token"], prediction["selected_stop_token"])',
    ):
        assert required_source in source

    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in code_cells
    )
    assert all(
        "def " not in "".join(cell.get("source", []))
        for cell in code_cells
    )


def test_local_evidence_classification_notebook_supports_custom_local_inspection() -> None:
    import ast
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    custom_cell = next(cell for cell in notebook["cells"] if cell["id"] == "local-evidence-custom-inspect")
    source = "".join(custom_cell["source"])

    for required_source in (
        "CUSTOM_TEXT_INDEX = 0",
        'CUSTOM_SPLIT = "MG"',
        "CUSTOM_WINDOW_LENGTH = 32",
        "load_text_row(PROCESSED_ROOT, DATASET, MODEL, CUSTOM_SPLIT, CUSTOM_TEXT_INDEX)",
        "build_local_evidence_windows(",
        "text_ids=(custom_text_id,)",
        "show_progress=False",
        "fit_local_evidence_model(",
        "score_local_evidence_windows(",
        "No windows available for custom text",
        'custom_windows.loc[custom_windows["mg_score"].idxmax()]',
        "Independent custom inspection evidence",
        "plot_local_evidence_timeline(custom_windows, custom_prediction)",
    ):
        assert required_source in source

    assert "scored_windows" not in source

    tree = ast.parse(source)
    calls = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert set(calls) >= {
        "build_local_evidence_windows",
        "fit_local_evidence_model",
        "score_local_evidence_windows",
    }
    build_call = calls["build_local_evidence_windows"]
    text_ids = next(keyword.value for keyword in build_call.keywords if keyword.arg == "text_ids")
    assert isinstance(text_ids, ast.Tuple)
    assert len(text_ids.elts) == 1
    assert isinstance(text_ids.elts[0], ast.Name)
    assert text_ids.elts[0].id == "custom_text_id"
    assert any(keyword.arg == "show_progress" for keyword in build_call.keywords)

    fit_call = calls["fit_local_evidence_model"]
    assert [argument.id for argument in fit_call.args[:2] if isinstance(argument, ast.Name)] == [
        "windows",
        "partitions",
    ]
    assert isinstance(fit_call.args[2], ast.Name)
    assert fit_call.args[2].id == "CUSTOM_WINDOW_LENGTH"

    score_call = calls["score_local_evidence_windows"]
    assert [argument.id for argument in score_call.args if isinstance(argument, ast.Name)] == [
        "custom_model",
        "custom_windows",
        "custom_features",
    ]

    custom_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "custom_windows =" in "".join(cell.get("source", []))
    ]
    assert len(custom_cells) == 8
    for cell in custom_cells:
        custom_source = "".join(cell["source"])
        assert "scored_windows" not in custom_source
        assert "custom_prediction" in custom_source
        assert "token_summary_series(" in custom_source

        custom_tree = ast.parse(custom_source)
        custom_calls = [
            node
            for node in ast.walk(custom_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]

        overlay_calls = [
            node
            for node in custom_calls
            if node.func.id == "plot_normalized_token_metrics_with_local_evidence"
        ]
        assert len(overlay_calls) == 1
        overlay_call = overlay_calls[0]
        assert [argument.id for argument in overlay_call.args[:2] if isinstance(argument, ast.Name)] == [
            "custom_summary",
            "custom_windows",
        ]
        token_count = overlay_call.args[2]
        assert isinstance(token_count, ast.Call)
        assert isinstance(token_count.func, ast.Name)
        assert token_count.func.id == "len"
        assert isinstance(token_count.args[0], ast.Name)
        assert token_count.args[0].id == "custom_token_labels"
        smoothing_window = next(
            keyword.value for keyword in overlay_call.keywords if keyword.arg == "window"
        )
        assert isinstance(smoothing_window, ast.Name)
        assert smoothing_window.id == "SMOOTHING_WINDOW"

        score_calls = [node for node in custom_calls if node.func.id == "score_local_evidence_windows"]
        assert len(score_calls) == 1
        assert score_calls[0].lineno < overlay_call.lineno

        load_similarity_call = next(node for node in custom_calls if node.func.id == "load_similarity_row")
        assert [argument.id for argument in load_similarity_call.args if isinstance(argument, ast.Name)] == [
            "PROCESSED_ROOT",
            "DATASET",
            "MODEL",
            "CUSTOM_SPLIT",
            "custom_text_id",
        ]

        summary_call = next(node for node in custom_calls if node.func.id == "validated_text_summary")
        assert [argument.id for argument in summary_call.args if isinstance(argument, ast.Name)] == [
            "custom_row",
            "custom_similarity_row",
        ]
        top_k = next(keyword.value for keyword in summary_call.keywords if keyword.arg == "top_k")
        assert isinstance(top_k, ast.Name)
        assert top_k.id == "TOP_K"

        assert any(
            isinstance(node, ast.Name)
            and node.id == "custom_prediction"
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(custom_tree)
        )

        build_call = next(node for node in custom_calls if node.func.id == "build_local_evidence_windows")
        split_keyword = next(keyword for keyword in build_call.keywords if keyword.arg == "splits")
        assert isinstance(split_keyword.value, ast.Tuple)
        assert len(split_keyword.value.elts) == 1
        assert isinstance(split_keyword.value.elts[0], ast.Name)
        assert split_keyword.value.elts[0].id == "CUSTOM_SPLIT"
        text_ids_keyword = next(keyword for keyword in build_call.keywords if keyword.arg == "text_ids")
        assert isinstance(text_ids_keyword.value, ast.Tuple)
        assert len(text_ids_keyword.value.elts) == 1
        assert isinstance(text_ids_keyword.value.elts[0], ast.Name)
        assert text_ids_keyword.value.elts[0].id == "custom_text_id"


def test_local_evidence_notebook_does_not_merge_duplicate_evidence_columns() -> None:
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "].merge(" not in source
    assert '"window_text"' not in source
    assert '"top_positive_feature"' not in source
    assert '"top_negative_feature"' not in source
