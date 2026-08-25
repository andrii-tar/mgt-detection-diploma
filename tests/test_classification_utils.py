from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
                            "feature_a": float(label * 10 + text_id / 100 + start_token / 1000),
                            "feature_b": float(label * 5 + window_length / 100 + start_token / 1000),
                        }
                    )
    return pd.DataFrame(rows)


def test_split_local_evidence_texts_keeps_parent_texts_disjoint() -> None:
    windows = local_evidence_windows()

    partitions = split_local_evidence_texts(windows, seed=42)

    parent_sets = [{(row.split, row.text_id) for row in partition.itertuples()} for partition in partitions.values()]
    assert not (parent_sets[0] & parent_sets[1] or parent_sets[0] & parent_sets[2] or parent_sets[1] & parent_sets[2])
    assert all(set(partition["label"]) == {0, 1} for partition in partitions.values())


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


def test_run_local_evidence_experiments_returns_text_scores_and_attributions() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)

    summary, text_predictions, scored_windows = run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=("feature_a", "feature_b"),
        seed=42,
        show_progress=False,
    )

    assert set(summary["aggregation"]) == {"maximum_window", "evidence_fraction"}
    assert set(text_predictions["aggregation"]) == {"maximum_window", "evidence_fraction"}
    assert text_predictions.groupby(["window_length", "aggregation"]).size().eq(len(partitions["test"])).all()
    assert {"mg_score", "prediction", "selected_start_token", "selected_stop_token"} <= set(text_predictions.columns)
    assert {
        "contribution__feature_a",
        "contribution__feature_b",
        "top_positive_feature",
        "top_negative_feature",
    } <= set(scored_windows.columns)


def test_reusable_local_evidence_model_scores_supplied_windows_with_contributions() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)

    model, feature_columns = fit_local_evidence_model(
        windows,
        partitions,
        window_length=32,
        feature_columns=("feature_a", "feature_b"),
        seed=42,
    )
    supplied = windows.loc[windows["window_length"].eq(32)].copy()
    scored = score_local_evidence_windows(model, supplied, feature_columns)

    assert feature_columns == ("feature_a", "feature_b")
    assert len(scored) == len(supplied)
    assert scored["mg_score"].between(0.0, 1.0).all()
    assert {
        "contribution__feature_a",
        "contribution__feature_b",
        "top_positive_feature",
        "top_negative_feature",
    } <= set(scored.columns)
    assert scored["contribution__feature_a"].notna().all()


def test_fit_local_evidence_model_ignores_non_training_rows_and_features() -> None:
    windows = local_evidence_windows()
    partitions = split_local_evidence_texts(windows, seed=42)
    train_parents = set(partitions["train"].set_index(["split", "text_id"]).index)
    non_training = ~pd.MultiIndex.from_frame(windows[["split", "text_id"]]).isin(train_parents)
    mutated_windows = windows.copy()
    mutated_windows.loc[non_training, "feature_a"] = 1_000_000.0
    mutated_windows.loc[non_training, "feature_b"] = -1_000_000.0

    original_model, feature_columns = fit_local_evidence_model(
        windows,
        partitions,
        window_length=32,
        feature_columns=("feature_a", "feature_b"),
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
        feature_columns=("feature_a", "feature_b"),
        seed=42,
    )

    with pytest.raises(ValueError, match="missing feature columns"):
        score_local_evidence_windows(model, windows.drop(columns="feature_b"), ("feature_a", "feature_b"))


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
        feature_columns=("feature_a", "feature_b"),
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
        feature_columns=("feature_a", "feature_b"),
        seed=42,
        show_progress=False,
    )


def test_local_evidence_preserves_all_nan_training_features() -> None:
    windows = local_evidence_windows()
    windows["all_nan_feature"] = float("nan")
    partitions = split_local_evidence_texts(windows, seed=42)

    _, _, scored_windows = run_local_evidence_experiments(
        windows,
        partitions,
        feature_columns=("feature_a", "all_nan_feature"),
        seed=42,
        show_progress=False,
    )

    assert "contribution__all_nan_feature" in scored_windows
    assert scored_windows["contribution__all_nan_feature"].eq(0.0).all()


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
        feature_columns=("feature_a", "feature_b"),
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
        feature_columns=("feature_a", "feature_b"),
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
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    for required_source in (
        "WINDOW_LENGTHS = (32, 256)",
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
    assert len(code_cells) == 6
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
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for required_source in (
        "CUSTOM_TEXT_INDEX = 0",
        'CUSTOM_SPLIT = "MG"',
        "CUSTOM_WINDOW_LENGTH = 32",
        "load_text_row,",
        "load_text_row(PROCESSED_ROOT, DATASET, MODEL, CUSTOM_SPLIT, CUSTOM_TEXT_INDEX)",
        'custom_text_id = int(custom_row["id"])',
        'scored_windows["text_id"].eq(custom_text_id)',
        'scored_windows["window_length"].eq(CUSTOM_WINDOW_LENGTH)',
        "No scored windows available for custom text",
        'custom_windows.loc[custom_windows["mg_score"].idxmax()]',
        "plot_local_evidence_timeline(custom_windows, custom_prediction)",
        "CUSTOM_SPLIT,",
        "CUSTOM_TEXT_INDEX,",
    ):
        assert required_source in source


def test_local_evidence_notebook_does_not_merge_duplicate_evidence_columns() -> None:
    import json

    notebook_path = Path(__file__).parents[1] / "notebooks" / "local_evidence_classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "].merge(" not in source
    assert '"window_text"' not in source
    assert '"top_positive_feature"' not in source
    assert '"top_negative_feature"' not in source
