from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from notebooks.classification_utils import (
    build_canonical_cohort,
    evaluate_scores,
    fit_single_feature_threshold,
    load_baseline_cohort,
    run_baseline_experiments,
    select_model_by_coverage,
    split_cohort,
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
