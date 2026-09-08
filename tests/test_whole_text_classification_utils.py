from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from notebooks.classification_utils import SCALAR_FEATURE_COLUMNS, evaluate_scores


METRIC_COLUMNS = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "hg_precision",
    "hg_recall",
    "hg_f1",
    "hg_support",
    "mg_precision",
    "mg_recall",
    "mg_f1",
    "mg_support",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positive_rate",
    "predicted_mg_rate",
}
RESULT_DIMENSIONS = {
    "scorer_llm",
    "source_datasets",
    "dataset",
    "partition",
    "classifier",
    "feature_set",
    "threshold",
    "direction",
}


def whole_text_utils():
    try:
        return importlib.import_module("notebooks.whole_text_classification_utils")
    except ModuleNotFoundError:
        pytest.fail("whole-text classification utility is missing")


def metric_row(row_id: int, split: str, value: float, **overrides: object) -> dict[str, object]:
    return {
        "id": row_id,
        "split": split,
        **{feature: value + index / 100 for index, feature in enumerate(SCALAR_FEATURE_COLUMNS)},
        **overrides,
    }


def write_metrics(
    processed_root: Path,
    dataset: str,
    scorer_llm: str,
    split: str,
    rows: list[dict[str, object]],
) -> None:
    path = processed_root / f"{dataset}__{scorer_llm}" / split / "metrics" / "metrics.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def cohort(dataset: str, partition_offset: int = 0, rows_per_label: int = 4) -> pd.DataFrame:
    rows = []
    for label, split in ((0, "HG"), (1, "MG")):
        for index in range(rows_per_label):
            row_id = partition_offset + index
            value = float(label * 10 + index)
            rows.append(
                {
                    "dataset": dataset,
                    "model": "scorer",
                    "split": split,
                    "id": row_id,
                    "label": label,
                    "identity": f"{dataset}:{split}:{row_id}",
                    **{
                        feature: value + feature_index / 100
                        for feature_index, feature in enumerate(SCALAR_FEATURE_COLUMNS)
                    },
                }
            )
    return pd.DataFrame(rows)


def stage1_validation_rows(validation: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row in validation.itertuples(index=False):
        tokens = [
            {"token": token, "logprob": -0.1, "top_logprobs": [{"logprob": -0.1}]}
            for token in ("one", " two", " three", " four")
        ]
        rows.append({"id": row.id, "split": row.split, "text": "one two three four", "tokens": tokens})
    return rows


def test_load_whole_text_cohort_filters_and_preserves_fixed_provenance(tmp_path: Path) -> None:
    utils = whole_text_utils()
    write_metrics(tmp_path, "source", "scorer", "HG", [metric_row(1, "HG", 1.0)])
    write_metrics(tmp_path, "source", "scorer", "MG", [metric_row(1, "MG", 9.0)])
    write_metrics(tmp_path, "source", "other", "HG", [metric_row(2, "HG", 2.0)])
    write_metrics(tmp_path, "target", "scorer", "HG", [metric_row(3, "HG", 3.0)])

    loaded = utils.load_whole_text_cohort(tmp_path, dataset="source", scorer_llm="scorer")

    assert list(loaded.columns) == [
        "dataset",
        "model",
        "split",
        "id",
        "label",
        "identity",
        *SCALAR_FEATURE_COLUMNS,
    ]
    assert loaded[["dataset", "model", "split", "id", "label"]].to_dict("records") == [
        {"dataset": "source", "model": "scorer", "split": "HG", "id": 1, "label": 0},
        {"dataset": "source", "model": "scorer", "split": "MG", "id": 1, "label": 1},
    ]
    assert loaded["identity"].tolist() == [("source", "HG", "1"), ("source", "MG", "1")]


def test_load_whole_text_cohort_reads_only_requested_metric_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    utils = whole_text_utils()
    write_metrics(tmp_path, "source", "scorer", "HG", [metric_row(1, "HG", 1.0)])
    write_metrics(tmp_path, "other", "scorer", "HG", [metric_row(2, "HG", 2.0)])
    calls: list[Path] = []
    original = pd.read_parquet

    def read_parquet(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    utils.load_whole_text_cohort(tmp_path, dataset="source", scorer_llm="scorer")

    assert {path.parents[2].name for path in calls} == {"source__scorer"}


def test_evaluate_whole_text_validation_lengths_reuses_bundle_and_adds_metadata() -> None:
    utils = whole_text_utils()
    validation = cohort("source", partition_offset=100)
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")}, {"source": validation}, scorer_llm="scorer"
    )

    result = utils.evaluate_whole_text_validation_lengths(
        fitted.bundle,
        {"source": validation},
        {"source": stage1_validation_rows(validation)},
        (2, 4),
    )

    assert set(result.metrics["validation_variant"]) == {"original", "first_N_words"}
    assert set(result.metrics.loc[result.metrics["validation_variant"].eq("first_N_words"), "word_limit"]) == {2, 4}
    assert result.predictions["word_limit"].notna().any()


def test_load_whole_text_cohort_rejects_missing_features_duplicates_and_empty_selection(tmp_path: Path) -> None:
    utils = whole_text_utils()
    duplicate = metric_row(1, "HG", 1.0)
    write_metrics(tmp_path, "duplicate", "scorer", "HG", [duplicate, duplicate])
    write_metrics(
        tmp_path,
        "missing",
        "scorer",
        "HG",
        [metric_row(2, "HG", 2.0, prompt_mu_score=None)],
    )
    missing_path = tmp_path / "missing__scorer" / "HG" / "metrics" / "metrics.parquet"
    missing_frame = pd.read_parquet(missing_path).drop(columns="prompt_mu_score")
    missing_frame.to_parquet(missing_path, index=False)

    with pytest.raises(ValueError, match="duplicate.*dataset, split, id"):
        utils.load_whole_text_cohort(tmp_path, dataset="duplicate", scorer_llm="scorer")
    with pytest.raises(ValueError, match="missing feature columns.*prompt_mu_score"):
        utils.load_whole_text_cohort(tmp_path, dataset="missing", scorer_llm="scorer")
    with pytest.raises(ValueError, match="empty cohort"):
        utils.load_whole_text_cohort(tmp_path, dataset="absent", scorer_llm="scorer")


def test_equivalent_integer_and_string_ids_are_rejected_as_duplicates() -> None:
    utils = whole_text_utils()
    duplicated = cohort("source")
    duplicate = duplicated.iloc[[0]].copy()
    duplicate["id"] = duplicate["id"].astype(str)

    with pytest.raises(ValueError, match="duplicate.*dataset, split, id"):
        utils.fit_whole_text_classifiers(
            {"source": pd.concat([duplicated, duplicate], ignore_index=True)},
            {"source": cohort("source", partition_offset=100)},
            scorer_llm="scorer",
        )


def test_equivalent_integer_and_string_ids_are_detected_as_partition_overlap() -> None:
    utils = whole_text_utils()
    validation = cohort("source", partition_offset=100)
    validation["id"] = validation["id"].astype(object)
    validation.loc[0, "id"] = "0"

    with pytest.raises(ValueError, match="overlap.*source"):
        utils.fit_whole_text_classifiers(
            {"source": cohort("source")},
            {"source": validation},
            scorer_llm="scorer",
        )


def test_null_ids_are_rejected_before_identity_validation() -> None:
    utils = whole_text_utils()
    invalid = cohort("source")
    invalid.loc[0, "id"] = None

    with pytest.raises(ValueError, match="null.*id"):
        utils.fit_whole_text_classifiers(
            {"source": invalid},
            {"source": cohort("source", partition_offset=100)},
            scorer_llm="scorer",
        )


def test_delimiter_bearing_ids_have_unambiguous_structured_identities() -> None:
    utils = whole_text_utils()
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")},
        {"source": cohort("source", partition_offset=100)},
        scorer_llm="scorer",
    )
    first = cohort("a", rows_per_label=1)
    first["id"] = first["id"].astype(object)
    first.loc[0, "id"] = "b:HG:c"
    second = cohort("a:HG:b", rows_per_label=1)
    second["id"] = second["id"].astype(object)
    second.loc[0, "id"] = "c"

    result = utils.evaluate_whole_text_classifiers(
        fitted.bundle,
        {"a": first, "a:HG:b": second},
        partition="test",
    )
    identities = result.predictions.loc[
        result.predictions["split"].eq("HG") & result.predictions["id"].isin(["b:HG:c", "c"]),
        "identity",
    ]

    assert identities.map(type).eq(tuple).all()
    assert identities.nunique() == 2


def test_fit_whole_text_classifiers_records_bundle_metrics_and_coverage() -> None:
    utils = whole_text_utils()
    result = utils.fit_whole_text_classifiers(
        {"alpha": cohort("alpha")},
        {"alpha": cohort("alpha", partition_offset=100)},
        scorer_llm="scorer",
    )

    assert result.bundle.scorer_llm == "scorer"
    assert result.bundle.source_datasets == ("alpha",)
    assert result.bundle.feature_columns == SCALAR_FEATURE_COLUMNS
    assert isinstance(result.bundle.logistic_pipeline, Pipeline)
    assert isinstance(result.bundle.logistic_pipeline.named_steps["impute"], SimpleImputer)
    assert result.bundle.logistic_pipeline.named_steps["impute"].strategy == "median"
    assert isinstance(result.bundle.logistic_pipeline.named_steps["scale"], StandardScaler)
    logistic = result.bundle.logistic_pipeline.named_steps["model"]
    assert isinstance(logistic, LogisticRegression)
    assert logistic.class_weight == "balanced"
    assert logistic.max_iter == 1000
    assert logistic.random_state == 42
    assert set(result.metrics["partition"]) == {"train", "validation"}
    assert set(result.metrics["classifier"]) == {"logistic_regression"}
    assert len(result.metrics) == 2
    assert result.coverage[["dataset", "partition", "rows", "hg_rows", "mg_rows"]].to_dict("records") == [
        {"dataset": "alpha", "partition": "train", "rows": 8, "hg_rows": 4, "mg_rows": 4},
        {"dataset": "alpha", "partition": "validation", "rows": 8, "hg_rows": 4, "mg_rows": 4},
    ]


def test_fit_uses_only_training_rows_for_logistic() -> None:
    utils = whole_text_utils()
    train = cohort("source")
    validation = cohort("source", partition_offset=100)
    reversed_validation = validation.copy()
    reversed_validation.loc[:, list(SCALAR_FEATURE_COLUMNS)] *= -1

    original = utils.fit_whole_text_classifiers(
        {"source": train}, {"source": validation}, scorer_llm="scorer"
    )
    mutated = utils.fit_whole_text_classifiers(
        {"source": train}, {"source": reversed_validation}, scorer_llm="scorer"
    )

    np.testing.assert_allclose(
        original.bundle.logistic_pipeline.predict_proba(train[list(SCALAR_FEATURE_COLUMNS)]),
        mutated.bundle.logistic_pipeline.predict_proba(train[list(SCALAR_FEATURE_COLUMNS)]),
    )


def test_fit_rejects_overlap_empty_missing_features_and_one_class_training() -> None:
    utils = whole_text_utils()
    valid = cohort("source")

    with pytest.raises(ValueError, match="overlap.*source"):
        utils.fit_whole_text_classifiers({"source": valid}, {"source": valid}, scorer_llm="scorer")
    with pytest.raises(ValueError, match="empty cohort.*source.*train"):
        utils.fit_whole_text_classifiers({"source": valid.iloc[0:0]}, {"source": valid}, scorer_llm="scorer")
    with pytest.raises(ValueError, match="missing feature columns.*perplexity"):
        utils.fit_whole_text_classifiers(
            {"source": valid.drop(columns="perplexity")}, {"source": valid}, scorer_llm="scorer"
        )
    with pytest.raises(ValueError, match="training data must contain both classes"):
        utils.fit_whole_text_classifiers(
            {"source": valid[valid["label"].eq(0)]},
            {"source": cohort("source", partition_offset=100)},
            scorer_llm="scorer",
        )


def test_fit_rejects_all_missing_training_features() -> None:
    utils = whole_text_utils()
    train = cohort("source")
    train["perplexity"] = np.nan
    validation = cohort("source", partition_offset=100)

    with pytest.raises(ValueError, match="training data has no finite values.*perplexity"):
        utils.fit_whole_text_classifiers(
            {"source": train},
            {"source": validation},
            scorer_llm="scorer",
        )


def test_fit_rejects_duplicate_feature_names() -> None:
    utils = whole_text_utils()

    with pytest.raises(ValueError, match="duplicate feature names.*prompt_mu_score"):
        utils.fit_whole_text_classifiers(
            {"source": cohort("source")},
            {"source": cohort("source", partition_offset=100)},
            scorer_llm="scorer",
            feature_columns=("prompt_mu_score", "prompt_mu_score"),
        )


def test_evaluate_frozen_bundle_retains_dataset_identity_and_required_outputs() -> None:
    utils = whole_text_utils()
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")},
        {"source": cohort("source", partition_offset=100)},
        scorer_llm="scorer",
    )
    coefficients = fitted.bundle.logistic_pipeline.named_steps["model"].coef_.copy()
    targets = {"target_a": cohort("target_a", rows_per_label=2), "target_b": cohort("target_b", rows_per_label=2)}

    result = utils.evaluate_whole_text_classifiers(fitted.bundle, targets, partition="test")

    assert set(result.metrics["dataset"]) == {"target_a", "target_b"}
    assert set(result.metrics["partition"]) == {"test"}
    assert set(result.predictions["dataset"]) == {"target_a", "target_b"}
    assert result.predictions.groupby(["dataset", "classifier", "feature_set"]).size().eq(4).all()
    assert set(result.predictions["classifier"]) == {"logistic_regression"}
    assert {
        "scorer_llm",
        "source_datasets",
        "dataset",
        "partition",
        "classifier",
        "feature_set",
        "threshold",
        "direction",
        "split",
        "id",
        "identity",
        "label",
        "mg_score",
        "prediction",
    } <= set(result.predictions.columns)
    assert set(result.predictions.loc[result.predictions["id"].eq(0), "identity"]) == {
        ("target_a", "HG", "0"),
        ("target_a", "MG", "0"),
        ("target_b", "HG", "0"),
        ("target_b", "MG", "0"),
    }
    np.testing.assert_array_equal(coefficients, fitted.bundle.logistic_pipeline.named_steps["model"].coef_)


def test_emitted_scores_equal_direct_frozen_model_scores() -> None:
    utils = whole_text_utils()
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")},
        {"source": cohort("source", partition_offset=100)},
        scorer_llm="scorer",
    )
    target = cohort("target", rows_per_label=2)

    result = utils.evaluate_whole_text_classifiers(fitted.bundle, {"target": target}, partition="test")

    logistic_rows = result.predictions[result.predictions["classifier"].eq("logistic_regression")]
    np.testing.assert_allclose(
        logistic_rows["mg_score"],
        fitted.bundle.logistic_pipeline.predict_proba(target[list(SCALAR_FEATURE_COLUMNS)])[:, 1],
    )


def test_evaluation_metrics_and_coverage_have_required_schema_and_values() -> None:
    utils = whole_text_utils()
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")},
        {"source": cohort("source", partition_offset=100)},
        scorer_llm="scorer",
    )
    target = cohort("target", rows_per_label=2)

    result = utils.evaluate_whole_text_classifiers(fitted.bundle, {"target": target}, partition="validation")

    assert set(result.metrics.columns) == RESULT_DIMENSIONS | METRIC_COLUMNS
    assert set(result.coverage.columns) == RESULT_DIMENSIONS | {
        "requested_rows",
        "rows",
        "hg_rows",
        "mg_rows",
    }
    assert len(result.metrics) == 1
    assert len(result.coverage) == 1
    assert result.coverage["requested_rows"].eq(4).all()
    assert result.coverage["rows"].eq(4).all()
    assert result.coverage["hg_rows"].eq(2).all()
    assert result.coverage["mg_rows"].eq(2).all()
    logistic = result.predictions[result.predictions["classifier"].eq("logistic_regression")]
    expected = evaluate_scores(target["label"], logistic["mg_score"], 0.5)
    logistic_metrics = result.metrics[result.metrics["classifier"].eq("logistic_regression")].iloc[0]
    for metric, value in expected.items():
        assert logistic_metrics[metric] == pytest.approx(value, nan_ok=True)


def test_evaluate_supports_one_class_target_and_rejects_invalid_cohorts() -> None:
    utils = whole_text_utils()
    fitted = utils.fit_whole_text_classifiers(
        {"source": cohort("source")},
        {"source": cohort("source", partition_offset=100)},
        scorer_llm="scorer",
    )
    hg_only = cohort("bbc").loc[lambda frame: frame["label"].eq(0)].reset_index(drop=True)

    result = utils.evaluate_whole_text_classifiers(fitted.bundle, {"bbc": hg_only}, partition="test")

    assert result.metrics["roc_auc"].isna().all()
    assert result.metrics["pr_auc"].isna().all()
    assert result.metrics["balanced_accuracy"].isna().all()
    assert result.metrics["hg_support"].eq(len(hg_only)).all()
    assert result.metrics["mg_support"].eq(0).all()
    with pytest.raises(ValueError, match="empty cohort.*bbc"):
        utils.evaluate_whole_text_classifiers(fitted.bundle, {"bbc": hg_only.iloc[0:0]}, partition="test")
    with pytest.raises(ValueError, match="duplicate.*dataset, split, id"):
        utils.evaluate_whole_text_classifiers(
            fitted.bundle,
            {"bbc": pd.concat([hg_only, hg_only.iloc[[0]]], ignore_index=True)},
            partition="test",
        )
