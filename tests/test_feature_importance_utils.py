import pytest
import pandas as pd

from notebooks.utils.feature_importance_utils import (
    importance_run_path,
    metric_degradation,
    permute_feature_by_parent,
    summarize_importance,
)


def test_permute_feature_by_parent_keeps_window_values_together():
    frame = pd.DataFrame(
        {
            "split": ["HG", "HG", "MG", "MG"],
            "text_id": [1, 1, 2, 2],
            "feature": [10.0, 10.0, 20.0, 20.0],
            "later_feature": [1.0, 1.0, 2.0, 2.0],
        }
    )

    result = permute_feature_by_parent(frame, feature="feature", seed=1)

    assert result.columns.tolist() == frame.columns.tolist()
    assert result.groupby(["split", "text_id"])["feature"].nunique().eq(1).all()
    assert sorted(result.groupby(["split", "text_id"])["feature"].first()) == [10.0, 20.0]


def test_permute_feature_by_parent_separates_matching_ids_in_each_split():
    frame = pd.DataFrame(
        {
            "split": ["HG", "HG", "MG", "MG"],
            "text_id": [1, 1, 1, 1],
            "feature": [10.0, 10.0, 20.0, 20.0],
        }
    )

    result = permute_feature_by_parent(frame, feature="feature", seed=1)

    assert result.groupby(["split", "text_id"])["feature"].nunique().eq(1).all()
    assert sorted(result.groupby(["split", "text_id"])["feature"].first()) == [10.0, 20.0]


def test_metric_degradation_orients_f1_and_boundary_mae():
    assert metric_degradation(metric="macro_f1", baseline=0.9, permuted=0.8) == pytest.approx(0.1)
    assert metric_degradation(metric="boundary_mae", baseline=5.0, permuted=7.0) == pytest.approx(2.0)


def test_summarize_importance_calculates_interval_and_rank():
    observations = pd.DataFrame(
        {
            "configuration_id": ["a"] * 4,
            "metric": ["macro_f1"] * 4,
            "feature": ["x", "x", "y", "y"],
            "degradation": [0.4, 0.2, 0.1, 0.1],
        }
    )

    result = summarize_importance(observations)

    assert result.loc[result["feature"].eq("x"), "mean_degradation"].item() == pytest.approx(0.3)
    assert result.loc[result["feature"].eq("x"), "rank"].item() == 1


def test_importance_run_path_is_outside_models_tree(tmp_path):
    result = importance_run_path(tmp_path, run_id="importance-1")

    assert result == tmp_path / "experiment-results" / "classification" / "feature-importance" / "importance-1"
    assert "models" not in result.parts
