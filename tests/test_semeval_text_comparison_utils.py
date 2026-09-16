from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from notebooks.utils.semeval_text_comparison_utils import (
    plot_boundary_and_regions,
    plot_reconstructed_token_scores,
    plot_window_level_scores,
    select_semeval_comparison_texts,
)


def _evaluations() -> pd.DataFrame:
    rows = []
    scores = {1: (1.0, 0.8), 2: (0.1, 0.2), 3: (0.5, 0.5), 4: (0.9, 0.9), 5: (0.3, 0.3), 6: (0.7, 0.7)}
    for text_id, values in scores.items():
        for window_length, mg_f1 in zip((32, 64), values):
            rows.append({
                "scorer_llm": "littlelamb", "feature_set": "extended",
                "model": "logistic_regression", "model_family": "logistic_regression",
                "evaluation_dataset": "semeval2024_task8_subtask_c_train", "semeval_split": "train",
                "text_id": text_id, "window_length": window_length, "mg_f1": mg_f1,
            })
    return pd.DataFrame(rows)


def _scored_windows() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "logistic_regression", "window_length": 4, "text_id": 11, "start_token": 0, "stop_token": 4, "mg_score": 0.1},
        {"model": "logistic_regression", "window_length": 4, "text_id": 11, "start_token": 4, "stop_token": 8, "mg_score": 0.8},
        {"model": "logistic_regression", "window_length": 8, "text_id": 11, "start_token": 0, "stop_token": 8, "mg_score": 0.2},
        {"model": "logistic_regression", "window_length": 8, "text_id": 11, "start_token": 8, "stop_token": 16, "mg_score": 0.9},
    ])


def _boundary_regions() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "logistic_regression", "window_length": 4, "text_id": 11, "token_count": 16, "gold_token_boundary": 8, "predicted_token_boundary": 7, "predicted_mg_region_start": 8, "predicted_mg_region_stop": 16, "no_crossing": False},
        {"model": "logistic_regression", "window_length": 8, "text_id": 11, "token_count": 16, "gold_token_boundary": 8, "predicted_token_boundary": 9, "predicted_mg_region_start": 9, "predicted_mg_region_stop": 12, "no_crossing": False},
        {"model": "logistic_regression", "window_length": 8, "text_id": 11, "token_count": 16, "gold_token_boundary": 8, "predicted_token_boundary": 9, "predicted_mg_region_start": 13, "predicted_mg_region_stop": 16, "no_crossing": False},
    ])


def test_selects_best_worst_and_random_groups_deterministically() -> None:
    selected = select_semeval_comparison_texts(_evaluations(), count=2, seed=7)
    assert selected.loc[selected["group"].eq("best"), "text_id"].tolist() == [1, 4]
    assert selected.loc[selected["group"].eq("worst"), "text_id"].tolist() == [2, 5]
    assert selected.equals(select_semeval_comparison_texts(_evaluations(), count=2, seed=7))
    assert {"aggregate_mg_f1", "mg_f1_window_32", "mg_f1_window_64"}.issubset(selected.columns)


def test_filters_feature_set_and_allows_undersized_inputs() -> None:
    evaluations = pd.concat([_evaluations(), _evaluations().assign(feature_set="scalar_all")], ignore_index=True)
    selected = select_semeval_comparison_texts(evaluations, feature_set="extended", count=20, seed=1)
    assert len(selected.loc[selected["group"].eq("best")]) == 6
    assert set(selected["feature_set"]) == {"extended"}


def test_plot_builders_return_expected_axes() -> None:
    boundary = plot_boundary_and_regions(_boundary_regions(), text_id=11, selection_group="worst")
    window = plot_window_level_scores(_scored_windows(), text_id=11, gold_boundary=8, selection_group="best")
    token = plot_reconstructed_token_scores(_scored_windows(), text_id=11, token_count=16, gold_boundary=8, selection_group="random")
    assert len(boundary.axes) == 1
    assert len(window.axes) == 1
    assert len(token.axes) == 1
    assert len(boundary.axes[0].lines) >= 2
    assert len(window.axes[0].lines) >= 2
    assert len(token.axes[0].lines) >= 2
    assert token.axes[0].get_ylim() == (0.0, 1.0)
    for figure in (boundary, window, token):
        plt.close(figure)
