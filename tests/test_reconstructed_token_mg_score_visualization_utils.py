import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
    align_text_records,
    build_excerpt_rows,
    load_text_records,
    load_visualization_models,
    plot_passage_scores,
    reconstruct_token_scores,
    save_visualization_outputs,
    select_visualization_examples,
    score_prepared_features,
    score_prepared_partition,
    semeval_boundary_errors_from_scored_windows,
)


def test_reconstruct_token_scores_averages_overlaps_per_model_and_keeps_gaps():
    scored_windows = pd.DataFrame(
        [
            {"text_id": "a", "model": "lr", "start_token": 0, "stop_token": 3, "mg_score": 0.2},
            {"text_id": "a", "model": "lr", "start_token": 1, "stop_token": 4, "mg_score": 0.8},
            {"text_id": "a", "model": "rf", "start_token": 0, "stop_token": 2, "mg_score": 0.9},
        ]
    )

    result = reconstruct_token_scores(scored_windows, text_id="a", token_count=5)
    lr_scores = result.loc[result["model"].eq("lr")].sort_values("token_position")["mg_score"].to_numpy()
    rf_scores = result.loc[result["model"].eq("rf")].sort_values("token_position")["mg_score"].to_numpy()

    np.testing.assert_allclose(lr_scores[:4], [0.2, 0.5, 0.5, 0.8])
    assert np.isnan(lr_scores[4])
    np.testing.assert_allclose(rf_scores[:2], [0.9, 0.9])
    assert np.isnan(rf_scores[2:]).all()


def test_reconstruct_token_score_dispersion_returns_mean_population_std_and_window_count():
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        reconstruct_token_score_dispersion,
    )

    scored_windows = pd.DataFrame(
        [
            {"text_id": "a", "model": "lr", "start_token": 0, "stop_token": 3, "mg_score": 0.2},
            {"text_id": "a", "model": "lr", "start_token": 1, "stop_token": 4, "mg_score": 0.8},
        ]
    )

    result = reconstruct_token_score_dispersion(scored_windows, text_id="a", token_count=5)
    lr = result.loc[result["model"].eq("lr")].sort_values("token_position")

    np.testing.assert_allclose(lr["mg_score_mean"].iloc[:4], [0.2, 0.5, 0.5, 0.8])
    np.testing.assert_allclose(lr["mg_score_std"].iloc[:4], [0.0, 0.3, 0.3, 0.0])
    assert lr["window_count"].tolist() == [1, 2, 2, 1, 0]
    assert np.isnan(lr["mg_score_mean"].iloc[4])
    assert np.isnan(lr["mg_score_std"].iloc[4])


@pytest.mark.parametrize(
    "window, message",
    [
        ({"start_token": -1, "stop_token": 2, "mg_score": 0.5}, "invalid scoring span"),
        ({"start_token": 0, "stop_token": 2, "mg_score": 1.1}, "out-of-range MG score"),
    ],
)
def test_reconstruct_token_score_dispersion_rejects_invalid_window_values(window, message):
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        reconstruct_token_score_dispersion,
    )

    scored_windows = pd.DataFrame([{"text_id": "a", "model": "lr", **window}])

    with pytest.raises(ValueError, match=message):
        reconstruct_token_score_dispersion(scored_windows, text_id="a", token_count=5)


def test_detect_window_run_boundaries_uses_first_hg_to_mg_transition_per_model():
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        detect_window_run_boundaries,
    )

    scored_windows = pd.DataFrame(
        [
            {"model": "logistic_regression", "start_token": start, "stop_token": start + 64, "mg_score": score}
            for start, score in ((80, 0.7), (32, 0.49), (0, 0.1), (64, 0.8), (16, 0.4), (48, 0.5))
        ]
        + [
            {"model": "random_forest", "start_token": start, "stop_token": start + 64, "mg_score": score}
            for start, score in zip((0, 16, 32, 48, 64, 80), (0.1, 0.7, 0.2, 0.8, 0.3, 0.9), strict=True)
        ]
    )

    result = detect_window_run_boundaries(scored_windows, run_length=3).set_index("model")

    assert result.loc["logistic_regression", "status"] == "found"
    assert result.loc["logistic_regression", "predicted_token_boundary"] == pytest.approx(72.0)
    assert result.loc["random_forest", "status"] == "not_found"
    assert np.isnan(result.loc["random_forest", "predicted_token_boundary"])


@pytest.mark.parametrize(
    "scored_windows, threshold, run_length, message",
    [
        (
            pd.DataFrame([{"model": "lr", "start_token": -1, "stop_token": 63, "mg_score": 0.2}]),
            0.5,
            3,
            "invalid scoring span",
        ),
        (
            pd.DataFrame([{"model": "lr", "start_token": 0, "stop_token": 64, "mg_score": 1.2}]),
            0.5,
            3,
            "out-of-range MG score",
        ),
        (
            pd.DataFrame([{"model": "lr", "start_token": 0, "stop_token": 64, "mg_score": 0.2}]),
            1.1,
            3,
            r"threshold must be in \[0, 1\]",
        ),
        (
            pd.DataFrame([{"model": "lr", "start_token": 0, "stop_token": 64, "mg_score": 0.2}]),
            0.5,
            0,
            "run_length must be a positive integer",
        ),
    ],
)
def test_detect_window_run_boundaries_rejects_invalid_inputs(scored_windows, threshold, run_length, message):
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        detect_window_run_boundaries,
    )

    with pytest.raises(ValueError, match=message):
        detect_window_run_boundaries(scored_windows, threshold=threshold, run_length=run_length)


def test_plot_window_boundary_profile_marks_threshold_gold_and_found_models_only():
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        plot_window_boundary_profile,
    )

    scored_windows = pd.DataFrame(
        [
            {"model": model, "start_token": start, "stop_token": start + 64, "mg_score": score}
            for model, values in (
                ("logistic_regression", (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)),
                ("random_forest", (0.9, 0.8, 0.2, 0.7, 0.1, 0.6)),
            )
            for start, score in zip((0, 16, 32, 48, 64, 80), values, strict=True)
        ]
    )
    boundary_predictions = pd.DataFrame(
        [
            {"model": "logistic_regression", "status": "found", "predicted_token_boundary": 72.0},
            {"model": "random_forest", "status": "not_found", "predicted_token_boundary": np.nan},
        ]
    )

    figure = plot_window_boundary_profile(
        scored_windows,
        boundary_predictions,
        gold_boundary=80,
        threshold=0.5,
        title="window boundary test",
    )

    labels = [line.get_label() for line in figure.axes[0].lines]
    assert "Decision threshold (0.50)" in labels
    assert "Gold boundary" in labels
    assert "Logistic Regression boundary" in labels
    assert "Random Forest boundary" not in labels
    predicted_line = next(line for line in figure.axes[0].lines if line.get_label() == "Logistic Regression boundary")
    np.testing.assert_allclose(predicted_line.get_xdata(), [72.0, 72.0])


def test_plot_passage_score_dispersion_shades_clipped_mean_plus_minus_std():
    from matplotlib.collections import PolyCollection

    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        plot_passage_score_dispersion,
    )

    scores = pd.DataFrame(
        [
            {
                "model": model,
                "token_position": position,
                "mg_score_mean": mean,
                "mg_score_std": std,
                "window_count": 2,
            }
            for model in ("logistic_regression", "random_forest")
            for position, mean, std in ((0, 0.9, 0.3), (1, 0.1, 0.3))
        ]
    )

    figure = plot_passage_score_dispersion(scores, title="dispersion test")

    axis = figure.axes[0]
    assert len(axis.lines) == 2
    bands = [collection for collection in axis.collections if isinstance(collection, PolyCollection)]
    assert len(bands) == 2
    for band in bands:
        vertices = band.get_paths()[0].vertices
        assert vertices[:, 1].min() >= 0.0
        assert vertices[:, 1].max() <= 1.0


@pytest.mark.parametrize(
    "window, message",
    [
        ({"start_token": -1, "stop_token": 2, "mg_score": 0.5}, "invalid scoring span"),
        ({"start_token": 1, "stop_token": 1, "mg_score": 0.5}, "invalid scoring span"),
        ({"start_token": 0, "stop_token": 6, "mg_score": 0.5}, "invalid scoring span"),
        ({"start_token": 0, "stop_token": 2, "mg_score": 1.1}, "out-of-range MG score"),
    ],
)
def test_reconstruct_token_scores_rejects_invalid_window_values(window, message):
    scored_windows = pd.DataFrame([{"text_id": "a", "model": "lr", **window}])

    with pytest.raises(ValueError, match=message):
        reconstruct_token_scores(scored_windows, text_id="a", token_count=5)


def test_align_text_records_joins_source_text_and_stage1_token_surfaces():
    source = pd.DataFrame(
        [{"id": " 17 ", "text": "Hello world", "boundary_word_index": 1}]
    )
    token_records = pd.DataFrame([{"id": 17, "tokens": [{"token": "Hello"}, {"token": " world"}]}])

    result = align_text_records(source, token_records, dataset="semeval")

    assert result.loc[0, "text_id"] == "17"
    assert result.loc[0, "tokens"] == ["Hello", " world"]
    assert result.loc[0, "token_count"] == 2
    assert result.loc[0, "boundary_word_index"] == 1


def test_align_text_records_rejects_duplicate_or_missing_token_ids():
    source = pd.DataFrame([{"id": "1", "text": "one", "source_llm": "model-a"}])
    token_records = pd.DataFrame(
        [
            {"id": "1", "tokens": [{"token": "one"}]},
            {"id": "1", "tokens": [{"token": "one"}]},
        ]
    )

    with pytest.raises(ValueError, match="duplicate token IDs"):
        align_text_records(source, token_records, dataset="name_that_llm")


def test_score_prepared_features_preserves_window_spans_and_adds_mg_scores():
    from sklearn.linear_model import LogisticRegression

    from notebooks.utils.model_training_utils import FittedFeatureVectorModel

    estimator = LogisticRegression().fit(pd.DataFrame({"feature": [0.0, 1.0]}), [0, 1])
    model = FittedFeatureVectorModel("logistic_regression", ("feature",), estimator)
    prepared = pd.DataFrame(
        [
            {"split": "MG", "text_id": "a", "label": 1, "start_token": 0, "stop_token": 64, "feature": 0.0},
            {"split": "MG", "text_id": "a", "label": 1, "start_token": 16, "stop_token": 80, "feature": 1.0},
        ]
    )

    scored = score_prepared_features(prepared, model)

    assert scored[["start_token", "stop_token"]].values.tolist() == [[0, 64], [16, 80]]
    assert scored["mg_score"].between(0, 1).all()


def test_load_visualization_models_requires_exact_artifact_metadata(monkeypatch, tmp_path):
    from notebooks.utils import reconstructed_token_mg_score_visualization_utils as utils

    artifact_paths = {"logistic_regression": tmp_path / "lr", "random_forest": tmp_path / "rf"}
    loaded = {"lr": object(), "rf": object()}
    filters_seen = []

    def discover(_root, *, filters):
        filters_seen.append(filters)
        return [artifact_paths[filters["model_family"]]]

    def metadata(path, *, expected):
        return {**expected, "feature_columns": list(utils.SIMPLE_SIMILARITY_COLUMNS)}

    monkeypatch.setattr(utils, "discover_model_artifacts", discover)
    monkeypatch.setattr(utils, "load_model_metadata", metadata)
    monkeypatch.setattr(
        utils,
        "load_model_artifact",
        lambda path, **kwargs: loaded["lr" if path.name == "lr" else "rf"],
    )

    result = load_visualization_models(tmp_path, run_id="saved-run")

    assert result == {"logistic_regression": loaded["lr"], "random_forest": loaded["rf"]}
    assert {filters["training_regime"] for filters in filters_seen} == {"Kaggle+MAGE+Outfox"}


def test_score_prepared_partition_loads_existing_partition_without_preparing(monkeypatch, tmp_path):
    from sklearn.linear_model import LogisticRegression

    from notebooks.utils import reconstructed_token_mg_score_visualization_utils as utils
    from notebooks.utils.model_training_utils import FittedFeatureVectorModel

    columns = utils.SIMPLE_SIMILARITY_COLUMNS
    estimator = LogisticRegression().fit(pd.DataFrame({column: [0.0, 1.0] for column in columns}), [0, 1])
    model = FittedFeatureVectorModel("logistic_regression", columns, estimator)
    prepared = pd.DataFrame(
        [{
            "split": "MG", "text_id": "a", "label": 1, "start_token": 0, "stop_token": 64,
            **{column: 0.5 for column in columns},
        }]
    )
    monkeypatch.setattr(utils, "prepared_partition_path", lambda *args, **kwargs: tmp_path / "cached")
    monkeypatch.setattr(utils, "load_prepared_partition", lambda path, *, expected: prepared)

    result = score_prepared_partition(
        tmp_path,
        dataset="name_that_llm_train",
        split="MG",
        model=model,
    )

    assert result.loc[0, "model"] == "logistic_regression"
    assert result.loc[0, "text_id"] == "a"
    assert result.loc[0, "start_token"] == 0


def test_load_text_records_uses_only_stage1_available_name_that_llm_rows(tmp_path):
    raw = tmp_path / "raw" / "name-that-llm"
    stage1 = tmp_path / "processed" / "name_that_llm_train__littlelamb" / "MG" / "data"
    raw.mkdir(parents=True)
    stage1.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": 1, "text": "one", "source_llm": "model-a"},
            {"id": 2, "text": "two", "source_llm": "model-b"},
            {"id": 3, "text": "not processed", "source_llm": "model-c"},
        ]
    ).to_csv(raw / "name_that_llm_train_MG.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "tokens": [{"token": "one"}]},
            {"id": 2, "tokens": [{"token": "two"}]},
        ]
    ).to_parquet(stage1 / "batch.parquet", index=False)

    result = load_text_records(tmp_path, dataset="name_that_llm_train")

    assert result["text_id"].tolist() == ["1", "2"]
    assert result["source_llm"].tolist() == ["model-a", "model-b"]
    assert result["tokens"].tolist() == [["one"], ["two"]]


def test_load_text_records_loads_bbc_hg_text_and_stage1_tokens(tmp_path):
    raw = tmp_path / "raw" / "bbc_news"
    stage1 = tmp_path / "processed" / "bbc_news_content__littlelamb" / "HG" / "data"
    raw.mkdir(parents=True)
    stage1.mkdir(parents=True)
    pd.DataFrame([{"id": 17, "text": "BBC article", "origin": "HG"}]).to_csv(
        raw / "bbc_news_content_HG.csv", index=False
    )
    pd.DataFrame([{"id": 17, "tokens": [{"token": "BBC"}, {"token": " article"}]}]).to_parquet(
        stage1 / "batch.parquet", index=False
    )

    result = load_text_records(tmp_path, dataset="bbc_news_content")

    assert result[["text_id", "text", "tokens", "token_count"]].to_dict("records") == [
        {"text_id": "17", "text": "BBC article", "tokens": ["BBC", " article"], "token_count": 2}
    ]


def test_score_prepared_features_allows_hg_label_zero_as_mg_score():
    from sklearn.linear_model import LogisticRegression

    from notebooks.utils.model_training_utils import FittedFeatureVectorModel

    estimator = LogisticRegression().fit(pd.DataFrame({"feature": [0.0, 1.0]}), [0, 1])
    model = FittedFeatureVectorModel("logistic_regression", ("feature",), estimator)
    prepared = pd.DataFrame(
        [{"split": "HG", "text_id": "17", "label": 0, "start_token": 0, "stop_token": 2, "feature": 0.0}]
    )

    scored = score_prepared_features(prepared, model, split="HG")

    assert scored.loc[0, "mg_score"] == pytest.approx(model.estimator.predict_proba(pd.DataFrame({"feature": [0.0]}))[0, 1])


def test_select_visualization_examples_ranks_both_datasets_and_manual_ids():
    token_scores = pd.DataFrame(
        [
            {"dataset": "name_that_llm_train", "text_id": text_id, "model": model, "token_position": pos, "mg_score": score}
            for text_id, lr, rf in [("a", 0.1, 0.2), ("b", 0.5, 0.5), ("c", 0.9, 0.8), ("d", 0.2, 0.9)]
            for model, score in [("logistic_regression", lr), ("random_forest", rf)]
            for pos in range(2)
        ]
        + [
            {"dataset": "semeval", "text_id": text_id, "model": model, "token_position": 0, "mg_score": 0.5}
            for text_id in ("1", "2", "3")
            for model in ("logistic_regression", "random_forest")
        ]
    )
    boundary_errors = pd.DataFrame(
        [
            {"text_id": text_id, "model": model, "absolute_token_error": error}
            for text_id, error in [("1", 1), ("2", 5), ("3", 10)]
            for model in ("logistic_regression", "random_forest")
        ]
    )

    selected = select_visualization_examples(
        token_scores,
        semeval_boundary_errors=boundary_errors,
        manual_ids={"name_that_llm_train": ["b"]},
        seed=7,
    )

    assert {"semeval_good", "semeval_typical", "semeval_poor"}.issubset(set(selected["selection_group"]))
    assert {"name_that_llm_low", "name_that_llm_middle", "name_that_llm_high", "name_that_llm_disagreement"}.issubset(set(selected["selection_group"]))
    assert ((selected["dataset"] == "name_that_llm_train") & (selected["text_id"] == "b") & (selected["selection_group"] == "manual")).any()


def test_build_excerpt_rows_centers_by_gold_boundary_or_classifier_disagreement():
    token_scores = pd.DataFrame(
        [
            {"text_id": "t", "model": model, "token_position": pos, "mg_score": score}
            for model, scores in [("logistic_regression", [0.1, 0.2, 0.3, 0.4, 0.5]), ("random_forest", [0.1, 0.9, 0.3, 0.4, 0.5])]
            for pos, score in enumerate(scores)
        ]
    )

    semeval = build_excerpt_rows(
        token_scores,
        ["a", "b", "c", "d", "e"],
        dataset="semeval",
        excerpt_width=3,
        gold_boundary=3,
    )
    name_that_llm = build_excerpt_rows(
        token_scores,
        ["a", "b", "c", "d", "e"],
        dataset="name_that_llm_train",
        excerpt_width=3,
    )

    assert semeval["token_position"].tolist() == [2, 3, 4]
    assert semeval["is_gold_boundary"].tolist() == [False, True, False]
    assert name_that_llm["token_position"].tolist() == [0, 1, 2]
    assert "is_gold_boundary" not in name_that_llm


def test_build_semeval_boundary_errors_uses_saved_selected_configuration():
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        semeval_boundary_errors_from_results,
    )

    predictions = pd.DataFrame(
        [
            {
                "text_id": "1", "model": model, "window_length": 64,
                "training_window": 64, "predicted_token_boundary": predicted,
                "scorer_llm": "littlelamb", "feature_set": "simple_similarity",
                "training_regime": "Mixed",
                "artifact_run_id": "selected-artifact",
                "evaluation_dataset": "semeval2024_task8_subtask_c_train",
                "semeval_split": "train",
            }
            for model, predicted in [("logistic_regression", 4), ("random_forest", 7)]
        ]
        + [
            {
                "text_id": "1", "model": model, "window_length": 64,
                "training_window": 64, "predicted_token_boundary": 99,
                "scorer_llm": "littlelamb", "feature_set": "simple_similarity",
                "training_regime": "Mixed",
                "artifact_run_id": "selected-artifact",
                "evaluation_dataset": "semeval2024_task8_subtask_c_dev",
                "semeval_split": "dev",
            }
            for model in ("logistic_regression", "random_forest")
        ]
        + [
            {
                "text_id": "1", "model": model, "window_length": 64,
                "training_window": 64, "predicted_token_boundary": 99,
                "scorer_llm": "littlelamb", "feature_set": "simple_similarity",
                "training_regime": "Mixed", "artifact_run_id": "other-artifact",
                "evaluation_dataset": "semeval2024_task8_subtask_c_train",
                "semeval_split": "train",
            }
            for model in ("logistic_regression", "random_forest")
        ]
    )
    records = pd.DataFrame(
        [{"text_id": "1", "text": "one two three four", "boundary_word_index": 2, "tokens": ["one", " two", " three", " four"]}]
    )

    errors = semeval_boundary_errors_from_results(
        predictions, records, artifact_run_id="selected-artifact", training_regime="Mixed"
    )

    assert errors["model"].tolist() == ["logistic_regression", "random_forest"]
    assert errors["absolute_token_error"].tolist() == [2, 5]


def test_semeval_boundary_errors_are_derived_from_the_selected_models_scores():
    token_records = pd.DataFrame(
        [{
            "text_id": "1", "text": "one two three four five", "boundary_word_index": 2,
            "tokens": ["one", " two", " three", " four", " five"],
        }]
    )
    scored_windows = pd.DataFrame(
        [
            {"text_id": "1", "model": "logistic_regression", "start_token": 0, "stop_token": 2, "mg_score": 0.1},
            {"text_id": "1", "model": "logistic_regression", "start_token": 2, "stop_token": 5, "mg_score": 0.9},
            {"text_id": "1", "model": "random_forest", "start_token": 0, "stop_token": 3, "mg_score": 0.1},
            {"text_id": "1", "model": "random_forest", "start_token": 3, "stop_token": 5, "mg_score": 0.9},
        ]
    )

    errors = semeval_boundary_errors_from_scored_windows(
        scored_windows,
        token_records,
        smoothing_window=1,
        sustained_length=2,
    )

    assert errors["model"].tolist() == ["logistic_regression", "random_forest"]
    assert errors["gold_token_boundary"].tolist() == [2, 2]
    assert errors["predicted_token_boundary"].tolist() == [2, 3]
    assert errors["absolute_token_error"].tolist() == [0, 1]


def test_semeval_boundary_errors_report_and_skip_unlabelable_annotations():
    token_records = pd.DataFrame(
        [
            {"text_id": "1", "text": "one two three", "boundary_word_index": 1, "tokens": ["one", " two", " three"]},
            {"text_id": "2", "text": "four five", "boundary_word_index": 99, "tokens": ["four", " five"]},
        ]
    )
    scored_windows = pd.DataFrame(
        [
            {
                "text_id": text_id, "model": model, "start_token": 0, "stop_token": 3 if text_id == "1" else 2,
                "mg_score": 0.9,
            }
            for text_id in ("1", "2")
            for model in ("logistic_regression", "random_forest")
        ]
    )

    errors = semeval_boundary_errors_from_scored_windows(
        scored_windows,
        token_records,
        smoothing_window=1,
        sustained_length=2,
    )

    assert errors["text_id"].unique().tolist() == ["1"]
    exclusions = errors.attrs["exclusions"]
    assert exclusions["text_id"].unique().tolist() == ["2"]
    assert exclusions["exclusion_reason"].str.startswith("boundary_word_index is outside").all()


def test_plot_passage_scores_has_profile_excerpt_and_semeval_boundary():
    token_scores = pd.DataFrame(
        [
            {"text_id": "t", "model": model, "token_position": position, "mg_score": score}
            for model, values in [("logistic_regression", [0.1, 0.3, 0.7]), ("random_forest", [0.2, 0.4, 0.8])]
            for position, score in enumerate(values)
        ]
    )
    excerpt = pd.DataFrame(
        {
            "token_position": [0, 1, 2],
            "token_surface": ["Hello", "world", "."],
            "logistic_regression": [0.1, 0.3, 0.7],
            "random_forest": [0.2, 0.4, 0.8],
            "is_gold_boundary": [False, True, False],
        }
    )

    figure = plot_passage_scores(
        {"dataset": "semeval", "text_id": "t", "source_llm": None},
        token_scores,
        excerpt,
        title="SemEval example",
    )

    assert len(figure.axes) >= 2
    assert len(figure.axes[0].lines) == 3
    assert any(line.get_linestyle() == "--" for line in figure.axes[0].lines)
    assert figure.axes[0].get_ylim() == (0.0, 1.0)
    assert any(line.get_linestyle() == "--" for line in figure.axes[1].lines)
    assert figure.axes[1].get_legend() is None


def test_plot_passage_scores_does_not_add_boundary_for_name_that_llm():
    token_scores = pd.DataFrame(
        [
            {"model": model, "token_position": position, "mg_score": 0.5}
            for model in ("logistic_regression", "random_forest")
            for position in range(2)
        ]
    )
    excerpt = pd.DataFrame(
        {
            "token_position": [0, 1],
            "token_surface": ["text", "."],
            "logistic_regression": [0.5, 0.5],
            "random_forest": [0.5, 0.5],
        }
    )

    figure = plot_passage_scores(
        {"dataset": "name_that_llm_train", "text_id": "t", "source_llm": "model-x"},
        token_scores,
        excerpt,
        title="Name That LLM example",
    )

    assert not any(line.get_linestyle() == "--" for line in figure.axes[1].lines)


def test_plot_passage_scores_labels_each_classifier_once_for_wrapped_excerpt():
    token_scores = pd.DataFrame(
        [
            {"model": model, "token_position": position, "mg_score": 0.5}
            for model in ("logistic_regression", "random_forest")
            for position in range(4)
        ]
    )
    excerpt = pd.DataFrame(
        {
            "token_position": range(4),
            "token_surface": ["a" * 40] * 4,
            "logistic_regression": [0.5] * 4,
            "random_forest": [0.5] * 4,
        }
    )

    figure = plot_passage_scores(
        {"dataset": "name_that_llm_train", "text_id": "t", "source_llm": "model-x"},
        token_scores,
        excerpt,
        title="Wrapped example",
    )

    labels = [text.get_text() for text in figure.axes[1].texts]
    assert labels.count("Logistic Regression") == 1
    assert labels.count("Random Forest") == 1


def test_gold_boundary_markers_stay_with_the_wrapped_token_line():
    token_scores = pd.DataFrame(
        [
            {"model": model, "token_position": position, "mg_score": 0.5}
            for model in ("logistic_regression", "random_forest")
            for position in range(4)
        ]
    )
    excerpt = pd.DataFrame(
        {
            "token_position": range(4),
            "token_surface": ["a" * 40] * 4,
            "logistic_regression": [0.5] * 4,
            "random_forest": [0.5] * 4,
            "is_gold_boundary": [False, False, True, False],
        }
    )

    figure = plot_passage_scores(
        {"dataset": "semeval", "text_id": "t", "source_llm": None},
        token_scores,
        excerpt,
        title="Wrapped SemEval example",
    )

    boundary_lines = [
        line for line in figure.axes[1].lines if line.get_linestyle() == "--"
    ]
    assert len(boundary_lines) == 2
    assert sorted(round(float(line.get_ydata().mean()), 2) for line in boundary_lines) == [1.0, 4.0]


def test_save_visualization_outputs_writes_png_svg_and_csv_without_overwrite(tmp_path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    metadata = pd.DataFrame([{"dataset": "semeval", "text_id": "1", "selection_group": "manual"}])
    output = tmp_path / "run" / "reconstructed_token_mg_scores"

    paths = save_visualization_outputs({"semeval__1": figure}, metadata, output)

    assert {path.suffix for path in paths} == {".png", ".svg", ".csv"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    with pytest.raises(FileExistsError, match="output directory already exists"):
        save_visualization_outputs({"semeval__1": figure}, metadata, output)
    plt.close(figure)
