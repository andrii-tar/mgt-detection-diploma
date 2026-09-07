from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace
from typing import Mapping

from notebooks.local_evidence_evaluation_utils import (
    LocalEvidenceDatasetEvaluationResult,
    LocalEvidenceSemEvalEvaluationResult,
    combine_local_evidence_dataset_results,
    evaluate_local_evidence_datasets,
    evaluate_local_evidence_validation_lengths,
    evaluate_local_evidence_semeval,
)
from notebooks.local_evidence_training_utils import train_local_evidence_models


FEATURES = ("feature_a", "feature_b")


def _semeval_model() -> SimpleNamespace:
    return SimpleNamespace(
        llm_name="llm",
        classifiers={("classifier", 2): object()},
        feature_columns=FEATURES,
        window_lengths=(2,),
        stride=1,
        top_k=1,
    )


def _windows(dataset: str) -> pd.DataFrame:
    rows = []
    for split, ids in (("HG", (0, 1, 2)), ("MG", (3, 4, 5))):
        label = int(split == "MG")
        for text_id in ids:
            value = float(label * 3 + text_id / 10)
            rows.append({
                "dataset": dataset, "split": split, "text_id": text_id,
                "label": label, "window_length": 2, "start_token": 0,
                "stop_token": 2, "feature_a": value, "feature_b": -value,
            })
    return pd.DataFrame(rows)


def _cohort(*ids: tuple[str, int]) -> pd.DataFrame:
    return pd.DataFrame(ids, columns=["split", "text_id"])


def test_evaluation_uses_only_requested_test_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "notebooks.local_evidence_training_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: _windows(args[1]),
    )
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: _windows(args[1]),
    )
    trained = train_local_evidence_models(
        processed_root="data", datasets=["source"], llm_name="llm",
        train_text_ids={"source": _cohort(("HG", 0), ("MG", 3))},
        validation_text_ids={"source": _cohort(("HG", 1), ("MG", 4))},
        classifier_names=("logistic_regression",), window_lengths=(2,), feature_columns=FEATURES,
    )

    result = evaluate_local_evidence_datasets(
        trained.model, processed_root="data", datasets=["target"],
        text_ids={"target": _cohort(("HG", 2), ("MG", 5))},
    )

    assert set(result.predictions["text_id"]) == {"target::HG::2", "target::MG::5"}
    assert result.metrics["evaluation_dataset"].tolist() == ["target", "target"]


def test_short_validation_evaluation_adds_length_metadata_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        llm_name="llm", window_lengths=(2,), stride=1, top_k=1,
        classifiers={}, decision_rules=pd.DataFrame(), feature_columns=FEATURES,
    )
    validation_ids = {"source": _cohort(("HG", 1))}
    windows = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0, "window_length": 2,
        "start_token": 0, "stop_token": 2, "feature_a": 0.1, "feature_b": 0.2,
    }])
    base_result = LocalEvidenceDatasetEvaluationResult(
        pd.DataFrame([{"macro_f1": 1.0}]),
        pd.DataFrame([{"prediction": 0}]),
        windows,
        pd.DataFrame([{"texts": 1, "window_rows": 1}]),
    )
    calls: list[Mapping[str, pd.DataFrame] | None] = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs.get("windows_override"))
        return base_result

    monkeypatch.setattr("notebooks.local_evidence_evaluation_utils.evaluate_local_evidence_datasets", evaluate)
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: windows,
    )

    result = evaluate_local_evidence_validation_lengths(
        model, "data", ["source"], validation_ids,
        {"source": [{"id": 1, "split": "HG", "text": "one two three", "tokens": [
            {"token": "one"}, {"token": " two"}, {"token": " three"},
        ]}]},
        (2,),
    )

    assert len(calls) == 2
    assert set(result.metrics["validation_variant"]) == {"original", "first_N_words"}
    assert set(result.metrics["classifier_family"]) == {"local_evidence_agg_mg_score"}


def test_combines_local_evidence_results_for_mixed_evaluation() -> None:
    rule = {
        "model": "classifier",
        "window_length": 2,
        "aggregation": "maximum_window",
        "window_threshold": float("nan"),
        "document_threshold": 0.5,
        "score_direction": "higher_is_mg",
    }
    results = []
    for dataset, text_id, label, mg_score in (
        ("source", "source::HG::1", 0, 0.1),
        ("target", "target::MG::2", 1, 0.9),
    ):
        prediction = pd.DataFrame([{
            **rule, "evaluation_dataset": dataset,
            "split": "HG" if label == 0 else "MG", "text_id": text_id,
            "label": label, "mg_score": mg_score,
            "prediction": int(mg_score >= 0.5),
        }])
        scored = pd.DataFrame([{
            **rule, "evaluation_dataset": dataset,
            "split": prediction.loc[0, "split"], "text_id": text_id,
            "label": label, "start_token": 0, "stop_token": 2,
            "mg_score": mg_score,
        }])
        coverage = pd.DataFrame([{
            "evaluation_dataset": dataset, "texts": 1, "window_rows": 1,
        }])
        results.append(LocalEvidenceDatasetEvaluationResult(pd.DataFrame(), prediction, scored, coverage))

    combined = combine_local_evidence_dataset_results(results, evaluation_regime="Mixed")

    assert set(combined.predictions["text_id"]) == {"source::HG::1", "target::MG::2"}
    assert len(combined.metrics) == 1
    assert combined.metrics["evaluation_regime"].tolist() == ["Mixed"]
    assert combined.predictions["evaluation_regime"].tolist() == ["Mixed", "Mixed"]
    assert combined.coverage["evaluation_regime"].tolist() == ["Mixed", "Mixed"]


def test_semeval_evaluation_reuses_scored_windows_for_regions_and_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = pd.DataFrame([
        {"id": 7, "text": "human text machine text", "boundary_word_index": 2},
    ])
    score_calls: list[pd.DataFrame] = []
    boundary_options: dict[str, int] = {}

    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_token_labels",
        lambda *args, **kwargs: {7: ["human", " text", " machine", " text"]},
    )

    def load_cohort(*args, **kwargs):
        requested = args[3]
        assert requested["text_id"].tolist() == [7]
        return pd.DataFrame([{
            "split": "MG", "text_id": 7, "label": 1, "window_length": 2,
            "start_token": start, "stop_token": start + 2,
            "feature_a": 0.9, "feature_b": 0.1,
        } for start in (0, 2)])

    monkeypatch.setattr("notebooks.local_evidence_evaluation_utils.load_local_evidence_cohort", load_cohort)

    def score(fitted, windows, feature_columns):
        score_calls.append(windows.copy())
        return windows.assign(model="classifier", mg_score=0.9)

    monkeypatch.setattr("notebooks.local_evidence_evaluation_utils.score_local_evidence_classifier", score)
    def predict_boundaries(scored, counts, **kwargs):
        boundary_options["sustained_length"] = kwargs["sustained_length"]
        return pd.DataFrame([{
            "model": "classifier", "window_length": 2, "text_id": 7,
            "predicted_token_boundary": 2, "token_count": 4, "no_crossing": False,
        }])
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.predict_boundary_positions",
        predict_boundaries,
    )

    result = evaluate_local_evidence_semeval(
        _semeval_model(), "data", annotations, smoothing_window=1, sustained_length=4,
    )

    assert isinstance(result, LocalEvidenceSemEvalEvaluationResult)
    assert len(score_calls) == 1
    assert boundary_options["sustained_length"] == 4
    assert set(result.summary["model"]) == {"classifier"}
    assert set(result.boundary_summary["model"]) == {"classifier"}
    assert result.boundary_predictions["gold_token_boundary"].tolist() == [2]


def test_semeval_omits_unavailable_and_misaligned_texts_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = pd.DataFrame([
        {"id": 7, "text": "human text machine text", "boundary_word_index": 2},
        {"id": 8, "text": "not represented", "boundary_word_index": 1},
        {"id": 9, "text": "wrong text machine text", "boundary_word_index": 2},
    ])
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_token_labels",
        lambda *args, **kwargs: {
            7: ["human", " text", " machine", " text"],
            9: ["different", " labels", " only", " here"],
        },
    )
    requested_ids: list[int] = []

    def load_cohort(*args, **kwargs):
        requested_ids.extend(args[3]["text_id"].tolist())
        return pd.DataFrame([
            {
                "split": "MG", "text_id": text_id, "label": 1, "window_length": 2,
                "start_token": start, "stop_token": start + 2,
                "feature_a": 0.9, "feature_b": 0.1,
            }
            for text_id in (7, 9)
            for start in (0, 2)
        ])

    monkeypatch.setattr("notebooks.local_evidence_evaluation_utils.load_local_evidence_cohort", load_cohort)
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.score_local_evidence_classifier",
        lambda fitted, windows, feature_columns: windows.assign(model="classifier", mg_score=0.9),
    )

    result = evaluate_local_evidence_semeval(_semeval_model(), "data", annotations, smoothing_window=1)

    assert requested_ids == [7, 9]
    assert result.boundary_predictions["text_id"].tolist() == [7]
    assert result.boundary_exclusions.to_dict("records") == [
        {"text_id": 8, "reason": "missing processed token labels"},
        {"text_id": 9, "reason": "token labels do not reconstruct the annotated text for text id 9"},
    ]


def test_semeval_propagates_unrelated_boundary_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = pd.DataFrame([
        {"id": 7, "text": "human text machine text", "boundary_word_index": 2},
    ])
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_token_labels",
        lambda *args, **kwargs: {7: ["human", " text", " machine", " text"]},
    )
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: pd.DataFrame([
            {
                "split": "MG", "text_id": 7, "label": 1, "window_length": 2,
                "start_token": start, "stop_token": start + 2,
                "feature_a": 0.9, "feature_b": 0.1,
            }
            for start in (0, 2)
        ]),
    )
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.score_local_evidence_classifier",
        lambda fitted, windows, feature_columns: windows.assign(model="classifier", mg_score=0.9),
    )
    monkeypatch.setattr(
        "notebooks.local_evidence_evaluation_utils.add_gold_boundaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing prediction columns: boom")),
    )

    with pytest.raises(ValueError, match="missing prediction columns: boom"):
        evaluate_local_evidence_semeval(_semeval_model(), "data", annotations, smoothing_window=1)
