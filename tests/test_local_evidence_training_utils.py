from __future__ import annotations

import pandas as pd
import pytest

from notebooks.local_evidence_training_utils import (
    LocalEvidenceModelBundle,
    train_local_evidence_models,
)
from notebooks.local_evidence_utils import load_local_evidence_cohort


FEATURES = ("feature_a", "feature_b")


def _windows(dataset: str) -> pd.DataFrame:
    rows = []
    for split, ids in (("HG", (0, 1, 2)), ("MG", (3, 4, 5))):
        label = int(split == "MG")
        for text_id in ids:
            for window_length in (2, 4):
                value = float(label * 3 + text_id / 10 + window_length / 100)
                rows.append({
                    "dataset": dataset, "split": split, "text_id": text_id,
                    "label": label, "window_length": window_length,
                    "start_token": 0, "stop_token": window_length,
                    "feature_a": value, "feature_b": -value,
                })
    return pd.DataFrame(rows)


def _cohort(*ids: tuple[str, int]) -> pd.DataFrame:
    return pd.DataFrame(ids, columns=["split", "text_id"])


def test_training_returns_bundle_with_fitted_models_and_validation_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "notebooks.local_evidence_training_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: _windows(args[1]),
    )

    result = train_local_evidence_models(
        processed_root="data",
        datasets=["source"],
        llm_name="llm",
        train_text_ids={"source": _cohort(("HG", 0), ("MG", 3))},
        validation_text_ids={"source": _cohort(("HG", 1), ("MG", 4))},
        classifier_names=("logistic_regression",),
        window_lengths=(2,),
        feature_columns=FEATURES,
    )

    assert isinstance(result.model, LocalEvidenceModelBundle)
    assert ("logistic_regression", 2) in result.model.classifiers
    assert set(result.model.decision_rules["aggregation"]) == {"maximum_window", "evidence_fraction"}
    assert {"partition", "aggregation", "macro_f1"} <= set(result.metrics.columns)


def test_training_rejects_overlapping_train_and_validation_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "notebooks.local_evidence_training_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: _windows(args[1]),
    )

    with pytest.raises(ValueError, match="overlap"):
        train_local_evidence_models(
            processed_root="data",
            datasets=["source"],
            llm_name="llm",
            train_text_ids={"source": _cohort(("HG", 0), ("MG", 3))},
            validation_text_ids={"source": _cohort(("HG", 0), ("MG", 4))},
            classifier_names=("logistic_regression",),
            window_lengths=(2,),
            feature_columns=FEATURES,
        )


def test_cohort_loader_filters_complete_cache_and_rejects_missing_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return _windows(args[1])

    monkeypatch.setattr("notebooks.local_evidence_utils.load_or_build_local_evidence_windows", load)
    selected = load_local_evidence_cohort(
        "data", "source", "llm", _cohort(("HG", 0), ("MG", 3)),
        window_lengths=(2,), stride=1, top_k=5,
    )

    assert set(selected["text_id"]) == {0, 3}
    assert calls[0][1]["splits"] == ("HG", "MG")
    with pytest.raises(KeyError, match="unavailable"):
        load_local_evidence_cohort("data", "source", "llm", _cohort(("HG", 99)))


def test_cohort_loader_forwards_requested_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def load(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return _windows("source")

    monkeypatch.setattr("notebooks.local_evidence_utils.load_or_build_local_evidence_windows", load)

    load_local_evidence_cohort("data", "source", "llm", _cohort(("HG", 0), ("MG", 3)))

    assert calls[0]["text_ids"].to_dict("records") == [
        {"split": "HG", "text_id": 0},
        {"split": "MG", "text_id": 3},
    ]
