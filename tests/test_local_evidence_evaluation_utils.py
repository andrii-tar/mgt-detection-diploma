from __future__ import annotations

import pandas as pd
import pytest

from notebooks.local_evidence_evaluation_utils import evaluate_local_evidence_datasets
from notebooks.local_evidence_training_utils import train_local_evidence_models


FEATURES = ("feature_a", "feature_b")


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
