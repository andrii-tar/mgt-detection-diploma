from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from notebooks.local_evidence_training_utils import train_local_evidence_models


FEATURES = ("feature_a", "feature_b")
METADATA = {
    "llm": "llm",
    "training_regime": "Mixed",
    "cohort_seed": 42,
    "max_per_split": 500,
}


def _windows(dataset: str) -> pd.DataFrame:
    rows = []
    for split, ids in (("HG", (0, 1, 2)), ("MG", (3, 4, 5))):
        label = int(split == "MG")
        for text_id in ids:
            for window_length in (2,):
                value = float(label * 3 + text_id / 10 + window_length / 100)
                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "text_id": text_id,
                    "label": label,
                    "window_length": window_length,
                    "start_token": 0,
                    "stop_token": window_length,
                    "feature_a": value,
                    "feature_b": -value,
                })
    return pd.DataFrame(rows)


def _cohort(*ids: tuple[str, int]) -> pd.DataFrame:
    return pd.DataFrame(ids, columns=["split", "text_id"])


def _trained(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "notebooks.local_evidence_training_utils.load_local_evidence_cohort",
        lambda *args, **kwargs: _windows(args[1]),
    )
    return train_local_evidence_models(
        processed_root="data",
        datasets=["source"],
        llm_name="llm",
        train_text_ids={"source": _cohort(("HG", 0), ("MG", 3))},
        validation_text_ids={"source": _cohort(("HG", 1), ("MG", 4))},
        classifier_names=("logistic_regression",),
        window_lengths=(2,),
        feature_columns=FEATURES,
    )


def _evaluation_results() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.DataFrame([{"macro_f1": 1.0}]),
        "predictions": pd.DataFrame([{"text_id": "source::HG::2", "prediction": 0}]),
        "coverage": pd.DataFrame([{"texts": 1, "window_rows": 1}]),
    }


def test_trusted_artifact_round_trip_restores_bundle_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = importlib.import_module("notebooks.local_evidence_artifact_utils")
    trained = _trained(monkeypatch)
    results = _evaluation_results()

    saved = artifacts.save_local_evidence_artifact(
        tmp_path,
        "run-1",
        trained,
        metadata=METADATA,
        evaluation_results=results,
    )
    loaded = artifacts.load_local_evidence_artifact(
        tmp_path,
        "run-1",
        expected_metadata=METADATA,
        trust_artifact=True,
    )

    assert loaded.path == saved
    assert loaded.model.llm_name == "llm"
    assert loaded.model.training_datasets == ("source",)
    assert set(loaded.model.classifiers) == {("logistic_regression", 2)}
    pdt.assert_frame_equal(loaded.model.decision_rules, trained.model.decision_rules)
    pdt.assert_frame_equal(loaded.training_metrics, trained.metrics)
    pdt.assert_frame_equal(loaded.training_coverage, trained.coverage)
    pdt.assert_frame_equal(loaded.evaluation_results["metrics"], results["metrics"])


def test_artifact_loading_requires_explicit_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = importlib.import_module("notebooks.local_evidence_artifact_utils")
    trained = _trained(monkeypatch)
    artifacts.save_local_evidence_artifact(
        tmp_path,
        "run-1",
        trained,
        metadata=METADATA,
        evaluation_results=_evaluation_results(),
    )

    models_utils = importlib.import_module("notebooks.local_evidence_models_utils")
    monkeypatch.setattr(models_utils.pickle, "load", lambda _: pytest.fail("artifact must not be unpickled"))

    with pytest.raises(PermissionError, match="trust_artifact=True"):
        artifacts.load_local_evidence_artifact(
            tmp_path,
            "run-1",
            expected_metadata=METADATA,
        )


def test_artifact_rejects_metadata_mismatch_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = importlib.import_module("notebooks.local_evidence_artifact_utils")
    trained = _trained(monkeypatch)
    artifacts.save_local_evidence_artifact(
        tmp_path,
        "run-1",
        trained,
        metadata=METADATA,
        evaluation_results=_evaluation_results(),
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        artifacts.load_local_evidence_artifact(
            tmp_path,
            "run-1",
            expected_metadata={**METADATA, "max_per_split": 100},
            trust_artifact=True,
        )


def test_artifact_rejects_training_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = importlib.import_module("notebooks.local_evidence_artifact_utils")
    trained = _trained(monkeypatch)
    path = artifacts.save_local_evidence_artifact(
        tmp_path,
        "run-1",
        trained,
        metadata=METADATA,
        evaluation_results=_evaluation_results(),
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_input_fingerprint"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="training fingerprint mismatch"):
        artifacts.load_local_evidence_artifact(
            tmp_path,
            "run-1",
            expected_metadata=METADATA,
            trust_artifact=True,
        )
