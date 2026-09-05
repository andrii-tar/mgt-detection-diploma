from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))

from scripts import token_similarity_stage2


def test_process_split_updates_state_after_each_similarity_batch(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_dir = data_root / "processed" / "demo__model" / "HG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": index,
                "dataset": "demo",
                "dataset_name": "demo dataset",
                "model": "model",
                "model_id": "model/id",
                "experiment_name": "demo__model",
                "experiment_params": {"dataset": "demo", "model": "model"},
                "split": "HG",
                "tokens": [],
            }
            for index in range(101)
        ]
    ).to_parquet(data_dir / "batch_00000000_00000100.parquet", index=False)

    monkeypatch.setattr(token_similarity_stage2, "DATA_ROOT", data_root)

    state_updates: list[dict[str, Any]] = []
    original_write_state = token_similarity_stage2._write_state

    def tracking_write_state(*args: Any, **kwargs: Any) -> None:
        original_write_state(*args, **kwargs)
        state_updates.append(
            json.loads(
                (data_root / "processed" / "demo__model" / "HG" / "similarity_state.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    monkeypatch.setattr(token_similarity_stage2, "_write_state", tracking_write_state)

    token_similarity_stage2._process_split(
        split="HG",
        count=101,
        normalized_embeddings=object(),
        dataset="demo",
        dataset_name="demo dataset",
        model_name="model",
        model_id="model/id",
    )

    assert [update["processed_count"] for update in state_updates] == [100, 101]
    assert [update["last_id"] for update in state_updates] == [99, 100]


def test_run_stage_2_resolves_runtime_data_root(tmp_path, monkeypatch):
    class DummyWeight:
        def detach(self):
            return self

    class DummyModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=DummyWeight())

    monkeypatch.setattr(token_similarity_stage2, "DATA_ROOT", Path("."))
    monkeypatch.setattr(token_similarity_stage2, "resolve_data_root", lambda: tmp_path)
    monkeypatch.setattr(token_similarity_stage2, "resolve_dataset_splits", lambda dataset: {"HG": tmp_path / "input.csv"})
    monkeypatch.setattr(token_similarity_stage2, "resolve_model_name", lambda model: "canonical_model")
    monkeypatch.setattr(token_similarity_stage2, "load_local_model", lambda model_id: (None, DummyModel(), None))
    monkeypatch.setattr(token_similarity_stage2.F, "normalize", lambda weights, dim: weights)
    stage2_args: list[object] = []
    monkeypatch.setattr(token_similarity_stage2, "_process_split", lambda *args: stage2_args.extend(args))

    token_similarity_stage2.run_stage_2(
        dataset="demo",
        model_name="model",
        model_id="model/id",
        count=1,
        splits=["HG"],
    )

    assert token_similarity_stage2.DATA_ROOT == tmp_path
    assert stage2_args[5] == "canonical_model"
