from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))

from scripts import process_pipeline


def test_process_split_scores_and_writes_small_chunks(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    split_path = tmp_path / "input.csv"
    pd.DataFrame([{"id": index, "text": f"text {index}"} for index in range(65)]).to_csv(split_path, index=False)

    monkeypatch.setattr(process_pipeline, "DATA_ROOT", data_root)

    scored_chunk_sizes: list[int] = []

    def fake_score_texts_with_model(
        texts: list[str],
        tokenizer: Any,
        model: Any,
        device: Any,
        *,
        top_n: int,
        inference_batch_size: int,
        progress_desc: str | None,
    ) -> list[dict[str, Any]]:
        scored_chunk_sizes.append(len(texts))
        return [{"text": text, "tokens": []} for text in texts]

    monkeypatch.setattr(process_pipeline, "score_texts_with_model", fake_score_texts_with_model)

    process_pipeline._process_split(
        split="HG",
        count=65,
        tokenizer=object(),
        model=object(),
        device=object(),
        dataset="kaggle_train",
        dataset_name="kaggle_train",
        split_path=str(split_path),
        model_name="gemma_2_2b_it",
        model_id="google/gemma-2-2b-it",
        top_n=20,
        inference_batch_size=10,
        write_batch_size=25,
    )

    assert scored_chunk_sizes == [25, 25, 15]

    output_dir = data_root / "processed" / "kaggle_train__gemma_2_2b_it" / "HG" / "data"
    assert sorted(path.name for path in output_dir.glob("*.parquet")) == [
        "batch_00000000_00000024.parquet",
        "batch_00000025_00000049.parquet",
        "batch_00000050_00000064.parquet",
    ]
    assert not list(output_dir.glob("*.tmp"))

    state = json.loads(
        (data_root / "processed" / "kaggle_train__gemma_2_2b_it" / "HG" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["last_id"] == 64
    assert state["processed_count"] == 65
