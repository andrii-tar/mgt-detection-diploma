from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.extract_data.prepare_semeval2024_task8_subtask_c import prepare_semeval2024_task8_subtask_c


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_prepares_mg_csvs_and_preserves_boundary_metadata(tmp_path: Path) -> None:
    input_root = tmp_path / "subtaskC"
    output_root = tmp_path / "output"
    input_root.mkdir()
    _write_jsonl(
        input_root / "subtaskC_train.jsonl",
        [
            {"id": "source-1", "text": "human words machine words", "label": 2},
            {"id": "source-2", "text": "machine only", "label": 0},
        ],
    )
    _write_jsonl(input_root / "subtaskC_dev.jsonl", [{"id": "source-3", "text": "one machine", "label": 1}])

    prepare_semeval2024_task8_subtask_c(input_root, output_root)

    train = pd.read_csv(output_root / "SemEval2024_task8_C_train_MG.csv", keep_default_na=False)
    dev = pd.read_csv(output_root / "SemEval2024_task8_C_dev_MG.csv", keep_default_na=False)
    assert list(train.columns) == [
        "id",
        "text",
        "boundary_word_index",
        "source_id",
        "split",
        "origin",
        "source_dataset",
    ]
    assert train.to_dict("records") == [
        {
            "id": 0,
            "text": "human words machine words",
            "boundary_word_index": 2,
            "source_id": "source-1",
            "split": "train",
            "origin": "MG",
            "source_dataset": "semeval_2024_task8_subtask_c",
        },
        {
            "id": 1,
            "text": "machine only",
            "boundary_word_index": 0,
            "source_id": "source-2",
            "split": "train",
            "origin": "MG",
            "source_dataset": "semeval_2024_task8_subtask_c",
        },
    ]
    assert dev[["id", "source_id", "split", "origin"]].to_dict("records") == [
        {"id": 0, "source_id": "source-3", "split": "dev", "origin": "MG"},
    ]


def test_rejects_boundary_outside_text_word_range(tmp_path: Path) -> None:
    input_root = tmp_path / "subtaskC"
    input_root.mkdir()
    _write_jsonl(input_root / "subtaskC_train.jsonl", [{"id": "source-1", "text": "only text", "label": 2}])
    _write_jsonl(input_root / "subtaskC_dev.jsonl", [])

    with pytest.raises(ValueError, match="boundary"):
        prepare_semeval2024_task8_subtask_c(input_root, tmp_path / "output")
