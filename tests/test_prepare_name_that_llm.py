from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.extract_data.prepare_name_that_llm import prepare_name_that_llm


def test_prepare_name_that_llm_preserves_ids_and_adds_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "output" / "name_that_llm_train_MG.csv"
    pd.DataFrame(
        [
            {
                "id": "0018b3765972",
                "prompt_id": "p081",
                "text": "first text",
                "model": "qwen3.8-max",
            },
            {
                "id": "0041ac86ee93",
                "prompt_id": "p073",
                "text": "second text",
                "model": "claude-sonnet-5",
            },
        ]
    ).to_csv(input_path, index=False)

    prepare_name_that_llm(input_path, output_path)

    result = pd.read_csv(output_path, dtype={"id": str})
    assert list(result.columns) == [
        "id",
        "source_id",
        "text",
        "prompt_id",
        "source_llm",
        "origin",
        "source_dataset",
    ]
    assert result["id"].tolist() == ["0", "1"]
    assert result["source_id"].tolist() == ["0018b3765972", "0041ac86ee93"]
    assert result["prompt_id"].tolist() == ["p081", "p073"]
    assert result["source_llm"].tolist() == ["qwen3.8-max", "claude-sonnet-5"]
    assert set(result["origin"]) == {"MG"}
    assert set(result["source_dataset"]) == {"name_that_llm"}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "a", "prompt_id": "p", "text": "x"}], "model"),
        (
            [
                {"id": "a", "prompt_id": "p", "text": "x", "model": "m"},
                {"id": "a", "prompt_id": "q", "text": "y", "model": "n"},
            ],
            "unique",
        ),
        ([{"id": "a", "prompt_id": "p", "text": "  ", "model": "m"}], "text"),
        ([{"id": "a", "prompt_id": "p", "text": "x", "model": "  "}], "model"),
    ],
)
def test_prepare_name_that_llm_rejects_invalid_input(
    tmp_path: Path, rows: list[dict[str, str]], message: str
) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match=message):
        prepare_name_that_llm(input_path, output_path)


def test_prepare_name_that_llm_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing input file"):
        prepare_name_that_llm(tmp_path / "missing.csv", tmp_path / "output.csv")
