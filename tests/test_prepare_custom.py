from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.extract_data.prepare_custom import prepare_custom


def test_prepare_custom_splits_origins_and_preserves_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "custom_examples.tsv"
    output_root = tmp_path / "output"
    pd.DataFrame(
        [
            {"LLM_origin": "gpt-4o", "text": "generated one"},
            {"LLM_origin": "human", "text": "human one"},
            {"LLM_origin": "claude", "text": "generated two"},
        ]
    ).to_csv(input_path, sep="\t", index=False)

    prepare_custom(input_path, output_root)

    assert pd.read_csv(output_root / "custom_2025_HG.csv").to_dict("records") == [
        {"id": 0, "text": "human one", "LLM_origin": "human"}
    ]
    assert pd.read_csv(output_root / "custom_2025_MG.csv").to_dict("records") == [
        {"id": 0, "text": "generated one", "LLM_origin": "gpt-4o"},
        {"id": 1, "text": "generated two", "LLM_origin": "claude"},
    ]


def test_prepare_custom_rejects_missing_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.tsv"
    pd.DataFrame([{"LLM_origin": "gpt-4o"}]).to_csv(input_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_custom(input_path, tmp_path / "output")
