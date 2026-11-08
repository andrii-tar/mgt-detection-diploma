from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from notebooks.utils.semeval_random_forest_token_visualization_utils import (
    MODEL_FAMILY,
    WINDOW_LENGTHS,
    build_gold_boundaries,
    load_selected_annotations,
    load_selected_token_labels,
    load_selected_windows,
    load_visualization_models,
    scan_annotation_ids,
    scan_stage1_ids,
    select_eligible_text_ids,
    score_visualization_windows,
    write_visualization_export,
)


class FakeEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.75, 0.25]]), (len(frame), 1))


@pytest.fixture
def fake_models() -> dict[int, SimpleNamespace]:
    return {
        window_length: SimpleNamespace(
            feature_columns=("feature_a",),
            estimator=FakeEstimator(),
        )
        for window_length in WINDOW_LENGTHS
    }


@pytest.fixture
def fake_windows() -> dict[int, pd.DataFrame]:
    return {
        window_length: pd.DataFrame(
            {
                "split": ["MG"],
                "text_id": [7],
                "label": [1],
                "start_token": [0],
                "stop_token": [window_length],
                "window_configuration": [str(window_length)],
                "feature_a": [3.0],
            }
        )
        for window_length in WINDOW_LENGTHS
    }


def test_load_visualization_models_requires_all_fixed_window_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="window_length=32"):
        load_visualization_models(tmp_path, run_id="run")


def test_score_visualization_windows_uses_probability_for_mg_class(
    fake_models: dict[int, SimpleNamespace], fake_windows: dict[int, pd.DataFrame]
) -> None:
    scored = score_visualization_windows(fake_models, fake_windows)

    assert scored.loc[0, "mg_score"] == pytest.approx(0.75)
    assert scored.loc[0, "model"] == MODEL_FAMILY
    assert set(scored["window_length"]) == set(WINDOW_LENGTHS)
    assert list(scored.columns) == [
        "split",
        "text_id",
        "label",
        "start_token",
        "stop_token",
        "model",
        "window_length",
        "mg_score",
    ]


def test_score_visualization_windows_rejects_missing_mg_class(
    fake_models: dict[int, SimpleNamespace], fake_windows: dict[int, pd.DataFrame]
) -> None:
    fake_models[32].estimator.classes_ = np.array([0])

    with pytest.raises(ValueError, match="MG class"):
        score_visualization_windows(fake_models, fake_windows)


def test_score_visualization_windows_rejects_malformed_token_spans(
    fake_models: dict[int, SimpleNamespace], fake_windows: dict[int, pd.DataFrame]
) -> None:
    fake_windows[32].loc[0, "stop_token"] = 0

    with pytest.raises(ValueError, match="invalid token spans"):
        score_visualization_windows(fake_models, fake_windows)


def test_score_visualization_windows_requires_all_fixed_window_lengths(
    fake_models: dict[int, SimpleNamespace],
    fake_windows: dict[int, pd.DataFrame],
) -> None:
    del fake_models[128]
    del fake_windows[128]

    with pytest.raises(ValueError, match="all fixed window lengths"):
        score_visualization_windows(fake_models, fake_windows)


def _prepared_partition(tmp_path: Path, window_length: int, text_ids: list[int]) -> tuple[Path, dict[str, object]]:
    path = tmp_path / str(window_length)
    path.mkdir()
    expected = {
        "scorer_llm": "gemma_3_270m_it",
        "dataset": "semeval",
        "split": "MG",
        "feature_set": "simple_similarity",
        "window_configuration": str(window_length),
        "stride": 16,
        "feature_columns": ["feature_a"],
    }
    (path / "metadata.json").write_text(json.dumps(expected), encoding="utf-8")
    pd.DataFrame(
        {
            "split": ["MG"] * len(text_ids),
            "text_id": text_ids,
            "label": [1] * len(text_ids),
            "start_token": [0] * len(text_ids),
            "stop_token": [window_length] * len(text_ids),
            "window_configuration": [str(window_length)] * len(text_ids),
            "feature_a": [1.0] * len(text_ids),
        }
    ).to_parquet(path / "data.parquet", index=False)
    return path, expected


def test_select_eligible_text_ids_intersects_all_sources_deterministically(tmp_path: Path) -> None:
    paths = {
        32: _prepared_partition(tmp_path, 32, [1, 2, 3, 4])[0],
        64: _prepared_partition(tmp_path, 64, [2, 3, 4, 5])[0],
        128: _prepared_partition(tmp_path, 128, [3, 4, 5, 6])[0],
    }

    first = select_eligible_text_ids(paths, stage1_ids={3, 4, 7}, annotation_ids={3, 4}, count=2, seed=9)
    second = select_eligible_text_ids(paths, stage1_ids={3, 4, 7}, annotation_ids={3, 4}, count=2, seed=9)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["text_id"]) == {3, 4}


def test_load_selected_windows_keeps_only_requested_ids(tmp_path: Path) -> None:
    path, expected = _prepared_partition(tmp_path, 32, [1, 2, 3])

    loaded = load_selected_windows({32: (path, expected)}, {2})

    assert loaded[32]["text_id"].tolist() == [2]


def test_selected_stage1_and_annotation_loaders_keep_requested_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed" / "semeval__gemma" / "MG" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2],
            "tokens": [
                [{"token": "human"}],
                [{"token": "machine"}],
            ],
        }
    ).to_parquet(data_dir / "part.parquet", index=False)
    annotation_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["human", "machine"],
            "boundary_word_index": [0, 0],
        }
    ).to_csv(annotation_path, index=False)

    assert scan_stage1_ids(tmp_path / "processed", "semeval", "gemma", "MG") == {1, 2}
    assert scan_annotation_ids(annotation_path) == {1, 2}
    assert load_selected_token_labels(tmp_path / "processed", "semeval", "gemma", "MG", {2}) == {2: ["machine"]}
    assert load_selected_annotations(annotation_path, {2})["id"].tolist() == [2]


def test_build_gold_boundaries_attaches_token_boundary() -> None:
    annotations = pd.DataFrame(
        {"id": [7], "text": ["hello machine"], "boundary_word_index": [1]}
    )

    boundaries = build_gold_boundaries(annotations, {7: ["hello ", "machine"]})

    assert boundaries.to_dict("records") == [
        {"text_id": 7, "gold_token_boundary": 1, "token_count": 2}
    ]


def test_write_visualization_export_writes_compact_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "visualization"
    selected = pd.DataFrame({"text_id": [7]})
    scored = pd.DataFrame(
        {
            "text_id": [7],
            "model": [MODEL_FAMILY],
            "window_length": [32],
            "start_token": [0],
            "stop_token": [2],
            "mg_score": [0.75],
        }
    )
    boundaries = pd.DataFrame(
        {"text_id": [7], "gold_token_boundary": [1], "token_count": [2]}
    )

    written = write_visualization_export(
        output,
        selected_texts=selected,
        scored_windows=scored,
        gold_boundaries=boundaries,
        metadata={"selection_seed": 42},
    )

    assert written == output
    assert (output / "selected_texts.csv").is_file()
    assert (output / "scored_windows.parquet").is_file()
    assert (output / "gold_boundaries.csv").is_file()
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8")) == {
        "selection_seed": 42
    }


def test_write_visualization_export_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "visualization"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_visualization_export(
            output,
            selected_texts=pd.DataFrame({"text_id": [7]}),
            scored_windows=pd.DataFrame({"text_id": [7]}),
            gold_boundaries=pd.DataFrame({"text_id": [7]}),
            metadata={},
        )
