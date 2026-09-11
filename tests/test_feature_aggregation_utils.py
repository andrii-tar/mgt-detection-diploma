from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from notebooks.feature_aggregation_utils import (
    FeatureSet,
    ensure_prepared_partition,
    load_prepared_partition,
    prepare_feature_partition,
    save_prepared_partition,
)


FEATURE_SET = FeatureSet(name="d_score_only", source_columns=("d_score_top20",))


def _write_processed_text(
    root: Path,
    *,
    dataset: str,
    scorer: str,
    split: str,
    token_count: int,
) -> None:
    tokens = [
        {
            "token": f" token-{index}",
            "logprob": -0.1 - index * 0.01,
            "top_logprobs": [{"logprob": -0.2}, {"logprob": -0.3}],
        }
        for index in range(token_count)
    ]
    path = root / f"{dataset}__{scorer}" / split / "data" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"id": 1, "split": split, "text": "text", "tokens": tokens}]).to_parquet(
        path, index=False
    )


def test_preparation_computes_d_score_over_each_requested_token_window(tmp_path: Path) -> None:
    _write_processed_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=4)

    numeric = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=2,
    )
    whole = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration="whole_text", stride=2,
    )

    assert numeric[["start_token", "stop_token"]].to_dict("records") == [
        {"start_token": 0, "stop_token": 2}, {"start_token": 2, "stop_token": 4},
    ]
    assert numeric["d_score_top20"].notna().all()
    assert whole[["start_token", "stop_token", "window_configuration"]].to_dict("records") == [
        {"start_token": 0, "stop_token": 4, "window_configuration": "whole_text"},
    ]


def test_prepared_partition_round_trip_requires_matching_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"split": "HG", "text_id": 1, "label": 0, "d_score_top20": 1.0}])
    path = save_prepared_partition(
        tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration="whole_text", stride=2,
    )

    loaded = load_prepared_partition(
        path, expected={"scorer_llm": "scorer", "dataset": "source", "split": "HG",
                        "feature_set": "d_score_only", "feature_columns": ["d_score_top20"],
                        "window_configuration": "whole_text"},
    )

    pd.testing.assert_frame_equal(loaded, frame)


def test_validation_rejects_invalid_feature_and_window_configuration() -> None:
    with pytest.raises(ValueError, match="feature set"):
        prepare_feature_partition("missing", scorer_llm="s", dataset="d", split="HG",
                                  feature_set=FeatureSet("empty", ()), window_configuration=2, stride=1)
    with pytest.raises(ValueError, match="positive integer"):
        prepare_feature_partition("missing", scorer_llm="s", dataset="d", split="HG",
                                  feature_set=FEATURE_SET, window_configuration=0, stride=1)
    with pytest.raises(ValueError, match="registered"):
        prepare_feature_partition("missing", scorer_llm="s", dataset="d", split="HG",
                                  feature_set=FeatureSet("unknown", ("not_a_metric",)),
                                  window_configuration=2, stride=1)


def test_load_rejects_duplicate_identity_and_changed_feature_order(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        {"split": "HG", "text_id": 1, "label": 0, "start_token": 0, "stop_token": 2, "a": 1.0, "b": 2.0},
        {"split": "HG", "text_id": 1, "label": 0, "start_token": 0, "stop_token": 2, "a": 1.0, "b": 2.0},
    ])
    feature_set = FeatureSet("two", ("a", "b"))
    with pytest.raises(ValueError, match="duplicate"):
        save_prepared_partition(tmp_path, frame, scorer_llm="s", dataset="d", split="HG",
                                feature_set=feature_set, window_configuration=2, stride=1)
    good_path = save_prepared_partition(tmp_path, frame.iloc[[0]], scorer_llm="s", dataset="good", split="HG",
                                        feature_set=feature_set, window_configuration=2, stride=1)
    metadata = json.loads((good_path / "metadata.json").read_text())
    metadata["feature_columns"] = ["b", "a"]
    (good_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature columns"):
        load_prepared_partition(good_path, expected={"feature_columns": ["a", "b"]})


def test_ensure_unavailable_does_not_publish_partition(tmp_path: Path) -> None:
    result = ensure_prepared_partition(
        tmp_path, tmp_path / "missing", scorer_llm="s", dataset="d", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=1,
    )
    assert result["status"] == "unavailable"
    assert result["error"]
    assert not (tmp_path / "experiment-results").exists()
