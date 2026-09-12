from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from notebooks.utils.feature_aggregation_utils import (
    FeatureSet,
    ensure_prepared_partition,
    load_prepared_partition,
    prepare_feature_partition,
    save_prepared_partition,
)


FEATURE_SET = FeatureSet(name="d_score_only", source_columns=("d_score_top20",))
LOCAL_FEATURE_SET = FeatureSet(
    name="simple_similarity",
    source_columns=(
        "mean_candidate_similarity__mean",
        "mean_candidate_similarity__std",
        "top20_coverage__mean",
        "top20_coverage__std",
        "observed_logprob__mean",
        "observed_logprob__std",
        "mean_candidate_logprob_delta__mean",
        "mean_candidate_logprob_delta__std",
    ),
)
MIXED_FEATURE_SET = FeatureSet(
    name="extended",
    source_columns=("d_score_top20", *LOCAL_FEATURE_SET.source_columns),
)


def _write_processed_text(
    root: Path,
    *,
    dataset: str,
    scorer: str,
    split: str,
    token_count: int,
    text_id: int = 1,
    label: int | None = None,
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
    row = {"id": text_id, "split": split, "text": "text", "tokens": tokens}
    if label is not None:
        row["label"] = label
    pd.DataFrame([row]).to_parquet(
        path, index=False
    )


def _write_similarity_text(
    root: Path, *, dataset: str, scorer: str, split: str, token_count: int, text_id: int = 1,
) -> None:
    path = root / f"{dataset}__{scorer}" / split / "token_similarities" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "id": text_id,
        "token_similarities": [
            {"top_logprobs": [{"similarity_to_gt": 0.2 + index * 0.1}]}
            for index in range(token_count)
        ],
    }]).to_parquet(path, index=False)


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


def test_preparation_computes_local_evidence_summary_features(tmp_path: Path) -> None:
    _write_processed_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=2)
    _write_similarity_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=2)

    frame = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=LOCAL_FEATURE_SET, window_configuration=2, stride=1,
    )

    assert frame.loc[0, list(LOCAL_FEATURE_SET.source_columns)].notna().all()
    assert frame.loc[0, "observed_logprob__mean"] == pytest.approx(-0.105)


def test_preparation_combines_scalar_and_local_evidence_features(tmp_path: Path) -> None:
    _write_processed_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=2)
    _write_similarity_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=2)

    frame = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=MIXED_FEATURE_SET, window_configuration=2, stride=1,
    )

    assert frame.loc[0, list(MIXED_FEATURE_SET.source_columns)].notna().all()


def test_ensure_reports_missing_stage_two_for_local_features(tmp_path: Path) -> None:
    _write_processed_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=2)

    result = ensure_prepared_partition(
        tmp_path, tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=LOCAL_FEATURE_SET, window_configuration=2, stride=1,
    )

    assert result["status"] == "unavailable"
    assert "token similarity" in str(result["error"]).lower()
    assert not (tmp_path / "experiment-results").exists()


def test_whole_text_emits_zero_length_rows_for_empty_and_none_tokens(tmp_path: Path) -> None:
    path = tmp_path / "source__scorer" / "HG" / "data" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame([
        {"id": 1, "split": "HG", "text": "empty", "tokens": []},
        {"id": 2, "split": "HG", "text": "none", "tokens": None},
    ]).to_parquet(path, index=False)

    frame = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration="whole_text", stride=1,
    )

    assert frame[["text_id", "start_token", "stop_token"]].to_dict("records") == [
        {"text_id": 1, "start_token": 0, "stop_token": 0},
        {"text_id": 2, "start_token": 0, "stop_token": 0},
    ]
    assert frame["d_score_top20"].isna().all()


def test_preparation_preserves_source_labels_and_rejects_inconsistent_parent_labels(tmp_path: Path) -> None:
    _write_processed_text(tmp_path, dataset="source", scorer="scorer", split="HG", token_count=1, label=0)
    frame = prepare_feature_partition(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=1, stride=1,
    )
    assert frame["label"].tolist() == [0]

    path = tmp_path / "source__scorer" / "HG" / "data" / "part-1.parquet"
    pd.DataFrame([{"id": 2, "split": "HG", "text": "text", "label": 1,
                   "tokens": [{"logprob": -0.1, "top_logprobs": [{"logprob": -0.2}]}]}]).to_parquet(
        path, index=False
    )
    with pytest.raises(ValueError, match="label"):
        prepare_feature_partition(
            tmp_path, scorer_llm="scorer", dataset="source", split="HG",
            feature_set=FEATURE_SET, window_configuration=1, stride=1,
        )

    inconsistent = pd.DataFrame([
        {"split": "HG", "text_id": 3, "label": 0, "start_token": 0, "stop_token": 1, "window_configuration": "1", "d_score_top20": 1.0},
        {"split": "HG", "text_id": 3, "label": 1, "start_token": 1, "stop_token": 2, "window_configuration": "1", "d_score_top20": 1.0},
    ])
    with pytest.raises(ValueError, match="inconsistent labels per parent"):
        save_prepared_partition(
            tmp_path, inconsistent, scorer_llm="scorer", dataset="inconsistent", split="HG",
            feature_set=FEATURE_SET, window_configuration=1, stride=1,
        )


def test_prepared_partition_round_trip_requires_matching_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0,
        "start_token": 0, "stop_token": 1, "window_configuration": "whole_text",
        "d_score_top20": 1.0,
    }])
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


def test_load_validates_required_metadata_and_row_count_without_expected_mapping(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0,
        "start_token": 0, "stop_token": 2, "window_configuration": "2",
        "d_score_top20": 1.0,
    }])
    path = save_prepared_partition(
        tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=2,
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["schema_version"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema version"):
        load_prepared_partition(path, expected={})

    metadata["schema_version"] = 1
    metadata["row_count"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="row count"):
        load_prepared_partition(path, expected={})


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
        {"split": "HG", "text_id": 1, "label": 0, "start_token": 0, "stop_token": 2, "window_configuration": "2", "a": 1.0, "b": 2.0},
        {"split": "HG", "text_id": 1, "label": 0, "start_token": 0, "stop_token": 2, "window_configuration": "2", "a": 1.0, "b": 2.0},
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


def test_ensure_rejects_existing_partition_with_changed_stride_or_schema(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0,
        "start_token": 0, "stop_token": 2, "window_configuration": "2",
        "d_score_top20": 1.0,
    }])
    path = save_prepared_partition(
        tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=2,
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["stride"] = 1
    metadata_path.write_text(json.dumps(metadata))
    result = ensure_prepared_partition(
        tmp_path, tmp_path / "missing", scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=2,
    )
    assert result["status"] == "unavailable"
    assert "stride" in result["error"]

    metadata["stride"] = 2
    metadata["schema_version"] = 999
    metadata_path.write_text(json.dumps(metadata))
    result = ensure_prepared_partition(
        tmp_path, tmp_path / "missing", scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=2,
    )
    assert result["status"] == "unavailable"
    assert "schema version" in result["error"]


def test_save_cleans_temporary_partition_after_persistence_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0,
        "start_token": 0, "stop_token": 2, "window_configuration": "2",
        "d_score_top20": 1.0,
    }])
    original = pd.DataFrame.to_parquet

    def fail_after_write(self: pd.DataFrame, path: object, *args: object, **kwargs: object) -> None:
        original(self, path, *args, **kwargs)
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_write)
    with pytest.raises(RuntimeError, match="write failed"):
        save_prepared_partition(
            tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
            feature_set=FEATURE_SET, window_configuration=2, stride=1,
        )
    assert not list((tmp_path / "experiment-results").rglob("*.tmp"))


def test_save_rejects_rows_from_a_different_requested_split(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "split": "MG", "text_id": 1, "label": 1,
        "start_token": 0, "stop_token": 2, "window_configuration": "2",
        "d_score_top20": 1.0,
    }])

    with pytest.raises(ValueError, match="requested split"):
        save_prepared_partition(
            tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
            feature_set=FEATURE_SET, window_configuration=2, stride=1,
        )


def test_save_requires_all_partition_identity_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"split": "HG", "text_id": 1, "label": 0, "d_score_top20": 1.0}])

    with pytest.raises(ValueError, match="identity"):
        save_prepared_partition(
            tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
            feature_set=FEATURE_SET, window_configuration="whole_text", stride=1,
        )


def test_load_rejects_malformed_partition_identity_and_feature_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "split": "HG", "text_id": 1, "label": 0,
        "start_token": 0, "stop_token": 2, "window_configuration": "2",
        "d_score_top20": 1.0,
    }])
    path = save_prepared_partition(
        tmp_path, frame, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FEATURE_SET, window_configuration=2, stride=1,
    )

    data = pd.read_parquet(path / "data.parquet")
    data.loc[0, "split"] = "MG"
    data.to_parquet(path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="requested split"):
        load_prepared_partition(path, expected={"split": "HG"})

    data.loc[0, "split"] = "HG"
    data = data.drop(columns=["window_configuration"])
    data.to_parquet(path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="identity"):
        load_prepared_partition(path, expected={})

    data["window_configuration"] = "2"
    data.loc[0, "window_configuration"] = "whole_text"
    data.to_parquet(path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="window configuration"):
        load_prepared_partition(path, expected={})

    data.loc[0, "window_configuration"] = "2"
    data.to_parquet(path / "data.parquet", index=False)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_columns"] = ["d_score_top20", "d_score_top20"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature columns"):
        load_prepared_partition(path, expected={})
