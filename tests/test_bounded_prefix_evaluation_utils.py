from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from notebooks.utils import bounded_prefix_evaluation_utils
from notebooks.utils.bounded_prefix_evaluation_utils import (
    bounded_prefix_prediction_key,
    bounded_prefix_successful_keys,
    collect_source_parent_identities,
    ensure_bounded_prefix_result_exports,
    iter_bounded_prefix_feature_batches,
    validate_prefix_bounds,
)
from notebooks.utils.feature_aggregation_utils import FeatureSet
from notebooks.utils.streaming_classification_evaluation_utils import prediction_artifact_path


def _write_source(root: Path, *, text_id: int, token_count: int, label: int | None = None) -> None:
    path = root / "source__scorer" / "HG" / "data" / f"part-{text_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = [
        {
            "token": f" token-{index}",
            "logprob": -0.1 - index * 0.01,
            "top_logprobs": [{"logprob": -0.2}, {"logprob": -0.3}],
        }
        for index in range(token_count)
    ]
    row = {"id": text_id, "split": "HG", "text": "text", "tokens": tokens}
    if label is not None:
        row["label"] = label
    pd.DataFrame([row]).to_parquet(path, index=False)


def _write_similarity(root: Path, *, text_id: int, token_count: int) -> None:
    path = root / "source__scorer" / "HG" / "token_similarities" / f"part-{text_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    similarities = [
        {"top_logprobs": [{"similarity_to_gt": 0.2 + index * 0.1}]}
        for index in range(token_count)
    ]
    pd.DataFrame([{"id": text_id, "token_similarities": similarities}]).to_parquet(path, index=False)


def test_prefix_batches_filter_clip_and_emit_regular_windows(tmp_path: Path) -> None:
    for text_id, token_count in ((1, 49), (2, 50), (3, 64), (4, 100)):
        _write_source(tmp_path, text_id=text_id, token_count=token_count)

    batches = list(iter_bounded_prefix_feature_batches(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG",
        feature_set=FeatureSet("d", ("d_score_top20",)), min_tokens=50, max_tokens=80,
        selected_ids={1, 2, 3, 4}, batch_size=1,
    ))

    frame = pd.concat(batches, ignore_index=True)
    assert frame[[
        "text_id", "original_token_count", "evaluated_token_count", "start_token", "stop_token",
    ]].to_dict("records") == [
        {"text_id": 2, "original_token_count": 50, "evaluated_token_count": 50, "start_token": 0, "stop_token": 32},
        {"text_id": 2, "original_token_count": 50, "evaluated_token_count": 50, "start_token": 16, "stop_token": 48},
        {"text_id": 2, "original_token_count": 50, "evaluated_token_count": 50, "start_token": 18, "stop_token": 50},
        {"text_id": 3, "original_token_count": 64, "evaluated_token_count": 64, "start_token": 0, "stop_token": 32},
        {"text_id": 3, "original_token_count": 64, "evaluated_token_count": 64, "start_token": 16, "stop_token": 48},
        {"text_id": 3, "original_token_count": 64, "evaluated_token_count": 64, "start_token": 32, "stop_token": 64},
        {"text_id": 4, "original_token_count": 100, "evaluated_token_count": 80, "start_token": 0, "stop_token": 32},
        {"text_id": 4, "original_token_count": 100, "evaluated_token_count": 80, "start_token": 16, "stop_token": 48},
        {"text_id": 4, "original_token_count": 100, "evaluated_token_count": 80, "start_token": 32, "stop_token": 64},
        {"text_id": 4, "original_token_count": 100, "evaluated_token_count": 80, "start_token": 48, "stop_token": 80},
    ]


def test_prefix_batches_build_scalar_and_similarity_features(tmp_path: Path) -> None:
    _write_source(tmp_path, text_id=1, token_count=64)
    _write_similarity(tmp_path, text_id=1, token_count=64)
    feature_set = FeatureSet(
        "mixed", ("d_score_top20", "observed_logprob__mean", "top20_coverage__std"),
    )

    frame = pd.concat(list(iter_bounded_prefix_feature_batches(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG", feature_set=feature_set,
        min_tokens=32, max_tokens=48, selected_ids={1}, batch_size=1,
    )), ignore_index=True)

    assert frame.loc[0, "original_token_count"] == 64
    assert frame.loc[0, "evaluated_token_count"] == 48
    assert frame.loc[0, list(feature_set.source_columns)].notna().all()


@pytest.mark.parametrize("bounds", [(0, 1), (2, 1), (True, 2), (1, True), (1.5, 2)])
def test_invalid_prefix_bounds_are_rejected(bounds: tuple[object, object]) -> None:
    with pytest.raises(ValueError, match="token bounds"):
        validate_prefix_bounds(*bounds)  # type: ignore[arg-type]


def test_prefix_batches_reject_missing_or_misaligned_similarity_data(tmp_path: Path) -> None:
    _write_source(tmp_path, text_id=1, token_count=2)
    feature_set = FeatureSet("local", ("observed_logprob__mean",))

    with pytest.raises(FileNotFoundError, match="token similarity"):
        list(iter_bounded_prefix_feature_batches(
            tmp_path, scorer_llm="scorer", dataset="source", split="HG", feature_set=feature_set,
            min_tokens=1, max_tokens=2, selected_ids={1}, batch_size=1,
        ))

    _write_similarity(tmp_path, text_id=2, token_count=2)
    with pytest.raises(KeyError, match="No token similarity row"):
        list(iter_bounded_prefix_feature_batches(
            tmp_path, scorer_llm="scorer", dataset="source", split="HG", feature_set=feature_set,
            min_tokens=1, max_tokens=2, selected_ids={1}, batch_size=1,
        ))


def test_parent_identity_collection_uses_source_labels_and_split_cap(tmp_path: Path) -> None:
    for text_id in (3, 1, 2):
        _write_source(tmp_path, text_id=text_id, token_count=2)

    parents = collect_source_parent_identities(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG", max_per_split=2, batch_size=1,
    )

    assert parents.to_dict("records") == [
        {"dataset": "source", "split": "HG", "text_id": 1, "label": 0, "token_count": 2},
        {"dataset": "source", "split": "HG", "text_id": 2, "label": 0, "token_count": 2},
    ]


def test_parent_identity_collection_filters_short_sources_before_split_cap(tmp_path: Path) -> None:
    _write_source(tmp_path, text_id=1, token_count=1)
    _write_source(tmp_path, text_id=2, token_count=2)
    _write_source(tmp_path, text_id=3, token_count=2)

    parents = collect_source_parent_identities(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG", max_per_split=2,
        min_text_tokens=2, batch_size=1,
    )

    assert list(parents["text_id"]) == [2, 3]


def test_parent_identity_collection_handles_an_empty_source(tmp_path: Path) -> None:
    path = tmp_path / "source__scorer" / "HG" / "data" / "part-empty.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"id": pd.Series(dtype="int64")}).to_parquet(path, index=False)

    parents = collect_source_parent_identities(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG", max_per_split=2, batch_size=2,
    )

    assert list(parents.columns) == ["dataset", "split", "text_id", "label", "token_count"]
    assert parents.empty


def test_parent_identity_collection_reports_parquet_batch_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for text_id in (1, 2, 3):
        _write_source(tmp_path, text_id=text_id, token_count=2)
    progress_calls: list[tuple[object, dict[str, object]]] = []
    updates: list[int] = []

    class Progress:
        def __enter__(self) -> "Progress":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def update(self, amount: int) -> None:
            updates.append(amount)

    def progress_factory(*args: object, **kwargs: object) -> Progress:
        progress_calls.append((args, kwargs))
        return Progress()

    monkeypatch.setattr(bounded_prefix_evaluation_utils, "tqdm", progress_factory)

    collect_source_parent_identities(
        tmp_path, scorer_llm="scorer", dataset="source", split="HG", max_per_split=3,
        batch_size=2, progress_description="Collecting parents: scorer/source/HG",
    )

    assert progress_calls == [
        ((), {"total": 3, "desc": "Collecting parents: scorer/source/HG", "unit": "batch", "leave": False}),
    ]
    assert updates == [1, 1, 1]


def test_prefix_prediction_keys_and_paths_differ_by_bounds(tmp_path: Path) -> None:
    first = bounded_prefix_prediction_key(
        Path("models/logreg/run"), models_root=Path("models"), evaluation_regime="Mixed",
        dataset="source", partition="test", min_tokens=16, max_tokens=64,
    )
    second = bounded_prefix_prediction_key(
        Path("models/logreg/run"), models_root=Path("models"), evaluation_regime="Mixed",
        dataset="source", partition="test", min_tokens=32, max_tokens=64,
    )

    assert first["min_tokens"] == 16 and second["min_tokens"] == 32
    assert first["evaluation_window"] == "prefix_16_64_window_32_stride_16"
    assert prediction_artifact_path(tmp_path, first) != prediction_artifact_path(tmp_path, second)


def test_prefix_exports_exclude_unavailable_rows_from_resume_keys(tmp_path: Path) -> None:
    exports = ensure_bounded_prefix_result_exports(tmp_path, "prefix-run")
    metrics = exports["prefix_evaluation_metrics"]
    coverage = exports["prefix_evaluation_coverage"]

    assert set(pd.read_csv(metrics).columns) == {
        "run_id", "artifact_id", "artifact_path", "model_family", "scorer_llm", "feature_set",
        "training_regime", "training_window", "evaluation_regime", "dataset", "partition",
        "min_tokens", "max_tokens", "eligible_parent_count", "scored_row_count", "accuracy",
        "balanced_accuracy", "macro_f1", "roc_auc", "false_positive_rate", "hg_support",
        "mg_support", "tn", "fp", "fn", "tp", "status", "error",
    }
    assert set(exports) == {
        "prefix_evaluation_metrics", "prefix_evaluation_coverage", "prefix_run_summary",
    }

    pd.DataFrame([{"artifact_id": "artifact", "min_tokens": 16, "status": "unavailable"}]).to_csv(
        metrics, index=False,
    )
    pd.DataFrame([{
        "artifact_id": "artifact", "min_tokens": 16, "availability_status": "unavailable",
    }]).to_csv(coverage, index=False)
    assert bounded_prefix_successful_keys(metrics, ("artifact_id", "min_tokens")) == set()
    assert bounded_prefix_successful_keys(coverage, ("artifact_id", "min_tokens")) == set()
