import math
from pathlib import Path

import pandas as pd
import pytest

import scripts.logprob_metrics as logprob_metrics
from scripts.logprob_metrics import (
    compute_logprob_metrics,
    compute_top20_coverage,
    compute_top20_d_score,
    process_stale_metrics,
    recreate_output,
    iter_unprocessed_rows,
    write_metrics_files,
)


def _top20_token(actual_logprob: float) -> dict:
    return {
        "logprob": actual_logprob,
        "top_logprobs": [
            {"logprob": math.log(0.5)},
            {"logprob": math.log(0.25)},
            {"logprob": math.log(0.25)},
        ],
    }


def test_compute_top20_d_score_uses_truncated_weighted_distribution():
    score = compute_top20_d_score([_top20_token(math.log(0.5))])

    assert score == pytest.approx(1.0)


def test_compute_top20_d_score_aggregates_token_variance_at_passage_level():
    score = compute_top20_d_score([
        _top20_token(math.log(0.5)),
        _top20_token(math.log(0.5)),
    ])

    assert score == pytest.approx(math.sqrt(2.0))


def test_compute_top20_d_score_returns_nan_without_valid_candidates():
    score = compute_top20_d_score([{"logprob": -1.0, "top_logprobs": []}])

    assert math.isnan(score)


def test_compute_top20_coverage_returns_original_probability_mass():
    tokens = [{
        "logprob": math.log(0.4),
        "top_logprobs": [
            {"logprob": math.log(0.4)},
            {"logprob": math.log(0.3)},
            {"logprob": math.log(0.2)},
        ],
    }]

    assert compute_top20_coverage(tokens) == pytest.approx(0.9)


def test_compute_logprob_metrics_from_token_logprobs():
    row = {
        "id": 12,
        "dataset": "kaggle_train",
        "dataset_name": "kaggle_train",
        "model": "opt_350m",
        "model_id": "facebook/opt-350m",
        "experiment_name": "kaggle_train__opt_350m",
        "split": "HG",
        "text": "One two three.",
        "tokens": [
            {"logprob": math.log(0.5)},
            {"logprob": math.log(0.25)},
            {"logprob": math.log(0.125)},
        ],
    }

    metrics = compute_logprob_metrics(row)

    expected_mean = sum(token["logprob"] for token in row["tokens"]) / 3
    expected_sigma = math.sqrt(sum((token["logprob"] - expected_mean) ** 2 for token in row["tokens"]) / 3)
    assert metrics["id"] == 12
    assert metrics["split"] == "HG"
    assert metrics["num_tokens"] == 3
    assert metrics["prompt_mu_score"] == pytest.approx(expected_mean)
    assert metrics["prompt_sigma_score"] == pytest.approx(expected_sigma)
    assert metrics["perplexity"] == pytest.approx(math.exp(-expected_mean))
    assert metrics["text_len_symbols"] == len("One two three.")
    assert metrics["text_len_words"] == 3
    assert metrics["text_len_sentences"] == 1
    assert math.isnan(metrics["d_score_top20"])
    assert metrics["d_score_top20_num_tokens"] == 0
    assert math.isnan(metrics["d_score_top20_coverage"])
    assert "dataset" not in metrics
    assert "dataset_name" not in metrics
    assert "model" not in metrics
    assert "model_id" not in metrics
    assert "experiment_name" not in metrics


def test_load_experiment_config_uses_runtime_data_root(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(logprob_metrics, "DATA_ROOT", None)
    monkeypatch.setenv("TEXT_DETECTION_DATA", str(tmp_path))

    logprob_metrics._load_experiment_config()

    assert logprob_metrics.DATA_ROOT == tmp_path


def test_compute_logprob_metrics_accepts_parquet_array_backed_candidates(tmp_path):
    row = {
        "id": 12,
        "split": "HG",
        "text": "One two three.",
        "tokens": [_top20_token(math.log(0.5))],
    }
    input_path = tmp_path / "stage1.parquet"
    pd.DataFrame([row]).to_parquet(input_path, index=False)
    restored_row = pd.read_parquet(input_path).iloc[0].to_dict()

    metrics = compute_logprob_metrics(restored_row)

    assert metrics["d_score_top20"] == pytest.approx(1.0)


def test_compute_logprob_metrics_rejects_missing_logprobs():
    with pytest.raises(ValueError, match="at least one token logprob"):
        compute_logprob_metrics({"id": 1, "text": "", "tokens": []})


def test_iter_unprocessed_rows_count_none_reads_all_rows(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {"id": 1, "tokens": [{"logprob": -1.0}]},
            {"id": 2, "tokens": [{"logprob": -2.0}]},
        ]
    ).to_parquet(input_dir / "batch.parquet", index=False)

    rows = list(iter_unprocessed_rows(input_dir, last_id=-1, count=None))

    assert [row["id"] for row in rows] == [1, 2]


def test_recreate_output_removes_metrics_dir_and_state(tmp_path):
    split_dir = tmp_path / "split"
    metrics_dir = split_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    state_path = split_dir / "metrics_state.json"
    (metrics_dir / "old.parquet").write_text("old", encoding="utf-8")
    state_path.write_text("{}", encoding="utf-8")

    recreate_output(split_dir)

    assert not metrics_dir.exists()
    assert not state_path.exists()


def test_write_metrics_files_creates_combined_parquet_and_csv(tmp_path):
    rows = [
        {"id": 2, "prompt_mu_score": -2.0, "split": "HG"},
        {"id": 1, "prompt_mu_score": -1.0, "split": "HG"},
    ]

    write_metrics_files(tmp_path, rows)

    parquet_path = tmp_path / "metrics.parquet"
    csv_path = tmp_path / "metrics.csv"
    assert parquet_path.exists()
    assert csv_path.exists()
    assert pd.read_parquet(parquet_path)["id"].tolist() == [1, 2]
    assert pd.read_csv(csv_path)["id"].tolist() == [1, 2]


def test_process_stale_metrics_scans_configured_dataset_model_pairs(tmp_path, monkeypatch):
    data_root = tmp_path
    monkeypatch.setattr(logprob_metrics, "DATA_ROOT", data_root)
    datasets = {"dataset_a": {"splits": {"HG": "", "MG": ""}}, "dataset_b": {"splits": {"HG": ""}}}
    models = {"model_a": "model-a", "model_b": "model-b"}

    _write_stage1_rows(data_root, "dataset_a", "model_a", "HG", [0, 1])
    _write_stage1_rows(data_root, "dataset_a", "model_a", "MG", [0, 1])
    _write_stage1_rows(data_root, "dataset_b", "model_a", "HG", [0])
    write_metrics_files(
        data_root / "processed" / "dataset_a__model_a" / "MG" / "metrics",
        [compute_logprob_metrics(_stage1_row(0, "dataset_a", "model_a", "MG"))],
    )
    write_metrics_files(
        data_root / "processed" / "dataset_b__model_a" / "HG" / "metrics",
        [compute_logprob_metrics(_stage1_row(0, "dataset_b", "model_a", "HG"))],
    )

    outputs = process_stale_metrics(datasets, models)

    assert outputs == [
        data_root / "processed" / "dataset_a__model_a" / "HG" / "metrics" / "metrics.parquet",
        data_root / "processed" / "dataset_a__model_a" / "MG" / "metrics" / "metrics.parquet",
    ]
    assert pd.read_parquet(outputs[0])["id"].tolist() == [0, 1]
    assert pd.read_parquet(outputs[1])["id"].tolist() == [0, 1]
    assert pd.read_parquet(data_root / "processed" / "dataset_b__model_a" / "HG" / "metrics" / "metrics.parquet")["id"].tolist() == [0]


def _stage1_row(row_id: int, dataset: str, model: str, split: str) -> dict[str, object]:
    return {
        "id": row_id,
        "dataset": dataset,
        "dataset_name": dataset,
        "model": model,
        "model_id": model,
        "experiment_name": f"{dataset}__{model}",
        "split": split,
        "text": "One two three.",
        "tokens": [{"logprob": -1.0}],
    }


def _write_stage1_rows(data_root, dataset: str, model: str, split: str, ids: list[int]) -> None:
    data_dir = data_root / "processed" / f"{dataset}__{model}" / split / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame([_stage1_row(row_id, dataset, model, split) for row_id in ids]).to_parquet(
        data_dir / f"batch_{ids[0]:08d}_{ids[-1]:08d}.parquet",
        index=False,
    )
