import math

import pandas as pd
import pytest

from scripts.logprob_metrics import compute_logprob_metrics, recreate_output, iter_unprocessed_rows


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
