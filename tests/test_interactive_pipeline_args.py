from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts import process_pipeline, token_similarity_stage2


def _menu_number(options, key):
    return str(sorted(options).index(key) + 1)


def test_process_pipeline_resolves_missing_args_interactively(monkeypatch):
    args = process_pipeline._parse_args([])
    responses = iter(
        [
            _menu_number(process_pipeline.MODEL_IDS, "gemma_3_1b_it"),
            _menu_number(process_pipeline.DATASETS, "kaggle_train"),
            "",
            "",
            "",
            "",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    resolved = process_pipeline._resolve_args(args)

    assert resolved.dataset == "kaggle_train"
    assert resolved.model == "gemma_3_1b_it"
    assert resolved.count == 100
    assert resolved.inference_batch_size == 2
    assert resolved.write_batch_size == 20
    assert resolved.top_n == 20
    assert resolved.split is None


def test_process_pipeline_keeps_explicit_args():
    args = process_pipeline._parse_args(
        [
            "--dataset",
            "outfox_chatgpt_train",
            "--model",
            "opt_350m",
            "--count",
            "7",
            "--inference-batch-size",
            "3",
            "--write-batch-size",
            "5",
        ]
    )

    resolved = process_pipeline._resolve_args(args)

    assert resolved.dataset == "outfox_chatgpt_train"
    assert resolved.model == "opt_350m"
    assert resolved.count == 7
    assert resolved.inference_batch_size == 3
    assert resolved.write_batch_size == 5
    assert resolved.split is None
    assert resolved.run_stage_2 is True


def test_process_pipeline_accepts_explicit_split():
    args = process_pipeline._parse_args(
        [
            "--dataset",
            "kaggle_train",
            "--model",
            "opt_350m",
            "--count",
            "7",
            "--inference-batch-size",
            "3",
            "--write-batch-size",
            "5",
            "--split",
            "MG",
        ]
    )

    resolved = process_pipeline._resolve_args(args)

    assert resolved.split == "MG"


def test_process_pipeline_can_disable_stage_2():
    args = process_pipeline._parse_args(
        [
            "--dataset",
            "kaggle_train",
            "--model",
            "opt_350m",
            "--count",
            "7",
            "--inference-batch-size",
            "3",
            "--write-batch-size",
            "5",
            "--no-stage-2",
        ]
    )

    resolved = process_pipeline._resolve_args(args)

    assert resolved.run_stage_2 is False


def test_process_pipeline_runs_stage_2_after_stage_1(monkeypatch):
    args = process_pipeline._parse_args(
        [
            "--dataset",
            "kaggle_train",
            "--model",
            "opt_350m",
            "--count",
            "7",
            "--inference-batch-size",
            "3",
            "--write-batch-size",
            "5",
        ]
    )

    calls = []
    monkeypatch.setattr(process_pipeline, "load_local_model", lambda model_id: ("tokenizer", "model", "device"))
    monkeypatch.setattr(process_pipeline, "_process_split", lambda *args: calls.append(("stage1", args[0])))
    monkeypatch.setattr(process_pipeline, "run_stage_2", lambda **kwargs: calls.append(("stage2", kwargs)))

    process_pipeline.main(args)

    assert calls[:2] == [("stage1", "HG"), ("stage1", "MG")]
    assert calls[2] == (
        "stage2",
        {
            "dataset": "kaggle_train",
            "model_name": "opt_350m",
            "model_id": "facebook/opt-350m",
            "count": 14,
            "splits": ["HG", "MG"],
        },
    )


def test_process_pipeline_runs_only_selected_split(monkeypatch):
    args = process_pipeline._parse_args(
        [
            "--dataset",
            "kaggle_train",
            "--model",
            "opt_350m",
            "--count",
            "7",
            "--inference-batch-size",
            "3",
            "--write-batch-size",
            "5",
            "--split",
            "MG",
        ]
    )

    calls = []
    monkeypatch.setattr(process_pipeline, "load_local_model", lambda model_id: ("tokenizer", "model", "device"))
    monkeypatch.setattr(process_pipeline, "_process_split", lambda *args: calls.append(("stage1", args[0])))
    monkeypatch.setattr(process_pipeline, "run_stage_2", lambda **kwargs: calls.append(("stage2", kwargs)))

    process_pipeline.main(args)

    assert calls[:1] == [("stage1", "MG")]
    assert calls[1] == (
        "stage2",
        {
            "dataset": "kaggle_train",
            "model_name": "opt_350m",
            "model_id": "facebook/opt-350m",
            "count": 7,
            "splits": ["MG"],
        },
    )

def test_similarity_stage2_resolves_missing_args_interactively(monkeypatch):
    args = token_similarity_stage2._parse_args([])
    responses = iter(
        [
            _menu_number(token_similarity_stage2.MODEL_IDS, "gpt2"),
            _menu_number(token_similarity_stage2.DATASETS, "outfox_chatgpt_train"),
            "",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    resolved = token_similarity_stage2._resolve_args(args)

    assert resolved.model == "gpt2"
    assert resolved.dataset == "outfox_chatgpt_train"
    assert resolved.count == 200
