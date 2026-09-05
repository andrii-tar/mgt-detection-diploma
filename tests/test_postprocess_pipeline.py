from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.postprocess_pipeline import _process_split, compare_ids, consolidate_parquet_files, run_postprocessing


def _write_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "value": [f"row-{value}" for value in ids]}).to_parquet(path, index=False)


def test_compare_ids_reports_missing_extra_and_duplicates() -> None:
    comparison = compare_ids([1, 2, 2, 3], [2, 3, 4, 4])

    assert comparison.missing_stage2_ids == {1}
    assert comparison.extra_stage2_ids == {4}
    assert comparison.duplicate_stage1 == 1
    assert comparison.duplicate_stage2 == 1
    assert comparison.is_exact is False


def test_consolidates_batches_sorted_by_id_without_changing_sources(tmp_path: Path) -> None:
    first = tmp_path / "data" / "batch_00000002_00000003.parquet"
    second = tmp_path / "data" / "batch_00000000_00000001.parquet"
    _write_ids(first, [3, 2])
    _write_ids(second, [1, 0])
    output = tmp_path / "postprocessed" / "stage1.parquet"

    result = consolidate_parquet_files([first, second], output)

    assert result == output
    assert pd.read_parquet(output)["id"].tolist() == [0, 1, 2, 3]
    assert pd.read_parquet(first)["id"].tolist() == [3, 2]
    assert pd.read_parquet(second)["id"].tolist() == [1, 0]


def test_process_split_rebuilds_canonical_stage_outputs(tmp_path: Path, monkeypatch) -> None:
    split_dir = tmp_path / "dataset_a__model_a" / "HG"
    stage1 = split_dir / "data" / "batch_00000000_00000001.parquet"
    stage2 = split_dir / "token_similarities" / "batch_00000000_00000001.parquet"
    _write_ids(stage1, [1, 0])
    _write_ids(stage2, [1, 0])
    rebuilt: list[str] = []
    monkeypatch.setattr("scripts.postprocess_pipeline._write_metrics", lambda path, _: rebuilt.append(f"metrics:{path.name}"))
    monkeypatch.setattr("scripts.postprocess_pipeline._write_similarity_stats", lambda path, _: rebuilt.append(f"stats:{path.name}"))

    status, comparison = _process_split(split_dir)

    assert status == "COMPLETE"
    assert comparison.is_exact
    assert rebuilt == ["metrics:stage1.parquet", "stats:stage2.parquet"]
    assert pd.read_parquet(split_dir / "postprocessed" / "stage1.parquet")["id"].tolist() == [0, 1]
    assert pd.read_parquet(split_dir / "postprocessed" / "stage2.parquet")["id"].tolist() == [0, 1]


def test_run_postprocessing_reports_progress_per_dataset_model_pair(tmp_path, monkeypatch, capsys):
    experiment = tmp_path / "demo__model_a"
    for split in ("HG", "MG"):
        (experiment / split / "data").mkdir(parents=True)

    progress = {}

    class RecordingBar:
        def __init__(self, iterable, **kwargs):
            progress.update(kwargs)
            self.iterable = iterable
            self.updates = 0
            progress["bar"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def update(self, amount):
            self.updates += amount

    monkeypatch.setattr("scripts.postprocess_pipeline.tqdm", RecordingBar)
    monkeypatch.setattr(
        "scripts.postprocess_pipeline._process_split",
        lambda split_dir: ("COMPLETE", compare_ids([1], [1])),
    )

    assert run_postprocessing(tmp_path) == 0
    assert {key: progress[key] for key in ("desc", "total", "unit")} == {
        "desc": "demo/model_a",
        "total": 2,
        "unit": "split",
    }
    assert progress["bar"].updates == 2
    assert "SUMMARY: splits=2 complete=2 invalid=0 local_evidence=0" in capsys.readouterr().out
