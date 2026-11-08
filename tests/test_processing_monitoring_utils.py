from __future__ import annotations

from pathlib import Path

import pandas as pd

from notebooks.utils.processing_monitoring_utils import (
    STAGES,
    scan_postprocessing_readiness,
    scan_processing_counts,
)


def _datasets(tmp_path: Path) -> dict[str, dict[str, object]]:
    return {
        "demo": {
            "dataset_name": "demo",
            "splits": {
                "HG": tmp_path / "raw_hg.csv",
                "MG": tmp_path / "raw_mg.csv",
            },
        }
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_scan_covers_every_configured_combination_and_stage(tmp_path: Path) -> None:
    result = scan_processing_counts(tmp_path, _datasets(tmp_path), {"model_a": "id-a", "model_b": "id-b"})

    assert list(result.columns) == [
        "dataset", "model", "split", "stage", "processed_texts", "status", "path", "error"
    ]
    assert len(result) == 2 * 2 * len(STAGES)
    assert set(result["stage"]) == set(STAGES)
    assert (result["processed_texts"] == 0).all()
    assert (result["status"] == "missing").all()


def test_scan_counts_artifacts_and_distinct_local_evidence_ids(tmp_path: Path) -> None:
    datasets = _datasets(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]}).to_csv(
        tmp_path / "raw_hg.csv", index=False
    )
    pd.DataFrame({"id": [4, 5], "text": ["d", "e"]}).to_csv(tmp_path / "raw_mg.csv", index=False)

    base = tmp_path / "processed" / "demo__model_a"
    _write_parquet(base / "HG" / "data" / "part-0.parquet", [{"id": 1}, {"id": 2}])
    _write_parquet(base / "HG" / "data" / "part-1.parquet", [{"id": 3}])
    _write_parquet(base / "HG" / "token_similarities" / "part.parquet", [{"id": 1}, {"id": 2}])
    _write_parquet(base / "HG" / "metrics" / "metrics.parquet", [{"id": 1}])
    _write_parquet(base / "HG" / "token_similarity_stats" / "token_similarity_stats.parquet", [{"id": 1}, {"id": 2}])
    _write_parquet(
        base / "local_evidence" / "windows.parquet",
        [{"id": 1, "split": "HG"}, {"id": 1, "split": "HG"}, {"id": 2, "split": "HG"}, {"id": 7, "split": "MG"}],
    )
    _write_parquet(base / "MG" / "data" / "empty.parquet", [])

    result = scan_processing_counts(tmp_path, datasets, {"model_a": "id-a"})
    hg = result[result["split"] == "HG"].set_index("stage")
    mg = result[result["split"] == "MG"].set_index("stage")

    assert hg.loc["raw_input", "processed_texts"] == 3
    assert hg.loc["logprobs", "processed_texts"] == 3
    assert hg.loc["token_similarity", "processed_texts"] == 2
    assert hg.loc["logprob_metrics", "processed_texts"] == 1
    assert hg.loc["token_similarity_stats", "processed_texts"] == 2
    assert hg.loc["local_evidence", "processed_texts"] == 2
    assert mg.loc["logprobs", "status"] == "empty"
    assert mg.loc["local_evidence", "processed_texts"] == 1


def test_scan_postprocessing_readiness_reports_ready_outputs(tmp_path: Path) -> None:
    base = tmp_path / "processed" / "demo__model_a"
    for split in ("HG", "MG"):
        split_root = base / split
        _write_parquet(split_root / "postprocessed" / "stage1.parquet", [{"id": 1}, {"id": 2}])
        _write_parquet(split_root / "postprocessed" / "stage2.parquet", [{"id": 1}, {"id": 2}])
        _write_parquet(split_root / "metrics" / "metrics.parquet", [{"id": 1}, {"id": 2}])
        _write_parquet(
            split_root / "token_similarity_stats" / "token_similarity_stats.parquet",
            [{"id": 1}, {"id": 2}],
        )
    _write_parquet(
        base / "local_evidence" / "windows.parquet",
        [
            {"split": "HG", "text_id": 1},
            {"split": "HG", "text_id": 2},
            {"split": "MG", "text_id": 1},
            {"split": "MG", "text_id": 2},
        ],
    )

    result = scan_postprocessing_readiness(tmp_path / "processed")

    assert list(result["status"]) == ["READY", "READY"]
    assert set(result["dataset"]) == {"demo"}
    assert set(result["model"]) == {"model_a"}
    assert (result["stage1_rows"] == 2).all()
    assert (result["stage2_rows"] == 2).all()
    assert (result["missing_stage2"] == 0).all()
    assert (result["extra_stage2"] == 0).all()
    assert (result["metrics_rows"] == 2).all()
    assert (result["similarity_stats_rows"] == 2).all()


def test_scan_postprocessing_readiness_reports_id_mismatch_and_missing_outputs(tmp_path: Path) -> None:
    split_root = tmp_path / "processed" / "demo__model_a" / "HG"
    _write_parquet(split_root / "postprocessed" / "stage1.parquet", [{"id": 1}, {"id": 2}, {"id": 2}])
    _write_parquet(split_root / "postprocessed" / "stage2.parquet", [{"id": 2}, {"id": 3}])

    result = scan_postprocessing_readiness(tmp_path / "processed").iloc[0]

    assert result["status"] == "ID_MISMATCH"
    assert result["missing_stage2"] == 1
    assert result["extra_stage2"] == 1
    assert result["duplicate_stage1"] == 1
    assert result["metrics_rows"] == 0
    assert result["similarity_stats_rows"] == 0
