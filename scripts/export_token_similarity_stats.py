from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

STATS_FILE_NAME = "token_similarity_stats.parquet"
STATS_COLUMNS = [
    "id",
    "dataset",
    "model",
    "experiment_name",
    "split",
    "selected_similarity_count",
    "selected_similarity_mean",
    "selected_similarity_std",
    "candidate_similarity_token_count",
    "candidate_similarity_mean",
    "candidate_similarity_std",
]


def _data_root() -> Path:
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value is None:
        raise RuntimeError("TEXT_DETECTION_DATA is not set")
    return Path(value)


def _experiment_name(dataset: str, model: str) -> str:
    return f"{dataset}__{model}"


def _similarity_dir(processed_root: Path, dataset: str, model: str, split: str) -> Path:
    return processed_root / _experiment_name(dataset, model) / split / "token_similarities"


def _stats_path(processed_root: Path, dataset: str, model: str, split: str) -> Path:
    return processed_root / _experiment_name(dataset, model) / split / "token_similarity_stats" / STATS_FILE_NAME


def token_similarity_stats(row: pd.Series) -> dict[str, float | int | str | None]:
    selected_similarities = []
    token_candidate_means = []
    token_candidate_stds = []
    for token in row["token_similarities"]:
        top_logprobs = token.get("top_logprobs")
        if top_logprobs is None or len(top_logprobs) == 0:
            continue
        selected_similarity = top_logprobs[0].get("similarity_to_gt")
        if selected_similarity is not None:
            selected_similarities.append(float(selected_similarity))
        candidate_similarities = [
            float(candidate["similarity_to_gt"])
            for candidate in top_logprobs
            if candidate.get("similarity_to_gt") is not None
        ]
        if candidate_similarities:
            series = pd.Series(candidate_similarities, dtype="float64")
            token_candidate_means.append(float(series.mean()))
            token_candidate_stds.append(float(series.std(ddof=0)))

    stats = {
        "id": int(row["id"]),
        "dataset": row["dataset"],
        "model": row["model"],
        "experiment_name": row["experiment_name"],
        "split": row["split"],
        "selected_similarity_count": len(selected_similarities),
        "selected_similarity_mean": None,
        "selected_similarity_std": None,
        "candidate_similarity_token_count": len(token_candidate_means),
        "candidate_similarity_mean": None,
        "candidate_similarity_std": None,
    }
    if selected_similarities:
        selected_series = pd.Series(selected_similarities, dtype="float64")
        stats["selected_similarity_mean"] = float(selected_series.mean())
        stats["selected_similarity_std"] = float(selected_series.std(ddof=0))
    if token_candidate_means:
        stats["candidate_similarity_mean"] = float(pd.Series(token_candidate_means, dtype="float64").mean())
        stats["candidate_similarity_std"] = float(pd.Series(token_candidate_stds, dtype="float64").mean())
    return stats


def export_token_similarity_stats(processed_root: Path, dataset: str, model: str, split: str) -> Path:
    similarity_dir = _similarity_dir(processed_root, dataset, model, split)
    if not similarity_dir.exists():
        raise FileNotFoundError(f"Missing token similarity directory: {similarity_dir}")

    rows = []
    paths = sorted(similarity_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No token similarity Parquet files found under {similarity_dir}")

    for path in tqdm(paths, desc=f"{dataset}/{model}/{split}", unit="file"):
        frame = pd.read_parquet(path)
        rows.extend(token_similarity_stats(row) for _, row in frame.iterrows())

    stats = pd.DataFrame(rows, columns=STATS_COLUMNS).sort_values("id").reset_index(drop=True)
    output_path = _stats_path(processed_root, dataset, model, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(output_path, compression="zstd", index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    from scripts.experiment_config import DATASETS, MODEL_IDS

    parser = argparse.ArgumentParser(description="Export combined token similarity stats for one dataset/model/split.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="Dataset alias from experiment_config.py.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS), help="Model alias from experiment_config.py.")
    parser.add_argument("--split", required=True, help="Split label, e.g. HG or MG.")
    parser.add_argument("--processed-root", type=Path, default=None, help="Defaults to TEXT_DETECTION_DATA/processed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    processed_root = args.processed_root if args.processed_root is not None else _data_root() / "processed"
    output_path = export_token_similarity_stats(processed_root, args.dataset, args.model, args.split)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
