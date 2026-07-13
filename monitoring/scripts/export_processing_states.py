from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from experiment_config import DATASETS


STAGE_OUTPUTS = {
    "logprobs": "data",
    "token_similarity": "token_similarities",
}


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)

    return max(row_count - 1, 0)


def dataset_sizes() -> dict[str, dict[str, int]]:
    sizes: dict[str, dict[str, int]] = {}
    for dataset, config in DATASETS.items():
        splits = config.get("splits", {})
        sizes[dataset] = {
            split: size
            for split, split_path in splits.items()
            if (size := count_csv_rows(Path(split_path))) is not None
        }
    return sizes


def enrich_state(state: dict[str, Any], sizes: dict[str, dict[str, int]], state_path: Path) -> dict[str, Any]:
    stage = state.get("stage") or "logprobs"
    output_dir = state_path.parent / STAGE_OUTPUTS.get(stage, "data")
    parquet_files = sorted(output_dir.glob("*.parquet")) if output_dir.exists() else []

    state["stage"] = stage
    state["output_format"] = "parquet"
    state["batch_count"] = len(parquet_files)
    state["output_path"] = str(output_dir)

    dataset = state.get("dataset")
    model = state.get("model")
    split = state.get("split")
    if model and split:
        state["model_split"] = f"{model} / {split}"

    dataset_size = sizes.get(dataset, {}).get(split)
    if dataset_size is None:
        return state

    processed_count = int(state.get("processed_count", 0))
    state["dataset_size"] = dataset_size
    state["processed_percent"] = processed_count / dataset_size if dataset_size else 0
    return state


def collect_states(processed_root: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    if not processed_root.exists():
        return states

    sizes = dataset_sizes()
    state_paths = sorted(processed_root.glob("**/state.json")) + sorted(processed_root.glob("**/similarity_state.json"))
    for state_path in state_paths:
        try:
            states.append(enrich_state(json.loads(state_path.read_text(encoding="utf-8")), sizes, state_path))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON state file {state_path}: {exc}")

    return states


def write_states(processed_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    states = collect_states(processed_root)
    output.write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(states)} processing states to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processing state files into one JSON array for Grafana.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("processing-states.json"))
    parser.add_argument("--watch", action="store_true", help="Keep refreshing the output file.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds when --watch is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    while True:
        write_states(args.processed_root, args.output)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
