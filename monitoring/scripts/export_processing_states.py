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


def dataset_choices() -> list[str]:
    return sorted(DATASETS)


def write_dataset_choices(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset_choices(), indent=2), encoding="utf-8")


def _state_key(state: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return state.get("dataset"), state.get("model"), state.get("split")


def enrich_state(
    state: dict[str, Any],
    sizes: dict[str, dict[str, int]],
    stage1_counts: dict[tuple[str | None, str | None, str | None], int],
    state_path: Path,
) -> dict[str, Any]:
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

    processed_count = int(state.get("processed_count", 0))
    dataset_size = sizes.get(dataset, {}).get(split)
    if dataset_size is not None:
        state["dataset_size"] = dataset_size

    if stage == "token_similarity":
        denominator = stage1_counts.get(_state_key(state), 0)
        state["progress_total"] = denominator
        state["progress_total_label"] = "stage 1 processed texts"
    else:
        denominator = dataset_size or 0
        state["progress_total"] = denominator
        state["progress_total_label"] = "dataset texts"

    state["processed_percent"] = processed_count / denominator if denominator else 0
    return state


def collect_states(processed_root: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    if not processed_root.exists():
        return states

    sizes = dataset_sizes()
    stage1_paths = sorted(processed_root.glob("**/state.json"))
    stage2_paths = sorted(processed_root.glob("**/similarity_state.json"))
    stage1_counts: dict[tuple[str | None, str | None, str | None], int] = {}

    for state_path in stage1_paths:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            stage1_counts[_state_key(state)] = int(state.get("processed_count", 0))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON state file {state_path}: {exc}")

    state_paths = stage1_paths + stage2_paths
    for state_path in state_paths:
        try:
            states.append(enrich_state(json.loads(state_path.read_text(encoding="utf-8")), sizes, stage1_counts, state_path))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON state file {state_path}: {exc}")

    return states


def write_states(processed_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    states = collect_states(processed_root)
    output.write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(states)} processing states to {output}")


def try_write_states(processed_root: Path, output: Path) -> bool:
    try:
        write_states(processed_root, output)
    except OSError as exc:
        print(f"Failed to export processing states from {processed_root}: {exc}", flush=True)
        return False
    except json.JSONDecodeError as exc:
        print(f"Failed to parse processing state JSON from {processed_root}: {exc}", flush=True)
        return False

    return True


def try_write_dataset_choices(output: Path) -> bool:
    try:
        write_dataset_choices(output)
    except OSError as exc:
        print(f"Failed to export dataset choices to {output}: {exc}", flush=True)
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processing state files into one JSON array for Grafana.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("processing-states.json"))
    parser.add_argument("--datasets-output", type=Path, default=Path("datasets.json"))
    parser.add_argument("--watch", action="store_true", help="Keep refreshing the output file.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds when --watch is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    while True:
        ok = try_write_states(args.processed_root, args.output)
        choices_ok = try_write_dataset_choices(args.datasets_output)
        if not args.watch:
            raise SystemExit(0 if ok and choices_ok else 1)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
