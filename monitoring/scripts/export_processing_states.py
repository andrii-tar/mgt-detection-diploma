from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def collect_states(processed_root: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    if not processed_root.exists():
        return states

    for state_path in sorted(processed_root.glob("**/state.json")):
        try:
            states.append(json.loads(state_path.read_text(encoding="utf-8")))
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
