from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts.data_root import resolve_data_root
from scripts.experiment_config import build_datasets


LOGGER = logging.getLogger(__name__)
DEFAULT_BINS = 50
DEFAULT_OUTPUT_NAME = Path("processed") / "text_length_distributions"


def count_words(value: object) -> int:
    missing = pd.isna(value)
    if value is None or (isinstance(missing, bool) and missing):
        return 0
    return len(str(value).split())


def load_word_counts(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "text" not in frame.columns:
        raise ValueError(f"Missing required column 'text' in {path}")
    return frame["text"].map(count_words).astype("int64")


def trim_to_95th_percentile(counts: pd.Series) -> pd.Series:
    cutoff = counts.quantile(0.95)
    return counts[counts <= cutoff]


def plot_dataset(
    dataset: str,
    splits: dict[str, Path],
    output_dir: Path,
    bins: int = DEFAULT_BINS,
) -> Path | None:
    available_splits = [(split, path) for split, path in splits.items() if path.exists()]
    if not available_splits:
        LOGGER.warning("No input files found for dataset %s", dataset)
        return None

    figure, axes = plt.subplots(1, len(available_splits), squeeze=False, figsize=(7 * len(available_splits), 5))
    for axis, (split, path) in zip(axes[0], available_splits):
        counts = load_word_counts(path)
        axis.hist(counts, bins=bins, color={"HG": "#4c78a8", "MG": "#f58518"}.get(split, "#777777"))
        axis.set_title(split)
        axis.set_xlabel("Word count")
        axis.set_ylabel("Document count")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(f"Text length distribution: {dataset}")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    for split, path in splits.items():
        if not path.exists():
            LOGGER.warning("Skipping missing %s file for %s: %s", split, dataset, path)
    return output_path


def plot_combined_dataset(
    dataset: str,
    splits: dict[str, Path],
    output_dir: Path,
    bins: int = DEFAULT_BINS,
) -> Path | None:
    available_splits = [(split, path) for split, path in splits.items() if path.exists()]
    if not available_splits:
        LOGGER.warning("No input files found for dataset %s", dataset)
        return None

    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {"HG": "#4c78a8", "MG": "#f58518"}
    for split, path in available_splits:
        counts = trim_to_95th_percentile(load_word_counts(path))
        axis.hist(counts, bins=bins, color=colors.get(split, "#777777"), label=split, alpha=0.65)

    axis.set_title(f"Text length distribution (95%): {dataset}")
    axis.set_xlabel("Word count")
    axis.set_ylabel("Document count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_combined_95.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    for split, path in splits.items():
        if not path.exists():
            LOGGER.warning("Skipping missing %s file for %s: %s", split, dataset, path)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw text word-count distributions for configured datasets.")
    parser.add_argument("--dataset", action="append", help="Dataset alias to plot; repeat to select multiple datasets.")
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG files.")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins (default: 50).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    data_root = resolve_data_root()
    datasets = build_datasets(data_root)
    selected_datasets = args.dataset or list(datasets)
    unknown = sorted(set(selected_datasets) - set(datasets))
    if unknown:
        raise ValueError(f"Unknown dataset alias(es): {', '.join(unknown)}")

    output_dir = args.output_dir or data_root / DEFAULT_OUTPUT_NAME
    for dataset in selected_datasets:
        plot_dataset(dataset, datasets[dataset]["splits"], output_dir, bins=args.bins)
        plot_combined_dataset(dataset, datasets[dataset]["splits"], output_dir, bins=args.bins)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
