from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import pytest

from scripts import plot_raw_text_length_distributions as plotter
from scripts.plot_raw_text_length_distributions import count_words, load_word_counts


def test_count_words_uses_whitespace_and_treats_empty_values_as_zero():
    assert count_words("  two   words\n") == 2
    assert count_words("") == 0
    assert count_words(None) == 0


def test_load_word_counts_requires_text_column(tmp_path: Path):
    path = tmp_path / "split.csv"
    pd.DataFrame({"id": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="text"):
        load_word_counts(path)


def test_plot_dataset_saves_separate_split_histograms_with_50_bins(tmp_path: Path, monkeypatch):
    hg = tmp_path / "hg.csv"
    mg = tmp_path / "mg.csv"
    pd.DataFrame({"text": ["one two", "three"]}).to_csv(hg, index=False)
    pd.DataFrame({"text": ["one", "two three four"]}).to_csv(mg, index=False)
    bins_seen = []
    original_hist = Axes.hist

    def capture_hist(self, values, *args, **kwargs):
        bins_seen.append(kwargs["bins"])
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    output = plotter.plot_dataset("demo", {"HG": hg, "MG": mg}, tmp_path / "plots")

    assert output == tmp_path / "plots" / "demo.png"
    assert output.exists()
    assert bins_seen == [50, 50]


def test_plot_dataset_saves_single_split_and_skips_missing_file(tmp_path: Path):
    hg = tmp_path / "hg.csv"
    pd.DataFrame({"text": ["one two"]}).to_csv(hg, index=False)

    output = plotter.plot_dataset(
        "demo",
        {"HG": hg, "MG": tmp_path / "missing.csv"},
        tmp_path / "plots",
    )

    assert output == tmp_path / "plots" / "demo.png"
    assert output.exists()


def test_main_processes_only_selected_configured_dataset(tmp_path: Path, monkeypatch):
    selected_hg = tmp_path / "selected_hg.csv"
    other_hg = tmp_path / "other_hg.csv"
    pd.DataFrame({"text": ["one two"]}).to_csv(selected_hg, index=False)
    pd.DataFrame({"text": ["three four"]}).to_csv(other_hg, index=False)
    datasets = {
        "selected": {"splits": {"HG": selected_hg}},
        "other": {"splits": {"HG": other_hg}},
    }
    monkeypatch.setattr(plotter, "resolve_data_root", lambda: tmp_path)
    monkeypatch.setattr(plotter, "build_datasets", lambda data_root: datasets)

    assert plotter.main(["--dataset", "selected"]) == 0
    assert (tmp_path / "processed" / "text_length_distributions" / "selected.png").exists()
    assert not (tmp_path / "processed" / "text_length_distributions" / "other.png").exists()
