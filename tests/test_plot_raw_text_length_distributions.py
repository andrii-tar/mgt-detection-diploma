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
    output_dir = tmp_path / "processed" / "text_length_distributions"
    assert (output_dir / "selected.png").exists()
    assert (output_dir / "selected_combined_95.png").exists()
    assert not (output_dir / "other.png").exists()
    assert not (output_dir / "other_combined_95.png").exists()


def test_trim_to_95th_percentile_is_applied_to_each_split_independently():
    hg = pd.Series([1, 2, 3, 4, 100])
    mg = pd.Series([10, 20, 30, 40, 1000])

    assert plotter.trim_to_95th_percentile(hg).tolist() == [1, 2, 3, 4]
    assert plotter.trim_to_95th_percentile(mg).tolist() == [10, 20, 30, 40]


def test_plot_combined_dataset_saves_overlay_image_with_separate_colors(tmp_path: Path, monkeypatch):
    hg = tmp_path / "hg.csv"
    mg = tmp_path / "mg.csv"
    pd.DataFrame({"text": ["one", "one two", "one two three", "one two three four", "x " * 100]}).to_csv(hg, index=False)
    pd.DataFrame({"text": ["one", "one two", "one two three", "one two three four", "y " * 200]}).to_csv(mg, index=False)
    calls = []
    original_hist = Axes.hist

    def capture_hist(self, values, *args, **kwargs):
        calls.append((list(values), kwargs["color"], kwargs["label"], kwargs["bins"]))
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    output = plotter.plot_combined_dataset("demo", {"HG": hg, "MG": mg}, tmp_path / "plots")

    assert output == tmp_path / "plots" / "demo_combined_95.png"
    assert output.exists()
    assert [call[2] for call in calls] == ["HG", "MG"]
    assert [call[3] for call in calls] == [50, 50]
    assert max(calls[0][0]) < 100
    assert max(calls[1][0]) < 200
