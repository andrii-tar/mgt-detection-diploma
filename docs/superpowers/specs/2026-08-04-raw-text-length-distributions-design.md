# Raw Text Length Distribution Plots

## Goal

Add a standalone script that creates word-count histograms for every configured
raw dataset. Each dataset gets one saved figure with separate HG and MG
histograms.

## Scope

- Use the dataset aliases and split paths from `scripts/experiment_config.py`.
- Resolve the input root through `TEXT_DETECTION_DATA` using the shared data-root
  helper.
- Read the `text` column from each existing HG/MG CSV.
- Count words with Python whitespace splitting (`len(text.split())`).
- Use 50 histogram bins by default.
- Save one PNG per dataset to
  `TEXT_DETECTION_DATA/processed/text_length_distributions/`.
- Allow selecting datasets and overriding the output directory from the CLI.

Source CSVs that are not configured HG/MG inputs must not be discovered or
plotted, preventing source data from being double-counted.

## Plot Layout

Each dataset produces one Matplotlib figure:

- One subplot for each available split.
- HG and MG are plotted separately rather than overlaid.
- Datasets with only one configured split produce a one-subplot figure.
- The X-axis is `Word count` and the Y-axis is `Document count`.
- The figure title identifies the dataset.
- Output filenames use the dataset alias with a `.png` suffix.

## CLI

The script supports:

```text
uv run python scripts/plot_raw_text_length_distributions.py
uv run python scripts/plot_raw_text_length_distributions.py --dataset mage_train
uv run python scripts/plot_raw_text_length_distributions.py --output-dir <path>
```

Without `--dataset`, all configured datasets are processed. A missing split
file is skipped with a warning. A present CSV without a `text` column is an
error with a clear message.

## Testing

Tests will verify:

- whitespace-based word counts, including empty text;
- configured HG/MG paths are used instead of recursive raw-file discovery;
- figures use 50 bins and separate split axes;
- one PNG is written per processed dataset;
- missing files are skipped and invalid CSV schemas fail clearly.
