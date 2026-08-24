# Rolling Statistics Histograms

## Goal

Calculate rolling-smoothed statistics for every text in both splits of a
selected dataset/model pair and visualize their distributions.

## Design

Add reusable notebook utility logic that processes all available texts in the
`HG` and `MG` splits for a selected dataset/model pair. For each text, it will
build the existing token-level summary, apply the existing rolling mean with a
default window of 5, and calculate the mean, population standard deviation,
and population variance of each finite smoothed metric.

The calculation returns a tidy DataFrame with one row per split, text ID, and
metric. It contains the split, text ID, metric name, mean, standard deviation,
and variance, allowing callers to inspect or export the results.

Plotting will produce three figures: one for per-text means, one for per-text
standard deviations, and one for per-text variances. Each figure will have
metric rows and separate HG/MG columns, so the class distributions remain
separate and metrics with different scales are not overlaid.

## Data Handling

The utility will read all stage-1 and token-similarity Parquet files under the
selected dataset/model split. Texts without a matching similarity row will
raise an explicit error rather than silently producing incomplete statistics.
Non-finite smoothed values will be excluded; a text with no finite values for
a metric will receive NaN statistics and will not contribute to that metric's
histogram.

## Verification

Tests will use temporary Parquet fixtures for both HG and MG, verify the tidy
per-text statistics and window-5 calculation, and verify that the plotting
function creates three figures with separate split columns. Existing notebook
utility tests and the full suite will also be run.
