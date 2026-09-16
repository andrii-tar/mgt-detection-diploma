# SemEval Per-Text Plotting Design

## Goal

For each selected SemEval text in the best, worst, and random groups, produce three comparable figures for window sizes 32, 64, and 128.

## Plot Contract

1. Boundary comparison uses one token-position axis. The gold boundary is a green dashed vertical line. Each available predicted boundary has a distinct labeled line for its window size. Predicted MG regions may be shown as transparent spans. Requested windows without data are represented in the legend as unavailable rather than silently omitted.
2. Window-level MG scores use one axis with one line per available window, plotted at window midpoints. The gold MG region is shaded and the gold boundary is marked.
3. Token-level MG scores use one axis with one reconstructed token-score line per available window. The gold MG region is shaded and the gold boundary is marked.

## Data Flow

The notebook continues to select five best, five worst, and five fixed-seed random texts from persisted SemEval evaluation tables. It reuses persisted boundary/region exports and prepared partitions; it does not rerun large inference jobs. Plotting receives only the requested window sizes in the stable order 32, 64, 128.

## Missing Data

Plots must remain valid when one or more requested windows are absent for a text. The figure should retain the requested window in its legend with an unavailable indication, while drawing traces only for data that exists.

## Testing

Tests will assert one axis per figure, explicit per-window labels/traces, gold-region markers, stable window ordering, and graceful handling of missing windows. Notebook contract tests will assert the requested window configuration is passed to the plotting section.
