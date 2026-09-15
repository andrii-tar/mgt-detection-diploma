# Universal Transfer Analysis Design

## Goal

Extend `notebooks/analyze_classifier_results.ipynb` with additive analyses for
the research goal of a universal text detector: a model that maintains reliable
performance across sources rather than only on its training source.

## Scope

All existing notebook tables and plots remain. The extension appends
cross-source transfer matrices, robust-model selection tables, and multivariate
visualizations. It reads one existing run only and does not change training,
evaluation, or result files.

## Universal Selection Rule

The primary score is worst eligible binary-source test macro-F1. Mean eligible
source macro-F1 breaks ties.

For a single-source training regime, the same-source test cell is excluded from
the primary universal score because it measures in-distribution performance,
not transfer. The remaining binary source cells form the eligible unseen-source
set. For `Mixed` and `Mixed+BBC`, all binary source test cells are included and
are labelled mixed-source robustness rather than unseen-source transfer.

BBC remains separate from the binary ranking. It is an HG-only safety test and
is reported through false-positive rate and predicted-MG rate.

## Additive Tables

- Full configuration transfer table with training regime, test dataset, macro
  F1, eligibility, and whether a cell is in-distribution.
- Robust universal ranking with worst eligible macro-F1, mean eligible macro-F1,
  standard deviation, eligible-source count, and configuration dimensions.
- Per-LLM robust selection table that identifies the best configuration by the
  lexicographic universal selection rule.
- Explicit excluded same-source transfer rows.
- Pareto table containing configurations not dominated on both worst-source and
  mean eligible macro-F1.

## Additive Visualizations

- A small-multiple grid of training-regime by test-dataset macro-F1 heatmaps,
  one per `(scorer LLM, feature set, classifier family, training window,
  evaluation window)` configuration. Missing cells remain blank and are not
  treated as zero.
- A focused best-per-LLM heatmap using the universal selection rule.
- A frontier scatter with mean eligible macro-F1 on the x-axis and worst
  eligible macro-F1 on the y-axis. Color identifies scorer LLM, marker shape
  identifies feature set, and Pareto-optimal configurations are labelled.
- A stability-profile line plot across binary test datasets. One line is one
  configuration; the selected robust configuration per LLM is emphasized.
- A directed single-source transfer diagram. Source-training and target-test
  nodes are connected only for cross-source pairs; edge color and width encode
  macro-F1. This exposes asymmetric transfer.
- A SemEval trade-off scatter with MG-region F1 on the x-axis and mean absolute
  boundary token error on the y-axis, with each configuration labelled by its
  scorer LLM and feature set.

## Availability And Interpretation

The current `feature-vector-classification-2026-09-12` run contains only
`Mixed` artifacts, so its matrices will have one training-regime row. The
notebook must display that availability constraint rather than inventing
single-source or `Mixed+BBC` values. A run containing the other artifacts will
populate the same analyses automatically.

All conclusions are descriptive. The notebook must state when rankings rely on
legacy positional recovery, incomplete source coverage, one seed, or a single
classifier family.

## Testing

Unit tests cover same-source exclusion, mixed-regime inclusion, lexicographic
robust ranking, Pareto marking, matrix construction with missing values,
transfer-edge construction, and SemEval trade-off inputs. Notebook contract
tests assert that the new transfer helpers and visualization sections are
present while existing analysis fragments remain present.
