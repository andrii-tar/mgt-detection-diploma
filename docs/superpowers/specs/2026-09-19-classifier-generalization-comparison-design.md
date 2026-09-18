# Classifier Generalization Comparison Design

## Goal

Provide a read-only notebook that identifies classifier configurations most likely
to generalize to unknown binary datasets. Rank candidates first by mean held-out
binary macro-F1, then worst held-out binary macro-F1, and finally SemEval
segmentation boundary MAE. Show every result for all eligible texts and for
parent texts with at least a configurable number of tokens.

## Scope

Create `notebooks/compare_classifier_generalization.ipynb` and supporting
version-controlled utility functions. The notebook reads existing classifier
evaluation CSVs and partitioned prediction artifacts under `TEXT_DETECTION_DATA`;
it does not train models, run inference, change result artifacts, or modify the
existing general analysis notebook.

## Configuration

The first code cell exposes:

- `RUN_ID`: saved classifier evaluation run to analyze.
- `MIN_TEXT_TOKENS = 200`: long-text threshold.
- `TOP_TIER_MACRO_F1_GAP = 0.02`: maximum mean held-out macro-F1 gap from the
  leader for segmentation tie-break comparison.

The notebook evaluates two named cohorts:

- `All texts`: every eligible scored parent text.
- `>= MIN_TEXT_TOKENS`: parent texts whose maximum scored `stop_token` is at
  least the threshold. This reconstruction is derived from prediction artifacts
  and does not require raw input data.

SemEval uses its recorded `token_count` to apply the same two cohorts.

## Evaluation Rules

1. A binary held-out result is one whose evaluation dataset is not represented
   by the training regime and whose parent set contains both HG and MG labels.
2. Rank each full configuration by the mean of its per-dataset macro-F1 values.
3. Break equal or near-equal primary scores with the minimum per-dataset
   macro-F1 value.
4. Compare SemEval boundary MAE only for candidates within
   `TOP_TIER_MACRO_F1_GAP` of the primary leader; lower MAE is preferred.
5. BBC HG-only results are a safety diagnostic. They contribute false-positive
   rate but never binary macro-F1 or the candidate ranking.
6. If no held-out binary dataset is available, the notebook must display an
   explicit insufficient-evidence message and avoid selecting a winner.

## Visualizations

- Paired generalization leaderboard for both cohorts. Each configuration shows
  mean held-out binary macro-F1 and its worst-to-best dataset range.
- Paired configuration-by-dataset heatmap for leading candidates, with dataset
  columns ordered by mean difficulty.
- Paired top-tier scatter plot: mean held-out macro-F1 on the x-axis and SemEval
  boundary MAE on the y-axis. It labels Pareto-optimal candidates.
- Paired parameter ablation plots for scorer LLM, model family, feature set,
  training regime, training window, and evaluation window. Effects compare only
  rows that match on all remaining configuration dimensions.
- Paired BBC safety frontier of held-out binary macro-F1 versus BBC
  false-positive rate, clearly labeled as a safety constraint.
- Cohort coverage table reporting datasets, unique parents, text counts, and
  binary held-out availability for each cohort.

## Data Flow And Interfaces

1. Load compact evaluation metrics and SemEval tables using the existing
   analysis utilities.
2. Discover prediction artifacts for the selected run and load detailed rows
   only as needed to calculate per-parent maximum `stop_token` and filtered
   binary metrics.
3. Calculate all-text and long-text summaries independently. Never mix rows
   across cohorts.
4. Return compact DataFrames from reusable utility functions for coverage,
   binary generalization summaries, segmentation summaries, and paired plot
   inputs. Plot functions accept those DataFrames and return Matplotlib figures.

## Validation

- Unit-test long-text parent selection from partitioned prediction artifacts.
- Unit-test held-out binary classification, explicitly excluding HG-only BBC.
- Unit-test ranking order: mean macro-F1, then worst macro-F1, then MAE only for
  top-tier candidates.
- Unit-test every summary includes both cohort labels and preserves empty or
  insufficient-evidence states.
- Notebook contract tests require output-free cells, first-cell parameters, and
  all decision plots.
