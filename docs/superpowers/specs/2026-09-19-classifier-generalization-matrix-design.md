# Classifier Generalization Matrix Visualization Design

## Goal

Replace dense flat classifier comparison plots with compact matrices that preserve all configuration dimensions without rendering long concatenated labels.

## Layout

Each matrix uses:

- Rows: `scorer_llm`
- Columns: `feature_set`
- Separate matrix facets for `model_family`, `training_window`, and `evaluation_window`
- Separate sections for `All texts` and `>= MIN_TEXT_TOKENS`

The matrix cell contains one aggregated metric and a compact numeric annotation. Exact facet values remain in facet titles and a configuration index table, not in row or column labels.

## Metrics

Render separate matrix families for:

- Mean macro-F1 across the selected binary evaluation datasets.
- Worst-case macro-F1 across those datasets.
- SemEval boundary MAE when SemEval results are available.

F1 matrices use a fixed `0..1` scale where higher values are better. MAE matrices use a sequential scale and clearly state that lower values are better. Missing combinations remain blank.

## Evidence Rules

The unseen-binary ranking remains separate from descriptive matrices. If no valid held-out binary dataset exists, display an explicit insufficiency message and do not fabricate ranking cells. BBC HG-only rows remain a separate safety diagnostic and are excluded from the primary binary matrices.

## Data Flow

1. Load compact evaluation and SemEval metric tables.
2. Filter binary rows by cohort.
3. Aggregate duplicate rows by scorer, feature set, model family, windows, and dataset.
4. Aggregate across datasets for mean and worst macro-F1.
5. Group matrix output by model family and window dimensions.
6. Render deterministic matrix facets for each cohort and metric.
7. Show a compact configuration index containing exact facet dimensions and row/column value counts.

## Testing

- Matrix builders expose deterministic row and column indexes.
- Missing combinations remain `NaN`, not zero.
- Mean, worst-case, and MAE use the correct aggregation and direction.
- Empty SemEval input produces a clear empty-state figure.
- Notebook tests reject flat leaderboard and long-label plotting calls.
- The notebook executes end-to-end without cell errors.
