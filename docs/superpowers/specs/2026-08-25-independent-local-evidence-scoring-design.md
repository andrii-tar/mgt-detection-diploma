# Independent Local Evidence Scoring Design

## Scope

Allow the notebook's custom inspection cell to score any text index in either
split, including texts absent from the held-out `scored_windows` table.

## Interface

Expose the fitted per-window logistic pipeline and its feature columns from
`run_local_evidence_experiments(...)` through a separate helper. The helper
will fit on training windows for a requested window length, using the existing
imputation, scaling, and logistic-regression configuration, then score supplied
windows with the existing contribution logic.

## Notebook Flow

The custom cell will:

1. Load the raw stage-1/stage-2 rows by zero-based split index.
2. Build local windows for that one text and requested window length.
3. Fit the selected window-length model using only training windows.
4. Score the custom windows and select the strongest MG window.
5. Display the timeline and existing token heatmap.

The output will be labeled as independent inspection evidence, not a held-out
test prediction. If the text is shorter than the requested window length, the
cell will raise a clear error.

## Testing

Add tests proving the fitted model can score a supplied non-test text and that
the notebook custom cell calls the independent scoring path. Preserve the
output-free notebook contract and run the full test suite.
