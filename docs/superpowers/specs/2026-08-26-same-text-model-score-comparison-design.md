# Same-Text Model Score Comparison

## Goal

Extend the Bayesian model comparison notebook with a reusable visualization that compares the `mg_score` produced by all available classifiers for the same text. This complements the existing aggregate black-box model comparison and keeps model outputs aligned to the same token positions.

## Scope

- Select a seeded random sample of texts from the existing `test` partition.
- For every selected text and every configured `WINDOW_LENGTHS` value, render one chart.
- Plot one token-position `mg_score` curve for each available model.
- Use the existing `scored_windows` result from `run_local_evidence_model_comparison`.
- Keep the existing heatmap and normalized metric overlay behavior unchanged.
- Keep PyMC optional; charts include only models present in `AVAILABLE_MODELS`.

## Design

Add a plotting utility in `notebooks/local_evidence_comparison_utils.py`:

```text
plot_same_text_model_scores(scored_windows, split, text_id, window_length) -> Figure
```

The utility will:

1. Filter rows by `split`, `text_id`, and `window_length`.
2. Validate that the filter is not empty and that required score/span columns exist.
3. Reuse `expand_model_scores_to_tokens` to average overlapping windows for each model at each token position.
4. Plot one line per model, with token position on the x-axis and `mg_score` on the y-axis.
5. Use a shared y-axis range of `0.0` to `1.0` and a legend identifying models.
6. Include split, text ID, label, and window length in the title when the filtered rows have a consistent label.

The notebook will call `select_comparison_texts(comparison_predictions, count=..., seed=SEED)` to obtain test text identities, then loop over those identities and `WINDOW_LENGTHS`. The selected rows will be used only to identify texts; scores will be filtered from `scored_windows`, ensuring every model is plotted against the same cached windows.

## Data Contract

The new utility requires `scored_windows` columns:

- `model`
- `split`
- `text_id`
- `label`
- `window_length`
- `start_token`
- `stop_token`
- `mg_score`

The output remains a Matplotlib `Figure`. No processed data, model weights, or new persisted artifacts are created.

## Error Handling

- Raise `ValueError` for missing required columns.
- Raise `ValueError` when no rows match a requested text/window combination.
- Preserve the existing span validation and overlap averaging behavior in `expand_model_scores_to_tokens`.
- Do not silently substitute another text or window length.

## Testing

Add utility tests covering:

- One line per model for a selected text and window length.
- Correct filtering so unrelated texts and lengths do not appear.
- Shared `0.0` to `1.0` y-axis limits and informative title.
- Missing columns and empty selections raising `ValueError`.

Update notebook contract tests to require the new utility import/call and the seeded test-text selection loop. Run focused comparison and notebook tests with the non-interactive Matplotlib backend.

## Out of Scope

- Heatmap output for this same-text view.
- Explicit text-ID configuration; selection remains seeded random test-text sampling.
- Overlaying multiple window lengths on one chart.
- Changes to classifier fitting, scoring, thresholding, or persisted data formats.
