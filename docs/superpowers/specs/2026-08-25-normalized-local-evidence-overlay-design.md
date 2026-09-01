# Normalized Local Evidence Overlay Design

## Goal

Add a single combined plot that compares token-level text metrics with the
window-level `mg_score` for the same text. The plot is intended for visual
inspection of where local machine-generated evidence coincides with changes in
the existing token metrics.

## Scope

- Add reusable plotting logic to
  `notebooks/token_candidate_similarity_heatmaps_utils.py`.
- Use the helper from the custom inspection cells in
  `notebooks/local_evidence_classification.ipynb`.
- Do not change the classifier, window model, thresholds, heatmaps, or
  existing plots.
- Keep the notebook output-free.

## Data Alignment

The helper receives the token-level metric summary for one text and scored
local-evidence windows for that same text.

- Each window score contributes to every token position in its half-open span
  `[start_token, stop_token)`.
- If multiple windows cover a token, their `mg_score` values are averaged.
- The result is a per-token local-evidence series with the same width as the
  token metric arrays.
- Token count mismatches and invalid window bounds raise clear `ValueError`s.
- Tokens without any covering window remain `NaN`.

## Visualization

Add `plot_normalized_token_metrics_with_local_evidence(...)` with these
properties:

- Plot all supplied token metrics and the aligned local `mg_score` on one
  shared axis.
- Min-max normalize each series independently to `[0, 1]` before plotting.
- Constant finite series map to a stable value rather than causing division by
  zero; all-`NaN` series remain unavailable.
- Preserve token position on the x-axis.
- Use distinct labels, including `local mg_score`, and a legend.
- Apply the existing rolling smoothing behavior when requested, while keeping
  the normalized raw series visible.
- Show a clear title indicating normalized values and the selected smoothing
  window.

## Notebook Integration

Each custom inspection cell will call the helper after scoring
`custom_windows`, passing:

- the selected text's token-level summary;
- the selected text's scored windows;
- the selected text token count;
- the existing smoothing window.

The combined plot will be displayed alongside the existing timeline and token
heatmap. The custom plot remains inspection evidence and is not presented as a
held-out document prediction.

## Testing

Add focused tests for:

- mean aggregation for overlapping windows;
- correct half-open token-span alignment;
- normalization of finite, constant, and all-`NaN` series;
- invalid bounds and length mismatch errors;
- notebook contract wiring and output-free structure.

Run the focused test module and the full suite with the Agg backend.
