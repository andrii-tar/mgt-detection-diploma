# Focused Presentation Plot Design

## Goal

Add a presentation-focused token plot for each processed text and window length, showing exactly three normalized lines:

- `mean_candidate_similarity`
- `top20_coverage`
- Logistic-regression local evidence `mg_score`

## Design

Add a dedicated helper next to the existing token-metric plotting helpers. It accepts the validated processed-text token summary, scored local-evidence windows, token count, and the existing smoothing-window parameter. The helper filters/labels the local evidence as the configured logistic-regression model and expands overlapping window scores to token positions using the existing utility.

Each of the three series is independently normalized to `[0, 1]`; the two processed-text metrics use the existing rolling-mean smoothing, while local evidence uses the existing token expansion and normalization behavior. The helper returns/displays one shared-axis Matplotlib figure with token position on the x-axis, normalized value on the y-axis, a concise title, and exactly three legend entries.

The notebook’s current visualization cell calls this helper once per selected text and `WINDOW_LENGTHS`, alongside the existing visualizations. The current broad overlay remains unchanged. The local evidence input is restricted to `LOCAL_EVIDENCE_MODEL`, which is the notebook’s logistic-regression model.

## Validation

- Unit tests verify the helper emits exactly three plotted lines and rejects missing or incorrectly sized metric arrays.
- Notebook contract tests verify the current visualization cell calls the focused helper with the processed summary and LR-filtered windows.
- Existing plotting and notebook tests remain passing.

## Scope

No changes to feature computation, score aggregation, model fitting, processed data, or existing plot semantics.
