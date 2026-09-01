# Local Evidence Coefficient Reporting

## Goal

Print the trained local-evidence model weights so that each input feature and
each of the four base token metrics can be inspected after fitting.

## Scope

- Report coefficients immediately after fitting each window-length model.
- Keep model fitting, scoring, thresholds, predictions, and persisted outputs
  unchanged.
- Treat the 36 input features as four groups of nine statistics:
  `observed_logprob`, `mean_candidate_logprob_delta`,
  `mean_candidate_similarity`, and `top20_coverage`.

## Output

For every `window_length`, print:

1. Every feature name and its signed logistic-regression coefficient.
2. One summary row per base metric containing:
   - mean signed coefficient;
   - mean absolute coefficient;
   - sum of absolute coefficients;
   - largest absolute coefficient and its statistic name.

Positive coefficients increase the MG logit and negative coefficients decrease
it. Coefficients are based on the pipeline's standardized features, so they can
be compared across input features within a fitted model.

## Implementation

Read coefficients from the fitted pipeline's `model.coef_[0]` and pair them
with the ordered `feature_columns` returned by model fitting. Group features by
the text before the `__` separator. Reporting is diagnostic output only and
must not alter the fitted pipeline or downstream data frames.

## Testing

Add focused tests that capture stdout and verify that each window length emits
all feature coefficients and all four grouped summaries, including signed and
absolute aggregate values. Run the focused tests and the full suite with the
Agg backend.
