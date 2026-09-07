# Short Validation Evaluation Design

## Goal

Evaluate classifiers trained on the existing Mixed training dataset against the
original validation dataset and validation variants truncated to the first `N`
words. The purpose is to measure how validation macro-F1 changes as available
text length decreases.

## Scope

- Reuse already-trained whole-text and local-evidence model bundles.
- Do not retrain classifiers for each length.
- Do not recalibrate thresholds for short validation variants.
- Evaluate only validation data; existing train, test, BBC, and cross-dataset
  analyses remain unchanged.
- Make the word limits configurable as a tuple/list supplied by the notebook
  user.

## Truncation Semantics

Each validation text is replaced by its prefix containing at most the first `N`
whitespace-delimited words. Texts shorter than `N` words remain unchanged. The
word boundary is mapped to the corresponding stored Stage 1 token boundary.
The token prefix is used to recompute features, so truncation changes the
actual classifier inputs rather than merely relabeling existing metric rows.

If a reliable word boundary cannot be derived from a row's stored text and
tokens, that row is excluded from the affected short-limit evaluation and
reported in coverage. No row is silently treated as untruncated.

## Feature Evaluation

### Whole Text

Read the existing Stage 1 rows for the Mixed validation cohort. For every word
limit, slice each row's token-level logprob data at the mapped prefix boundary
and call `compute_logprob_metrics` on the prefix. Preserve dataset, model,
split, ID, and label provenance, then evaluate the unchanged trained
`WholeTextClassifierBundle`.

### Local Evidence

Use the paired Stage 1 and Stage 2 token data for the same validation cohort.
Truncate both token-aligned inputs to the selected prefix and rebuild local
windows/features only within that prefix. Reuse the trained local-evidence
model registry and its validation-derived rules. Windows extending beyond the
prefix are not included.

## Results

Produce one long-form comparison table containing:

- scorer LLM
- classifier or evaluation block
- `training_regime` (`Mixed`)
- validation variant (`original` or `first_N_words`)
- word limit (`NaN` for original)
- row/coverage counts
- macro-F1
- accuracy, HG-F1, and MG-F1 for context

Display a macro-F1 plot with word limit on the x-axis. The original validation
score is shown as a horizontal baseline for each model/configuration. Short
validation results are shown separately for each requested word limit.

## Configuration

The notebook exposes a parameter with the user's selected values, for example:

```python
VALIDATION_WORD_LIMITS = (50, 100, 200, 400)
```

The implementation validates that every limit is a positive integer and
preserves the supplied order for result presentation.

## Error Handling and Coverage

- Invalid limits raise `ValueError` before evaluation starts.
- Missing Stage 1/Stage 2 data raises the existing data-loading errors.
- Rows that cannot be mapped to a valid prefix are excluded only from that
  short-limit result and recorded with an exclusion reason.
- Empty short cohorts produce a clear evaluation error rather than a misleading
  metric.
- The original validation result is computed independently and remains the
  comparison baseline.

## Testing

Add tests for:

- positive-limit validation and invalid inputs
- first-`N`-word token-boundary calculation
- unchanged behavior for texts shorter than the limit
- scalar prefix metric recomputation
- local windows not crossing the prefix boundary
- evaluation reusing the supplied trained bundle without fitting
- result metadata and macro-F1 comparison output

The notebook remains output-free after verification.
