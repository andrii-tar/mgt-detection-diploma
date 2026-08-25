# Task 2 Implementation Report

## Scope

Implemented the combined normalized local evidence overlay without changing notebook integration.

## Changes

- Added `plot_normalized_token_metrics_with_local_evidence` in `notebooks/token_candidate_similarity_heatmaps_utils.py`.
  - Expands overlapping `mg_score` windows to token positions through the approved Task 1 helper.
  - Validates that every summary series is one-dimensional and has exactly `token_count` values.
  - Computes and independently normalizes each raw series, rolling series, and the aligned local evidence series.
  - Draws all series on one shared axis and labels the evidence line `local mg_score`.
  - Sets the normalized y-axis range to `[0, 1]` and includes the smoothing window in the title.
- Added focused plotting and validation tests in `tests/test_notebook_utils.py`.
  - Verifies one shared axis, raw and rolling lines, evidence labeling, and normalized finite values.
  - Verifies summary length validation, invalid-window errors, and all-NaN handling.

## TDD Evidence

- Red: `uv run pytest tests/test_notebook_utils.py -k "normalized_token_metrics" -q` failed with four expected `AttributeError` failures because the plotting helper did not exist.
- Green: `uv run pytest tests/test_notebook_utils.py -k "normalized_token_metrics or expand_window_scores or normalize_series" -q` passed after the minimal implementation.

## Verification

- `uv run pytest tests/test_notebook_utils.py -k "normalized_token_metrics or expand_window_scores or normalize_series" -q`: 19 passed, 60 deselected.
- `git diff --check`: passed.
- Confirmed staged changes were limited to the utility and test files specified by the brief.
- Notebook files were not modified.

## Self-Review

- Alignment and normalization reuse the approved Task 1 helpers rather than duplicating their behavior.
- Raw and rolling metric lines are plotted on the same axis as the normalized local evidence line.
- Non-finite values remain non-finite and do not cause plotting to raise.
- Validation occurs before figure creation for malformed summary lengths.

## Commit

- `88cef52cab663b04dd104637812da038b5d84780` (`feat: plot normalized local evidence overlay`)

## Concerns

- None identified for the Task 2 requirements.
- Notebook integration remains intentionally deferred to the next task.
