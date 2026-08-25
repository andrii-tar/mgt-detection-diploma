# Task 1 Implementation Report

## Scope

Implemented the reusable alignment and normalization helpers required for the normalized local evidence overlay.

## Changes

- Added `_expand_window_scores_to_tokens` in `notebooks/token_candidate_similarity_heatmaps_utils.py`.
  - Validates required columns, positive token count, half-open spans, and token bounds.
  - Averages finite `mg_score` values over overlapping token spans.
  - Returns `NaN` for tokens without finite window coverage.
- Added `_normalize_series` in `notebooks/token_candidate_similarity_heatmaps_utils.py`.
  - Scales finite values independently to `[0, 1]`.
  - Maps constant finite values to zero.
  - Preserves non-finite values.
- Added focused tests in `tests/test_notebook_utils.py` for overlap averaging, invalid inputs, normalization, constant values, and all-NaN input.

## TDD Evidence

- Red: the focused test run failed during collection because the requested helpers did not exist.
- Green: the focused test run passed after the minimal implementation was added.

## Verification

- `uv run pytest tests/test_notebook_utils.py -k "expand_window_scores or normalize_series" -q`: 11 passed.
- `uv run pytest tests/test_notebook_utils.py -q`: 71 passed.

## Self-Review

- Confirmed changes are limited to the requested utility and test files for the implementation commit.
- Confirmed half-open boundary behavior through the overlap test and range validation cases.
- Confirmed uncovered tokens and non-finite normalization values remain `NaN`.

## Commit

- `8c1f4bdd4084506a6d18f761a5bbcdd3638c567f` (`feat: add normalized overlay helpers`)

## Concerns

- None identified for the Task 1 requirements.

## Review Fix

- Fixed `_expand_window_scores_to_tokens` to coerce `mg_score` values with `pd.to_numeric(errors="coerce")` before finite checks.
- Nullable and non-numeric scores are now treated as missing and ignored, while finite numeric scores continue to aggregate normally.
- Added regression coverage for mixed finite, `pd.NA`, and non-numeric scores in `tests/test_notebook_utils.py`.

## Fix Verification

- `uv run pytest tests/test_notebook_utils.py -k "expand_window_scores or normalize_series" -q`: 12 passed, 60 deselected in 21.23s.
- `uv run pytest tests/test_notebook_utils.py -q`: 72 passed in 27.23s.

## Final Verification Rerun

- `uv run pytest tests/test_notebook_utils.py -k "expand_window_scores or normalize_series" -q`: 12 passed, 60 deselected in 4.73s.
- `uv run pytest tests/test_notebook_utils.py -q`: 1 failed, 71 passed in 8.62s. The existing plotting test failed because the environment's Tk installation could not load `tk.tcl`; the focused helper tests passed.

## Review Gap Tests

- Added explicit coverage for negative `token_count`.
- Added explicit coverage for uncovered token positions returning `NaN`.
- Added explicit coverage for all-invalid window scores returning all `NaN`.

## Review Gap Verification

Exact output from `uv run pytest tests/test_notebook_utils.py -k "expand_window_scores or normalize_series" -q`:

```text
...............                                                          [100%]
15 passed, 60 deselected in 6.03s
```

## Task 1 Addendum

### Scope

- Added the `bayesian` optional dependency extra with `pymc>=5.0`.
- Regenerated `uv.lock` with `uv lock`.
- Added `notebooks/local_evidence_models_utils.py` with stable model identifiers, explicit `BayesianConfig` defaults, and deterministic adapter fit/score functions.
- Added focused adapter contract tests.

### Adapter Contract

- Logistic regression uses median imputation with `keep_empty_features=True`, `StandardScaler`, class-balanced `LogisticRegression`, `max_iter=1000`, and the supplied seed.
- GaussianNB uses training-learned median imputation and `GaussianNB` without scaling or non-negative feature transformations.
- Both deterministic adapters preserve input window columns and add `model`, bounded `mg_score`, and missing `mg_score_lower`/`mg_score_upper` uncertainty columns.
- PyMC is not imported at module import time. Bayesian fitting remains explicitly deferred to the next task.

### TDD Evidence

The new test initially failed during collection with `ModuleNotFoundError` because the adapter module did not exist. After the minimal implementation was added, the focused adapter tests passed.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py tests/test_classification_utils.py -q`: `39 passed in 27.23s`.
- `uv lock`: `Resolved 178 packages in 1ms`.

### Concerns

- The repository ignores `uv.lock`; it was force-added because the brief explicitly requires the lockfile update to be committed.
- Bayesian fitting and posterior uncertainty are intentionally not implemented in Task 1.
