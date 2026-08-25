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

## Task 1 Review Fix Addendum

### Fixes

- Re-ran `uv lock`; it resolves the committed project metadata and keeps the `bayesian` extra mapped to `pymc>=5.0`.
- Added GaussianNB common-schema and probability-bound assertions.
- Added lazy-PyMC coverage proving deterministic fitting does not import PyMC, missing PyMC produces an actionable optional-dependency error only on the Bayesian path, and available PyMC still reports the planned Task 2 deferral through the existing public fit API.
- Added `_load_pymc()` so the optional import remains lazy and the deferred limitation is explicit without changing the public adapter signatures.

### Review-Fix TDD Evidence

- Red: the new lazy-PyMC tests failed because the implementation raised immediately and had no `_load_pymc()` hook.
- Green: the adapter test file passed after adding the lazy loader and explicit deferred error.

### Review-Fix Verification

- `uv lock`: `Resolved 178 packages in 1ms`.
- `uv sync --extra bayesian`: succeeded on retry and installed `pymc==6.3.1` plus its optional dependencies. The first attempt was blocked by a Windows file lock on `.venv`'s `DejaVuSans.ttf`.
- `uv sync --extra notebook --extra bayesian`: completed successfully to restore the notebook test extra.
- `uv run pytest tests/test_local_evidence_models_utils.py -q`: `7 passed`.
- `PYTHONPATH=C:\Users\Andrii\AppData\Local\Temp\opencode\matplotlib-review uv run pytest tests/test_local_evidence_models_utils.py tests/test_classification_utils.py -q`: `42 passed in 38.37s`.

### Review-Fix Concerns

- The project `.venv` had a transient Windows file lock and an incomplete matplotlib installation after the first optional-extra sync; the retry succeeded, and the specified tests passed using a clean temporary matplotlib target while project kernels held the locked file.
- Bayesian model fitting and posterior uncertainty remain intentionally deferred to Task 2.

## Task 2 Addendum: GaussianNB and PyMC ADVI

### Scope

- Completed the GaussianNB adapter contract and added explicit signed continuous-feature coverage.
- Implemented the lazy PyMC Bayesian logistic adapter using training-only median imputation and standardization.
- Added ADVI fitting with configurable iterations, fixed seeds, configurable posterior draws, latent-probability intervals, and posterior coefficient summaries.

### Implementation

- GaussianNB continues to use only median imputation and `GaussianNB`; signed values are not transformed or standardized.
- PyMC uses `Normal(0, prior_scale)` feature coefficients, `Normal(0, intercept_scale)` intercept, sigmoid latent MG probability, and Bernoulli labels.
- Bayesian scores use posterior mean latent probabilities and configured credible-interval quantiles; deterministic uncertainty columns remain missing.
- Missing PyMC raises an actionable error naming `uv sync --extra bayesian`, while deterministic models retain lazy-import independence.

### TDD Evidence

- Red: the new suite failed during collection because `summarize_pymc_coefficients` was not yet available.
- Green: after implementation and updating the optional-dependency expectation, the specified adapter suite passed.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: `8 passed in 14.34s`.
- PyMC smoke tests ran, not skipped, using `advi_iterations=50` and `posterior_draws=40`.

### Concerns

- None identified for Task 2. Shared comparison evaluation and notebook work remain intentionally deferred to Task 3.

## Task 2 Review-Fix Addendum

### Changes

- Strengthened GaussianNB coverage to assert the fitted estimator type and confirm its pipeline contains median imputation plus GaussianNB only, without scaling or non-negative transformation.
- Added a deterministic preprocessing-boundary regression test that mutates separate non-training rows and confirms fitted training imputation statistics and transformed training values remain unchanged.
- Production behavior was not modified.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: `9 passed in 23.89s`.
- PyMC smoke tests ran, not skipped.

### Concerns

- None identified. Shared comparison evaluation and notebook work remain intentionally deferred to Task 3.

## Task 2 Partition-Boundary Fix

### Changes

- Added an optional `partition` boundary to `fit_local_evidence_classifier`; partitioned frames now fit only rows where `partition == "train"` and reject frames without train rows.
- Applied the same filtering to deterministic and PyMC preprocessing/model fitting while preserving existing behavior for train-only frames without a partition column.
- Rewrote GaussianNB leakage coverage to pass full partitioned fixtures and mutate non-training feature values; PyMC coverage also fits a full partitioned fixture.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: `10 passed in 5.43s`.
- PyMC smoke tests ran, not skipped.

### Concerns

- None identified. Shared comparison evaluation and notebook work remain intentionally deferred to Task 3.

## Task 2 Review-Fix Follow-Up

### Changes

- Reworked the GaussianNB preprocessing regression test to mutate validation/test rows in a combined dataframe before passing its selected training rows into separate original and mutated fits.
- The test now compares learned imputer statistics and training-row predictions between both fits, while production code remains unchanged.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: `9 passed in 7.96s`.
- PyMC smoke tests ran, not skipped.

### Concerns

- None identified. Shared comparison evaluation and notebook work remain intentionally deferred to Task 3.

## Task 3 Addendum: Shared Multi-Model Evaluation

### Scope

- Added `run_local_evidence_model_comparison` to `notebooks/local_evidence_models_utils.py`.
- Added focused LR/GaussianNB comparison, training-boundary, and validation-threshold regression tests.
- No visualization utilities or comparison notebook were added.

### Implementation

- The comparison loop validates requested model names, partitions windows by parent text, iterates sorted window lengths, and fits each adapter only on training rows.
- Validation and test windows are scored independently with model identity and bounded probability columns preserved.
- Maximum-window and evidence-fraction document aggregations use thresholds and score directions selected from validation predictions only.
- Summary rows include model, window length, aggregation, thresholds, score direction, validation macro F1, support counts, accuracy, macro F1, and the remaining evaluation metrics.
- Prediction rows include model identity, raw document score, prediction, document threshold, and window threshold.
- Logistic-regression coefficient rows and PyMC posterior coefficient rows share a coefficient-summary schema; GaussianNB contributes no coefficient rows.
- The adapter module keeps its import path independent of Matplotlib because the existing local-evidence module imports plotting dependencies at module import time.

### TDD Evidence

- Red: comparison tests initially failed during collection because `run_local_evidence_model_comparison` did not exist.
- Green: comparison tests passed after the minimal shared loop and output contract were implemented.

### Verification

- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py -k comparison -q`: `3 passed, 10 deselected`.
- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py -q`: `13 passed`.
- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py tests/test_classification_utils.py -q`: blocked during test collection by the existing environment error `ImportError: cannot import name '_docstring' from 'matplotlib'` while importing `local_evidence_utils.py`.

### Concerns

- The required combined command cannot complete until the installed Matplotlib package is repaired; the Task 3 adapter/comparison suite passes independently.

## Task 3 Review-Fix Addendum

### Changes

- Strengthened `test_comparison_fits_only_training_parent_windows` to mutate non-training labels and compare the resulting scored-window model probabilities, proving the fitted outputs remain unchanged without relying only on coefficient summaries.
- Added a PyMC comparison-level interval test with low ADVI and posterior-draw settings. It checks that `mg_score_lower` and `mg_score_upper` propagate through both `maximum_window` and `evidence_fraction` document predictions with valid ordering, and skips only when PyMC is unavailable.
- Removed duplicated comparison threshold, score-direction, aggregation, partition, and metric logic. The comparison loop now reuses `_partition_windows`, `aggregate_maximum_window`, `aggregate_evidence_fraction`, `_fit_document_threshold`, `_select_evidence_parameters`, `_decision_scores`, and `evaluate_scores` from `notebooks.local_evidence_utils`.
- Existing Matplotlib import behavior was not changed.

### Verification

- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py -k "not comparison" -q`: `10 passed, 4 deselected`.
- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py tests/test_classification_utils.py -q`: blocked during collection with the exact existing error `ImportError: cannot import name '_docstring' from 'matplotlib'` from `matplotlib.mlab` while importing `notebooks.local_evidence_utils`.
- `uv run python -m py_compile notebooks/local_evidence_models_utils.py tests/test_local_evidence_models_utils.py`: passed.

### Concerns

- The four comparison tests and the combined classification suite remain blocked by the pre-existing broken Matplotlib installation; no code workaround was introduced.

## Task 4 Addendum: Multi-Model Visualization Utilities

### Scope

- Added `notebooks/local_evidence_comparison_utils.py` with reusable multi-model token expansion, score heatmap, combined normalized plot, and deterministic test-text selection helpers.
- Added focused tests in `tests/test_local_evidence_comparison_utils.py`.
- No existing plotting helper or notebook was modified; the comparison notebook remains intentionally deferred.

### Implementation

- `expand_model_scores_to_tokens` expands overlapping half-open windows independently for each model, averages finite overlapping scores deterministically, and returns one row per model/token position. Bayesian lower and upper interval columns are preserved when present.
- `plot_model_score_heatmap` uses model rows, token-position columns, and a shared `[0, 1]` color scale.
- `plot_normalized_token_metrics_with_model_evidence` reuses the established rolling and normalization helpers, retains semi-transparent normalized metric lines, adds one normalized local-evidence line per model, places the legend below the plot, and draws interval bands only where finite bounds are available.
- `select_comparison_texts` samples only `split == "test"` rows with a fixed random seed without mutating the source frame.

### TDD Evidence

- Red: the focused test collection failed because `notebooks.local_evidence_comparison_utils` did not exist.
- Green implementation validation was limited by the pre-existing Matplotlib installation error described below; the new module and tests compile successfully.

### Verification

- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_comparison_utils.py tests/test_notebook_utils.py -q`: blocked during collection with the exact existing error `ImportError: cannot import name '_docstring' from 'matplotlib'` from `matplotlib.mlab` while importing both test modules.
- `$env:MPLBACKEND='Agg'; uv run pytest tests/test_local_evidence_models_utils.py -q`: `10 passed, 4 failed`; all four failures were existing comparison tests blocked when they lazily imported `notebooks.local_evidence_utils`, which raises the same Matplotlib `_docstring` error.
- `uv run python -m py_compile notebooks/local_evidence_comparison_utils.py tests/test_local_evidence_comparison_utils.py`: passed.
- `git diff --check`: passed.

### Commit

- Task 4 implementation commit is the commit containing this report append and the two new files.

### Concerns

- The requested visualization tests cannot collect until the environment's installed Matplotlib package is repaired. No dependency version or existing plotting behavior was changed as a workaround.
