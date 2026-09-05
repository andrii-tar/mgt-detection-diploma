# Historical Report (restored from `3e91f00`)

## Scope

Implemented reusable alignment and normalization helpers for the normalized local evidence overlay.

## Changes

- Added `_expand_window_scores_to_tokens` with required-column, span, token-bound, finite-score, overlap, and uncovered-token handling.
- Added `_normalize_series` with independent finite-value scaling to `[0, 1]`, constant-value handling, and non-finite preservation.
- Added focused helper tests for overlap averaging, invalid inputs, normalization, constant values, all-NaN input, and review edge cases.

## TDD Evidence

- Red: focused tests failed during collection because the requested helpers did not exist.
- Green: focused tests passed after the minimal implementation.

## Verification

- `uv run pytest tests/test_notebook_utils.py -k "expand_window_scores or normalize_series" -q`: 15 passed.
- `uv run pytest tests/test_notebook_utils.py -q`: the helper tests passed; one plotting test was blocked by the environment's Tk/Matplotlib installation.
- `git diff --check`: passed.

## Self-Review

Confirmed half-open boundary behavior, range validation, uncovered-token `NaN` results, and preservation of non-finite normalization values.

## Commit

- `8c1f4bdd4084506a6d18f761a5bbcdd3638c567f` (`feat: add normalized overlay helpers`)

## Concerns

- Broader plotting tests were affected by the pre-existing Matplotlib/Tk environment issue.

## Review Fix

- Coerced `mg_score` with `pd.to_numeric(errors="coerce")` before finite checks.
- Added coverage for nullable and non-numeric scores, negative token counts, uncovered positions, and all-invalid window scores.

## Fix Verification

- Focused helper tests passed: 15 passed.
- Full notebook tests had an existing Matplotlib `tk.tcl` environment failure.

## Task 1 Addendum

### Scope

- Added the Bayesian optional dependency extra and local evidence model adapter utilities.
- Added focused adapter contract tests.

### Adapter Contract

- Logistic regression uses training median imputation, scaling, class balancing, `max_iter=1000`, and the supplied seed.
- GaussianNB uses training-learned median imputation without scaling or non-negative transformations.
- Deterministic adapters preserve the common output schema and bounded scores.
- PyMC import and Bayesian fitting were deferred through a lazy optional path.

### TDD Evidence

The adapter tests initially failed during collection because the adapter module did not exist, then passed after the minimal implementation.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py tests/test_classification_utils.py -q`: 42 passed in the clean optional-dependency environment.
- `uv lock`: resolved 178 packages.

### Concerns

- Bayesian fitting and posterior uncertainty were intentionally deferred to the next task.
- The local environment had transient Matplotlib and Windows file-lock issues during optional-extra setup.

## Task 1 Review-Fix Addendum

### Fixes

- Re-ran `uv lock` and verified the Bayesian extra.
- Added GaussianNB schema/bounded-score, lazy-PyMC, and actionable missing-dependency coverage.
- Added `_load_pymc` to keep optional imports lazy.

### Review-Fix TDD Evidence

- Red: lazy-PyMC tests failed because fitting raised immediately and `_load_pymc` was absent.
- Green: the adapter suite passed after the lazy loader and deferred error were added.

### Review-Fix Verification

- `uv lock`: resolved 178 packages.
- Bayesian and notebook extras synchronized successfully after retry.
- Adapter and classification tests passed in the clean temporary Matplotlib environment.

### Review-Fix Concerns

- Bayesian fitting and posterior uncertainty remained intentionally deferred to Task 2.

## Task 2 Addendum: GaussianNB and PyMC ADVI

### Scope

Completed the GaussianNB adapter contract and implemented the lazy PyMC Bayesian logistic adapter with training-only preprocessing, ADVI fitting, posterior draws, credible intervals, and coefficient summaries.

### Implementation

- GaussianNB retained median-only preprocessing and signed continuous features.
- PyMC used configurable priors, posterior draws, fixed seeds, posterior mean probabilities, and credible intervals.
- Deterministic uncertainty columns remained missing.

### TDD Evidence

The suite initially failed because `summarize_pymc_coefficients` was unavailable, then passed after implementation.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: 8 passed.
- PyMC smoke tests ran with low ADVI and posterior-draw settings.

### Concerns

- Shared comparison evaluation and notebook work remained deferred to Task 3.

## Task 2 Review-Fix Addendum

### Changes

- Strengthened GaussianNB estimator and preprocessing-boundary coverage without changing production behavior.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: 9 passed.

### Concerns

- Shared comparison evaluation and notebook work remained deferred to Task 3.

## Task 2 Partition-Boundary Fix

### Changes

- Added optional partition filtering so all adapters fit only rows with `partition == "train"`, rejecting partitioned input without train rows.
- Preserved existing behavior for frames without a partition column.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: 10 passed.

### Concerns

- Shared comparison evaluation and notebook work remained deferred to Task 3.

## Task 2 Review-Fix Follow-Up

### Changes

- Strengthened the combined-frame GaussianNB preprocessing regression test to mutate non-training rows and compare learned statistics and training predictions.

### Verification

- `uv run pytest tests/test_local_evidence_models_utils.py -q`: 9 passed.

### Concerns

- Shared comparison evaluation and notebook work remained deferred to Task 3.

## Task 3 Addendum: Shared Multi-Model Evaluation

### Scope

Added `run_local_evidence_model_comparison` and focused LR/GaussianNB comparison, training-boundary, and validation-threshold tests.

### Implementation

- Validated models, partitioned by parent text, iterated window lengths, and fit only training rows.
- Scored validation/test windows with the common schema.
- Selected thresholds and score directions from validation only.
- Reused shared aggregation, threshold, decision, and metric helpers.
- Produced common coefficient-summary output while omitting GaussianNB coefficients.

### TDD Evidence

Comparison tests initially failed because the function did not exist, then passed after the shared loop was implemented.

### Verification

- Comparison-focused tests passed.
- Combined classification tests were blocked by the existing Matplotlib `_docstring` import error.

### Concerns

- The combined command remained blocked until the installed Matplotlib package is repaired.

## Task 3 Review-Fix Addendum

### Changes

- Strengthened training-boundary label coverage.
- Added PyMC comparison-level interval propagation coverage.
- Reused shared partition, aggregation, threshold, decision, and metric helpers.

### Verification

- Non-comparison adapter tests passed.
- Combined classification tests remained blocked by the existing Matplotlib `_docstring` import error.
- `py_compile` passed.

### Concerns

- Comparison tests requiring `local_evidence_utils` remained blocked by the broken Matplotlib installation.

## Task 4 Addendum: Multi-Model Visualization Utilities

### Scope

Added reusable multi-model token expansion, score heatmap, normalized plot, and deterministic test-text selection helpers with focused tests.

### Implementation

- Expanded overlapping half-open windows independently per model and preserved Bayesian bounds.
- Used a shared `[0, 1]` heatmap scale and retained normalized metric lines.
- Selected only test texts with a fixed seed without mutating the source frame.

### TDD Evidence

Focused collection initially failed because the visualization module did not exist; the module and tests compiled after implementation.

### Verification

- Visualization runtime tests were blocked by the existing Matplotlib `_docstring` import error.
- `py_compile` and `git diff --check` passed.

### Concerns

- Visualization tests remained unavailable until Matplotlib is repaired.

## Task 4 Review-Fix Addendum

### Fixes

- Filtered comparison predictions by `partition == "test"`.
- Normalized PyMC interval bounds using shared central-score limits.
- Replaced hard-coded `95%` interval text with neutral `credible interval` wording.

### Verification

- Runtime tests remained blocked by the existing Matplotlib error.
- `py_compile` and `git diff --check` passed.

### Concerns

- Focused plotting tests remain unavailable until Matplotlib is repaired.

## Task 5 Addendum: Comparison Notebook

### Scope

Added the output-free `notebooks/bayesian_model_comparison.ipynb` workflow and structural contract tests without modifying the existing local-evidence notebook.

### Notebook Contract

- Exposes experiment, model, cache, window, seed, selected-text, Bayesian, model-list, and recomputation settings.
- Loads the cache, creates one parent-text partition, and calls comparison once.
- Includes deterministic models and conditionally includes PyMC with an actionable install hint.
- Displays grouped metrics, selected test-text prediction/error tables, and reused visualizations.
- Leaves code cells output-free and reusable logic in imported modules.

### TDD Evidence

Notebook contract tests initially failed because the notebook did not exist, then passed after it was added and the fixture was corrected.

### Verification

- `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/bayesian_model_comparison.ipynb`: passed.
- Notebook contract tests: 5 passed.
- `git diff --check`: passed.

### Concerns

- Full data/model execution was not run; broader runtime suites retain the existing Matplotlib concern.

# Gaussian Mixture Task 1 Report

## Commit

Implementation and tests: `60bd243bd7425cf985ecff50ea4f798a51cbc105`

## Tests

Command:

```text
uv run pytest tests/test_local_evidence_models_utils.py -q
```

Output:

```text
...................                                                      [100%]
19 passed in 22.04s
```

## Spec Compliance Verdict

COMPLIANT. Added the frozen configuration and registry entry, training-only median imputation, one-dimensional per-feature/per-class Gaussian mixtures, one-component single-row fallback, class-prior scoring with stable softmax, deterministic scores, common schema, NaN uncertainty bounds, both-class validation, and comparison forwarding.

## Quality Verdict

PASS. Tests were written before implementation, the expected RED collection failure was observed, and the named test suite passes with no warnings or errors.

## Concerns

- Scikit-learn rejects a one-row fit even for one component, so the single observation is duplicated solely to satisfy its minimum sample requirement. The fitted estimator still uses one component and the configured regularization.
- Only the named test file was run; the full repository suite was not requested.

## Review Fix

The original Task 1 report history was available and has been preserved above.

### Finding

`GaussianMixtureNBConfig` could raise `TypeError` for a non-numeric `reg_covar` instead of the required clear `ValueError`, and invalid negative/NaN values lacked dedicated regression coverage.

### Resolution

Commit: `8c0f3e20e16626f073dd7c4abaf7708f919a9a1f`

Added explicit numeric validation for `reg_covar` and tests for negative, NaN, and non-numeric values. Existing zero and infinite-value coverage remains in place.

### Verification

Command:

```text
uv run pytest tests/test_local_evidence_models_utils.py -q
```

Output:

```text
......................                                                   [100%]
22 passed in 48.02s
```

Review-fix verdict: COMPLIANT. The concrete code-quality finding is fixed and covered by regression tests.

Review-fix concerns: The original report's concern about one-row GaussianMixture fitting and its limited test scope remain unchanged.

## Current Progress-Bar Test Addendum

### Test File Changed

- `tests/test_postprocess_pipeline.py`: added `test_run_postprocessing_reports_progress_per_dataset_model_pair` and strengthened its recording-bar assertion.

### Focused Red Verification

- Command: `uv run pytest tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair -q`
- Output summary: `1 failed in 0.50s`; failure occurs while monkeypatching `scripts.postprocess_pipeline.tqdm`.

### Expected Failure Reason

The failure is expected because `scripts/postprocess_pipeline.py` has no production `tqdm` integration and therefore exposes no module-level `tqdm` symbol for the test to patch.

### Review-Fix Assertion

The recording bar now stores each update amount, and the assertion requires exactly two updates with unit increments: `progress["bar"].updates == [1, 1]`.
