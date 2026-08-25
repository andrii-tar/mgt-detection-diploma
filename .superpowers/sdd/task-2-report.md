# Task 2 Report

## Status

Complete.

## Files

- `notebooks/local_evidence_classification.ipynb`
- `tests/test_classification_utils.py`
- `.superpowers/sdd/task-2-report.md`

## Summary

- Added a notebook contract requiring independent custom inspection model fitting and scoring.
- Updated the custom cell to load the raw row by index, build requested split/length windows with progress disabled, reject empty windows, fit from the existing training partition, score independently, and select the highest `mg_score` window.
- Labeled and displayed independent timeline evidence and the selected-token heatmap.
- Cleared notebook outputs and execution counts.

## Commands And Outputs

```text
$env:MPLBACKEND='Agg'; uv run pytest -q tests/test_classification_utils.py::test_local_evidence_classification_notebook_supports_custom_local_inspection
1 passed in 3.71s
```

```text
uv run jupyter nbconvert --clear-output --inplace notebooks/local_evidence_classification.ipynb
```

```text
$env:MPLBACKEND='Agg'; uv run pytest -q
162 passed in 56.78s
```

## TDD Evidence

The updated contract was run before implementation and failed because the custom cell did not contain `build_local_evidence_windows`. It passed after the notebook implementation.

## Self-Review

- The custom cell does not reference test-only `scored_windows`.
- Model fitting uses `windows` and `partitions`, so training remains restricted to the existing training partition.
- Custom scoring is applied only after filtering the newly built windows to the requested raw row.
- Notebook JSON is valid and code-cell outputs/counts are cleared.
- Pre-existing changes in `docs/superpowers/specs/2026-08-25-local-evidence-timeline-design.md` and `notebooks/token_candidate_similarity_heatmaps_utils.py` were preserved and excluded from the Task 2 commit.

## Concerns

- The custom window builder still reads all rows in the requested split before filtering to the selected raw row because its public interface does not accept a single text ID.

## Review Fix

- Added optional `text_ids` filtering to `_iter_stage_one_rows` and `build_local_evidence_windows`.
- When `text_ids` is supplied, the builder loads only the matching similarity rows and generates windows only for matching stage-one rows.
- Updated the custom cell to pass `text_ids=(custom_text_id,)`, preventing full-split custom window construction.
- Strengthened the contract with AST assertions for the build, fit, and score calls, the selected text ID argument, progress disabling, and the absence of `scored_windows` in the custom cell.
- The prior concern is resolved for window generation; parquet files are still read to locate the selected row because the current storage layout is file-based.

## Review Fix Commands And Outputs

```text
$env:MPLBACKEND='Agg'; uv run pytest -q tests/test_classification_utils.py::test_local_evidence_classification_notebook_supports_custom_local_inspection
1 passed in 3.91s
```

```text
uv run jupyter nbconvert --clear-output --inplace notebooks/local_evidence_classification.ipynb
[NbConvertApp] Converting notebook notebooks/local_evidence_classification.ipynb to notebook
[NbConvertApp] Writing 9199 bytes to notebooks\local_evidence_classification.ipynb
```

```text
$env:MPLBACKEND='Agg'; uv run pytest -q
........................................................................ [ 44%]
........................................................................ [ 88%]
..................                                                       [100%]
162 passed in 52.67s
```

## Contract Assertion Fix

- Added AST assertions that `fit_local_evidence_model` receives `windows`, `partitions`, and `CUSTOM_WINDOW_LENGTH`.
- Added AST assertions that `score_local_evidence_windows` receives `custom_model`, `custom_windows`, and `custom_features`.
- No notebook or implementation behavior was changed; the notebook remains output-free.

## Contract Assertion Fix Commands And Outputs

```text
uv run pytest tests/test_classification_utils.py -q
.................................                                        [100%]
33 passed in 14.59s
```

```text
$env:MPLBACKEND='Agg'; uv run pytest -q
........................................................................ [ 44%]
........................................................................ [ 88%]
..................                                                       [100%]
162 passed in 50.13s
```
