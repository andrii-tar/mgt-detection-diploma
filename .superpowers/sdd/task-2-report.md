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
