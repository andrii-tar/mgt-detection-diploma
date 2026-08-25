# Task 3 Report: Custom Notebook Overlay Integration

## Status

Implemented and verified.

## Changes

- Updated `notebooks/local_evidence_classification.ipynb` imports with the normalized local-evidence plotting helper and selected-text similarity summary helpers.
- Added `SMOOTHING_WINDOW = 10` to the notebook parameters.
- Added the normalized token-metric/local-evidence plot to all four existing custom inspection cells.
- Each custom cell loads the similarity row using `custom_text_id` and `CUSTOM_SPLIT`, validates the selected text summary, and plots the scored `custom_windows` with the configured smoothing window.
- Preserved the existing timeline and token-similarity inspection calls.
- Preserved the notebook's nine code cells, with every code cell output-free and unexecuted.
- Extended `tests/test_classification_utils.py` to verify overlay wiring in all four custom cells.
- Updated the notebook contract's code-cell count from six to the notebook's current nine-cell structure.

## Verification

The required red-phase contract run failed because the four custom cells lacked the new overlay wiring. After implementation:

```text
uv run pytest tests/test_classification_utils.py -k "notebook" -q
4 passed, 29 deselected
```

Full focused module:

```text
uv run pytest tests/test_classification_utils.py -q
33 passed
```

Additional self-review:

- `git diff --check` passed.
- Notebook JSON and output-free assertions passed.
- No existing cells were removed or execution outputs introduced.

## Commits

- `adfb9b649a15041624e6ea3ad4098d1ed676612e` - `feat: show local evidence with token metrics`

## Task 3 Review Follow-Up

Strengthened `test_local_evidence_classification_notebook_supports_custom_local_inspection` so every custom inspection cell is parsed independently and must:

- call `plot_normalized_token_metrics_with_local_evidence` with `custom_summary`, `custom_windows`, the selected token-label count, and `window=SMOOTHING_WINDOW`;
- load the similarity row with `CUSTOM_SPLIT` and `custom_text_id`;
- validate the summary from `custom_row` and `custom_similarity_row`;
- build windows with `splits=(CUSTOM_SPLIT,)` and `text_ids=(custom_text_id,)`;
- avoid `scored_windows` references.

The notebook implementation was unchanged.

Exact verification commands and output:

```text
uv run pytest tests/test_classification_utils.py -k "notebook" -q
....                                                                     [100%]
4 passed, 29 deselected in 16.05s
```

```text
$env:MPLBACKEND="Agg"; uv run pytest tests/test_notebook_utils.py -q
........................................................................ [ 91%]
.......                                                                  [100%]
79 passed in 4.94s
```
