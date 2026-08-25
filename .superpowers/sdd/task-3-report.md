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
