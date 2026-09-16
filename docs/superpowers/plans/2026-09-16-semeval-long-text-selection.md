# SemEval Long-Text Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict SemEval visualization selection to texts with at least 200 tokens before ranking and random sampling.

**Architecture:** Extend the reusable selector with an optional set of eligible text IDs. The analysis notebook derives that set from its filtered persisted boundary predictions, where `token_count` is authoritative, and passes it into the selector. Existing behavior remains unchanged when no eligibility set is supplied.

**Tech Stack:** Python 3.12+, pandas, NumPy, pytest, Jupyter notebooks.

## Global Constraints

- Use `token_count` from persisted `semeval_boundary_predictions.csv`.
- Apply the `token_count >= 200` filter before best/worst ranking and random sampling.
- Preserve five best, five worst, and fixed-seed random selection behavior.
- Preserve output-free notebook requirements.

---

### Task 1: Add Selector Eligibility Tests

**Files:**
- Modify: `tests/test_semeval_text_comparison_utils.py`
- Modify: `tests/test_feature_vector_classification_notebooks.py`

**Interfaces:**
- Test `select_semeval_comparison_texts(..., eligible_text_ids=...)`.
- Test notebook source for deriving `eligible_text_ids` from `token_count >= 200` and passing it to the selector.

- [ ] **Step 1: Add a failing utility test**

Create an evaluation fixture with IDs below and above the threshold, call the selector with only the long IDs, and assert every selected row belongs to the eligible set.

- [ ] **Step 2: Add a failing notebook contract assertion**

Assert the analysis notebook contains `token_count >= 200`, constructs an eligible text-ID collection from boundary rows, and passes `eligible_text_ids` to `select_semeval_comparison_texts`.

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```powershell
uv run pytest -q tests/test_semeval_text_comparison_utils.py tests/test_feature_vector_classification_notebooks.py -k "semeval or analysis_notebook"
```

Expected: FAIL because the selector has no `eligible_text_ids` parameter and the notebook does not pass one.

### Task 2: Implement Eligibility Filtering

**Files:**
- Modify: `notebooks/utils/semeval_text_comparison_utils.py`

**Interfaces:**
- Extend `select_semeval_comparison_texts` with `eligible_text_ids: Collection[int] | None = None`.
- Preserve the existing return schema and all behavior when the argument is `None`.

- [ ] **Step 1: Add the collection type import and keyword parameter**

Use `collections.abc.Collection` and add the optional keyword after the existing selection parameters.

- [ ] **Step 2: Filter IDs before pivoting**

After `filter_semeval_configuration`, restrict `filtered` to `text_id` values contained in `eligible_text_ids`. Normalize IDs numerically or as strings consistently so CSV-loaded integer and string IDs match.

- [ ] **Step 3: Run the utility test**

Run:

```powershell
uv run pytest -q tests/test_semeval_text_comparison_utils.py::test_selects_only_eligible_long_texts
```

Expected: PASS.

### Task 3: Wire the Notebook Filter

**Files:**
- Modify: `notebooks/analyze_classifier_results.ipynb`

**Interfaces:**
- Derive `eligible_text_ids` from the already run-filtered `semeval_boundaries` table.
- Pass `eligible_text_ids=eligible_text_ids` to `select_semeval_comparison_texts`.

- [ ] **Step 1: Derive eligible IDs**

Add:

```python
eligible_text_ids = set(
    semeval_boundaries.loc[semeval_boundaries["token_count"].ge(200), "text_id"].dropna()
)
```

- [ ] **Step 2: Pass the IDs into selection**

Add `eligible_text_ids=eligible_text_ids` to the existing selector call without changing group count or seed.

- [ ] **Step 3: Keep the notebook output-free**

Ensure code cells retain `execution_count: null` and `outputs: []`.

- [ ] **Step 4: Run notebook tests and JSON validation**

Run:

```powershell
uv run pytest -q tests/test_feature_vector_classification_notebooks.py -k analysis_notebook
uv run python -c "import json; json.load(open('notebooks/analyze_classifier_results.ipynb', encoding='utf-8')); print('notebook JSON valid')"
```

Expected: all selected notebook tests pass and JSON validation prints `notebook JSON valid`.

### Task 4: Full Verification

**Files:**
- Verify: `notebooks/utils/semeval_text_comparison_utils.py`
- Verify: `notebooks/analyze_classifier_results.ipynb`
- Verify: `tests/test_semeval_text_comparison_utils.py`
- Verify: `tests/test_feature_vector_classification_notebooks.py`

- [ ] **Step 1: Run the full test suite**

Run:

```powershell
uv run pytest
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```powershell
uv run python -m compileall -q notebooks/utils tests
```

Expected: both commands complete without errors.

- [ ] **Step 3: Inspect the final diff**

Confirm only the selector, its tests, and the analysis notebook changed; confirm no generated data or notebook outputs were added.
