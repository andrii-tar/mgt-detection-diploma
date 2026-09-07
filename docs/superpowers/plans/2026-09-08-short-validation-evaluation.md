# Short Validation Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate Mixed-trained statistical whole-text and local-evidence aggregate-MG-score classifiers on original and first-`N`-word validation prefixes, then compare validation macro-F1.

**Architecture:** Build one deterministic word-to-token prefix utility over stored Stage 1 rows. Use it to derive truncated scalar features and truncated local-evidence windows without retraining or threshold recalibration. Add a shared long-form result schema consumed by the existing classification notebook.

**Tech Stack:** Python 3.12+, pandas, NumPy, scikit-learn, PyArrow Parquet, pytest, Jupyter notebook utilities.

## Global Constraints

- Evaluate both classifier families for every requested short-validation limit: statistical whole-text classifiers using scalar features, and local-evidence classifiers using aggregated `mg_score` features.
- Do not retrain classifiers for each length.
- Do not recalibrate thresholds for short validation variants.
- Use the first `N` whitespace-delimited words; texts shorter than `N` words remain unchanged.
- Exclude rows that cannot be mapped to a reliable prefix and record an exclusion reason.
- Evaluate only validation data; preserve existing train, test, BBC, and cross-dataset analyses.
- Keep the notebook output-free after verification.
- Do not run model-processing jobs or commit external data/artifacts.

---

### Task 1: Build Word-Prefix Token Utilities

**Files:**
- Modify: `notebooks/logprob_metrics_distributions_utils.py`
- Test: `tests/test_notebook_utils.py`

**Interfaces:**
- Produces `validate_word_limits(word_limits: Sequence[int]) -> tuple[int, ...]`.
- Produces `token_prefix_length_for_words(text: str, tokens: Sequence[Mapping[str, object]], word_limit: int) -> int`.
- Produces `truncate_logprob_row_by_words(row: Mapping[str, object], word_limit: int) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for limit validation and token boundaries**

Add tests using token strings with leading spaces and punctuation. Assert that a
three-word prefix includes exactly the tokens needed to reconstruct the first
three whitespace-delimited words, that a limit larger than the text returns all
tokens, and that zero, negative, boolean, and non-integer limits raise
`ValueError`.

```python
def test_token_prefix_length_for_words_stops_at_requested_word() -> None:
    tokens = [{"token": token} for token in ("One", " two", " three", " four")]
    assert token_prefix_length_for_words("One two three four", tokens, 3) == 3


def test_token_prefix_keeps_short_text_unchanged() -> None:
    tokens = [{"token": token} for token in ("One", " two")]
    assert token_prefix_length_for_words("One two", tokens, 10) == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_validate_word_limits_rejects_invalid_values(limit: object) -> None:
    with pytest.raises(ValueError):
        validate_word_limits((limit,))
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
uv run pytest tests/test_notebook_utils.py -k "word_limit or word_prefix" -q
```

Expected: FAIL because the three new functions do not exist.

- [ ] **Step 3: Implement deterministic word-prefix slicing**

Count words with `text.split()`. Reconstruct the token prefix using the stored
token strings and require that its normalized whitespace-separated words match
the requested text prefix. Return the full token count for texts at or below
the limit. Raise `ValueError` when token strings are absent, token text cannot
reconstruct the source prefix, or the prefix contains no valid logprob token.
Copy the row and replace only its `tokens` list and `text` prefix; preserve ID,
split, and all provenance fields.

```python
def truncate_logprob_row_by_words(row, word_limit):
    tokens = list(row.get("tokens", []))
    text = str(row.get("text") or "")
    end = token_prefix_length_for_words(text, tokens, word_limit)
    prefix = dict(row)
    prefix["tokens"] = tokens[:end]
    prefix["text"] = " ".join(text.split()[:word_limit])
    return prefix
```

- [ ] **Step 4: Run the focused tests and verify success**

Run the same focused command. Expected: all selected tests PASS, including the
existing prefix-metric tests.

- [ ] **Step 5: Commit the utility and tests**

```powershell
git add notebooks/logprob_metrics_distributions_utils.py tests/test_notebook_utils.py
git commit -m "feat: add word-limited token prefixes"
```

### Task 2: Evaluate Short Whole-Text Validation Cohorts

**Files:**
- Modify: `notebooks/whole_text_classification_utils.py`
- Test: `tests/test_whole_text_classification_utils.py`

**Interfaces:**
- Produces `evaluate_whole_text_validation_lengths(bundle: WholeTextClassifierBundle, validation_rows: Mapping[str, pd.DataFrame], stage1_rows: Mapping[str, Sequence[Mapping[str, object]]], word_limits: Sequence[int]) -> WholeTextEvaluationResult`.
- The returned predictions and metrics include `validation_variant` and `word_limit`.

- [ ] **Step 1: Write failing tests for original and short variants**

Construct a small trained bundle with a spy pipeline and Stage 1 rows whose
prefix metrics differ from full-text metrics. Assert that the function emits
one original result plus one result for each requested limit, preserves labels
and IDs, and calls prediction on the same supplied pipeline without fitting.

```python
def test_short_validation_reuses_bundle_and_adds_length_metadata():
    result = evaluate_whole_text_validation_lengths(bundle, cohorts, stage1_rows, (2, 4))
    assert set(result.metrics["validation_variant"]) == {"original", "first_N_words"}
    assert set(result.metrics.loc[result.metrics["validation_variant"] == "first_N_words", "word_limit"]) == {2, 4}
    assert result.predictions["word_limit"].notna().any()
    assert spy_pipeline.fit_calls == 0
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
uv run pytest tests/test_whole_text_classification_utils.py -k "validation_lengths" -q
```

Expected: FAIL because the evaluator does not exist.

- [ ] **Step 3: Implement short scalar-cohort evaluation**

Validate limits once, evaluate the original validation cohort through the
existing `evaluate_whole_text_classifiers`, and build each short cohort by
calling `truncate_logprob_row_by_words` followed by `compute_logprob_metrics`.
Join the recomputed features with the original provenance and labels, then call
`evaluate_whole_text_classifiers` with the unchanged bundle. Add variant and
limit metadata to metrics, predictions, and coverage. Record excluded IDs and
reasons in coverage using the existing coverage columns.

- [ ] **Step 4: Run focused and existing whole-text tests**

Run:

```powershell
uv run pytest tests/test_whole_text_classification_utils.py -q
```

Expected: all whole-text tests PASS.

- [ ] **Step 5: Commit the whole-text evaluator**

```powershell
git add notebooks/whole_text_classification_utils.py tests/test_whole_text_classification_utils.py
git commit -m "feat: evaluate whole-text models on short validation"
```

### Task 3: Evaluate Short Local-Evidence Validation Cohorts

**Files:**
- Modify: `notebooks/local_evidence_utils.py` and `notebooks/local_evidence_evaluation_utils.py`
- Test: `tests/test_local_evidence_evaluation_utils.py` and `tests/test_local_evidence_models_utils.py`

**Interfaces:**
- Produces `limit_local_evidence_windows_by_words(windows: pd.DataFrame, stage1_rows: Mapping[tuple[str, str], Mapping[str, object]], word_limit: int) -> pd.DataFrame`.
- Produces `evaluate_local_evidence_validation_lengths(model: LocalEvidenceModelBundle, processed_root: Path | str, datasets: Sequence[str], validation_text_ids: Mapping[str, pd.DataFrame], stage1_rows: Mapping[str, Sequence[Mapping[str, object]]], word_limits: Sequence[int], show_progress: bool = False) -> LocalEvidenceDatasetEvaluationResult` using the existing aggregate-MG-score model registry and rules.

- [ ] **Step 1: Write failing tests for prefix-window boundaries**

Use synthetic Stage 1 rows with six tokens and a window-length-two local cache.
Assert that a short prefix produces only windows with
`stop_token <= prefix_token_count`, preserves the original text ID and label,
and does not include a window that crosses the prefix.

```python
def test_short_local_windows_do_not_cross_word_prefix():
    windows = limit_local_evidence_windows_by_words(windows, stage1_rows, 3)
    assert windows["stop_token"].le(3).all()
```

- [ ] **Step 2: Run local-evidence focused tests and verify failure**

Run:

```powershell
uv run pytest tests/test_local_evidence_evaluation_utils.py -k "short or prefix" -q
```

Expected: FAIL because the local window prefix filter does not exist.

- [ ] **Step 3: Implement paired truncation and window reconstruction**

Map each validation text to a Stage 1 token boundary and filter the existing
local-evidence cache to windows whose `stop_token` is within that boundary.
This is sufficient because cached local features are window-local and avoids
recomputing the unchanged Stage 2 token similarities. Keep all existing local
model rules and aggregate-MG-score thresholds unchanged.

- [ ] **Step 4: Add the trained-bundle evaluator and verify it does not fit**

For each Mixed validation word limit, score the reconstructed local windows
through the existing model registry. Aggregate window MG scores with the same
document-level aggregation used by the current validation evaluator. Emit the
original result and short results with `validation_variant`, `word_limit`,
`classifier_family="local_evidence_agg_mg_score"`, row counts, and macro-F1.

- [ ] **Step 5: Run local-evidence tests**

Run:

```powershell
uv run pytest tests/test_local_evidence_evaluation_utils.py tests/test_local_evidence_models_utils.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit local-evidence short validation**

```powershell
git add notebooks/local_evidence_utils.py notebooks/local_evidence_evaluation_utils.py tests/test_local_evidence_evaluation_utils.py tests/test_local_evidence_models_utils.py
git commit -m "feat: evaluate local evidence on short validation"
```

### Task 4: Add Notebook Comparison and Regression Contracts

**Files:**
- Modify: `notebooks/cross_dataset_classification_poc.ipynb`
- Modify: `notebooks/cross_dataset_classification_utils.py` for shared short-validation result normalization
- Test: `tests/test_cross_dataset_classification_poc_notebook.py` and the relevant utility test module

**Interfaces:**
- Notebook parameter: `VALIDATION_WORD_LIMITS: tuple[int, ...]`.
- Notebook result: `short_validation_comparison` with original and `first_N_words` rows for both classifier families.

- [ ] **Step 1: Add notebook contract tests before editing the notebook**

Assert that the notebook source contains `VALIDATION_WORD_LIMITS`, calls both
short-validation evaluators, includes both classifier-family labels, and
displays `short_validation_comparison` and `macro_f1`.

- [ ] **Step 2: Run contract tests and verify failure**

Run:

```powershell
uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q
```

Expected: FAIL because the new parameter and comparison are absent.

- [ ] **Step 3: Add configurable notebook orchestration**

Set `VALIDATION_WORD_LIMITS` to an editable tuple, pass it to both evaluators
for each Mixed-trained scorer LLM/model configuration, concatenate original and
short results, and normalize metadata to the shared long-form schema. Preserve
the existing classification, BBC, Mixed, and visualization cells unchanged
except for adding the new section.

- [ ] **Step 4: Add comparison table and macro-F1 plot**

Display `short_validation_comparison` and plot short-limit macro-F1 grouped by
classifier family and scorer LLM. Draw the original validation macro-F1 as the
baseline for each group. Keep empty/unavailable configurations visible through
coverage rather than silently dropping them.

- [ ] **Step 5: Run the full verification suite and clear notebook outputs**

Run:

```powershell
uv run pytest -q
uv run jupyter nbconvert --clear-output --inplace notebooks/cross_dataset_classification_poc.ipynb
git diff --check
```

Expected: all tests PASS, the notebook has no stored outputs, and `git diff
--check` reports no whitespace errors.

- [ ] **Step 6: Commit notebook integration**

```powershell
git add notebooks/cross_dataset_classification_poc.ipynb notebooks/cross_dataset_classification_utils.py tests/test_cross_dataset_classification_poc_notebook.py
git commit -m "feat: compare short validation macro f1"
```

## Self-Review

- The plan covers word-based truncation, scalar whole-text evaluation, local-evidence aggregate-MG-score evaluation, unchanged trained models/rules, configurable limits, original baselines, coverage, tests, and output cleanup.
- No token-only fallback is required unless the stored text/token reconstruction check fails; such rows are excluded and reported as specified.
- All function names used by later tasks are defined in the preceding task's interface section.
- No implementation task changes training data or runs the model-processing pipeline.
