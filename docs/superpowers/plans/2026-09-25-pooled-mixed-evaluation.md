# Pooled Mixed Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Mixed` and `Mixed+BBC` classifier evaluation use one pooled cohort capped at 1,000 `HG` and 1,000 `MG` parent texts before partitioning, then emit one combined result per evaluation partition and window.

**Architecture:** Add a reusable pooled-cohort function alongside the existing per-dataset cohort function. Keep the current single-dataset evaluator path intact; branch only mixed compositions to pool source identities, score all selected source batches into one prediction artifact, and aggregate one combined metric and coverage row.

**Tech Stack:** Python 3.12+, pandas, NumPy/scikit-learn cohort utilities, PyArrow Parquet streaming, pytest, Jupyter notebooks stored as `.ipynb` JSON.

## Global Constraints

- Single-dataset evaluation behavior and artifact formats remain unchanged.
- Mixed sampling uses the existing deterministic artifact cohort seed.
- Mixed pooled limits are 1,000 total parent texts for each pooled `HG`/`MG` source split before train/validation/test partitioning.
- Parent identity must remain `(dataset, split, text_id)` throughout selection and scoring.
- Mixed results emit one combined metric and coverage row per partition/window; no per-dataset mixed metric rows.
- Do not run the full model evaluation matrix as automated verification.
- Do not delete existing model artifacts or prepared feature partitions.

---

### Task 1: Add Pooled Cohort Selection

**Files:**
- Modify: `notebooks/utils/classification_experiment_utils.py:133-163` to add `build_pooled_feature_parent_cohorts` without changing `build_feature_parent_cohorts`.
- Test: `tests/test_classification_experiment_utils.py` next to the existing cohort tests around lines 145-205.

**Interfaces:**
- Consumes: A parent-level or feature-row `DataFrame` with `dataset`, `split`, `text_id`, `label`, and optionally `token_count` or `stop_token`.
- Produces: `build_pooled_feature_parent_cohorts(frame, *, seed, max_per_split=1000, min_text_tokens=None) -> dict[str, set[tuple[str, str, object]]]`, returning `(dataset, split, text_id)` identities for `train`, `validation`, and `test`.

- [ ] **Step 1: Write the failing pooled-cohort tests**

Add tests that construct three datasets with overlapping numeric `text_id` values and both `HG`/`MG` source splits. Assert that:

```python
cohorts = build_pooled_feature_parent_cohorts(
    frame,
    seed=42,
    max_per_split=4,
)
selected = set().union(*cohorts.values())

assert all(len({identity for identity in selected if identity[1] == split}) <= 4 for split in ("HG", "MG"))
assert all(len(identity) == 3 for identity in selected)
assert {identity[0] for identity in selected} <= {"Kaggle", "MAGE", "Outfox"}
```

Also add a deterministic repeatability assertion:

```python
assert build_pooled_feature_parent_cohorts(frame, seed=42, max_per_split=4) == cohorts
```

Use enough rows that the cap is exercised and include duplicate `text_id` values across datasets. Add a minimum-token test showing that `min_text_tokens` filters before the pooled cap, using the same `token_count`/`stop_token` convention as `build_feature_parent_cohorts`.

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
uv run pytest tests/test_classification_experiment_utils.py -k pooled -q
```

Expected: collection or test failure because `build_pooled_feature_parent_cohorts` does not exist.

- [ ] **Step 3: Implement the pooled helper**

Implement the helper with the following behavior:

```python
def build_pooled_feature_parent_cohorts(
    frame: pd.DataFrame,
    *,
    seed: int,
    max_per_split: int = 1000,
    min_text_tokens: int | None = None,
) -> dict[str, set[tuple[str, str, object]]]:
    """Select deterministic pooled parent identities across datasets."""
```

Validate the same required columns as the existing helper. Collapse feature rows to one parent row per `(dataset, split, text_id)` and compute token counts per same key when filtering is requested. For each source split, use `DataFrame.sample(n=min(max_per_split, len(parents)), random_state=seed)` on the pooled parent rows so the cap is total across datasets and the selected sample is not biased toward one dataset's numeric IDs. Pass the pooled sampled parent rows to the existing `split_cohort` helper with the same seed, then convert each returned partition to `(str(dataset), str(split), text_id)` tuples. Do not alter `build_feature_parent_cohorts` or `build_deterministic_parent_cohorts`.

- [ ] **Step 4: Run the utility tests and verify they pass**

Run:

```powershell
uv run pytest tests/test_classification_experiment_utils.py -k "pooled or cohort" -q
```

Expected: all selected cohort tests pass, including the pre-existing per-dataset cohort tests.

- [ ] **Step 5: Commit the utility change**

```powershell
git add notebooks/utils/classification_experiment_utils.py tests/test_classification_experiment_utils.py
git commit -m "feat: add pooled classifier evaluation cohorts"
```

### Task 2: Combine Mixed Evaluation Artifacts

**Files:**
- Modify: `notebooks/evaluate_classifiers.ipynb` evaluation loop around the `source_names`, cohort construction, prediction-artifact, and metric aggregation cells.
- Test: `tests/test_feature_vector_classification_notebooks.py` around `test_evaluate_notebook_uses_reduced_composition_matrix_and_windows`.

**Interfaces:**
- Consumes: `build_pooled_feature_parent_cohorts` from Task 1, existing prepared Parquet partitions, `iter_prepared_batches`, `write_prediction_artifact`, and `aggregate_prediction_artifact`.
- Produces: For mixed composition IDs, prediction artifacts whose `dataset` key is the composition ID and whose rows contain all selected source datasets; one `evaluation_metrics`, `evaluation_coverage`, and `run_summary` row per partition/window.

- [ ] **Step 1: Extend notebook contract tests before changing the notebook**

Add source assertions requiring the notebook to contain the pooled mixed branch and combined output semantics:

```python
assert "build_pooled_feature_parent_cohorts" in source
assert "MIXED_MAX_PER_SPLIT = 1000" in source
assert "is_mixed_evaluation = len(source_names) > 1" in source
assert "dataset=evaluation_composition_id" in source
assert "build_pooled_feature_parent_cohorts(" in source
```

Assert that the mixed path uses one composition-level artifact instead of a per-source artifact loop by checking for a composition-level `prediction_artifact_path` call and a combined `aggregate_prediction_artifact` call. Retain the existing assertions proving single-dataset compositions and the reduced windows remain configured.

- [ ] **Step 2: Run the notebook contract tests and verify they fail**

Run:

```powershell
uv run pytest tests/test_feature_vector_classification_notebooks.py -k "reduced_composition or evaluate_notebook" -q
```

Expected: failure of the new pooled-branch assertions.

- [ ] **Step 3: Add the mixed pooled branch to the notebook**

Import `build_pooled_feature_parent_cohorts` and define:

```python
MIXED_MAX_PER_SPLIT = 1000
```

Within the evaluation-composition loop, set `is_mixed_evaluation = len(source_names) > 1`. Use `build_feature_parent_cohorts(..., max_per_split=int(metadata.get("max_per_split", 600)))` for single-dataset evaluations and `build_pooled_feature_parent_cohorts(..., max_per_split=MIXED_MAX_PER_SPLIT)` for mixed evaluations. Keep the existing `min_text_tokens` and cohort seed values.

For mixed evaluations, derive selected IDs by `(dataset, source_split)` from the returned identity tuples. Build one lazy batch iterator that loops through each source dataset and source split, calls `iter_prepared_batches` with that source's selected IDs, scores the batches, and assigns both `source_dataset=<source dataset>` and `dataset=evaluation_composition_id`. Feed this iterator once to `write_prediction_artifact` using an artifact key whose `dataset` is the composition ID.

For single-dataset evaluations, preserve the current per-dataset artifact key and scoring loop. Do not change their cohort limit, artifact path, or metric row shape.

- [ ] **Step 4: Aggregate one combined metric and coverage row for mixed evaluations**

After the mixed prediction artifact is complete, call `aggregate_prediction_artifact` once. Create one metrics row with `dataset=evaluation_composition_id` and one coverage row with:

```python
{
    "dataset": evaluation_composition_id,
    "partition": partition,
    "split": "HG+MG",
    "class": "binary",
    "parent_count": len(selected_cohort),
    "row_count": artifact_metrics["hg_support"] + artifact_metrics["mg_support"],
    "availability_status": "available",
    "rows": artifact_metrics["hg_support"] + artifact_metrics["mg_support"],
    "hg_rows": artifact_metrics["hg_support"],
    "mg_rows": artifact_metrics["mg_support"],
}
```

Preserve the existing dimensions, resume keys, and error handling. Make mixed completion checks use the composition-level artifact key so a complete mixed artifact is skipped on resume. Append one mixed `run_summary` row per partition, not one row per source dataset.

- [ ] **Step 5: Run the notebook contract tests and verify they pass**

Run:

```powershell
uv run pytest tests/test_feature_vector_classification_notebooks.py -k "reduced_composition or evaluate_notebook" -q
```

Expected: all selected notebook contract tests pass.

- [ ] **Step 6: Commit the notebook change**

```powershell
git add notebooks/evaluate_classifiers.ipynb tests/test_feature_vector_classification_notebooks.py
git commit -m "feat: combine pooled mixed evaluation results"
```

### Task 3: Full Focused Verification

**Files:**
- Modify: none unless a test exposes a contract mismatch.
- Test: `tests/test_classification_experiment_utils.py`, `tests/test_feature_vector_classification_notebooks.py`.

**Interfaces:**
- Consumes: The pooled utility and composition-level mixed evaluation path from Tasks 1-2.
- Produces: Verified behavior with no changes to unrelated notebook execution state.

- [ ] **Step 1: Run the complete focused classifier test set**

Run:

```powershell
uv run pytest tests/test_classification_experiment_utils.py tests/test_feature_vector_classification_notebooks.py -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Inspect the final diff and worktree scope**

Run:

```powershell
git status --short
git diff --check
git diff --stat HEAD~2..HEAD
```

Confirm that only the pooled utility, evaluator notebook, and their tests are part of the implementation commits. Leave unrelated local execution-state changes in `notebooks/prepare_and_train_classifiers.ipynb` unstaged.

- [ ] **Step 3: Commit any required test-only correction**

If Task 3 identifies an implementation defect, return to the smallest relevant task, add a regression test first, rerun its focused command, then commit the correction with:

```powershell
git add notebooks/utils/classification_experiment_utils.py notebooks/evaluate_classifiers.ipynb tests/test_classification_experiment_utils.py tests/test_feature_vector_classification_notebooks.py
git commit -m "fix: correct pooled mixed evaluation"
```
