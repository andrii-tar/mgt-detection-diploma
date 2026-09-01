# SemEval Sustained Boundary Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace noisy first-threshold SemEval boundary predictions with smoothed token-level sustained crossings and reevaluate boundary accuracy.

**Architecture:** `predict_boundary_positions` will expand window scores to tokens, apply a centered rolling median, and select the first run of three consecutive scores above threshold. The notebook will expose the smoothing and run parameters and report early, late, and fallback rates alongside existing error metrics.

**Tech Stack:** Python 3.12, pandas, NumPy, pytest, Jupyter nbconvert.

## Global Constraints

- Keep raw and processed data outside the repository under `TEXT_DETECTION_DATA`.
- Preserve the existing prediction output columns and existing aggregate metrics.
- Keep reusable logic in `notebooks/semeval_boundary_utils.py`, not notebook cells.
- Do not store executed notebook outputs in the committed source notebook.

---

### Task 1: Add Sustained-Crossing Tests

**Files:**
- Modify: `tests/test_semeval_boundary_utils.py`

**Interfaces:**
- Tests target `predict_boundary_positions(scored_windows, token_counts, threshold=0.5, smoothing_window=5, sustained_length=3)`.

- [ ] **Step 1: Write failing tests**

Add tests for a one-token spike being ignored, the first three-token run being selected, no sustained run falling back to `token_count`, and invalid even/zero smoothing or run lengths raising `ValueError`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_semeval_boundary_utils.py -q`

Expected: failures because the new keyword arguments and sustained behavior do not exist.

- [ ] **Step 3: Commit the red tests**

Run: `git add tests/test_semeval_boundary_utils.py; git commit -m "test: specify sustained SemEval crossings"`

### Task 2: Implement Smoothed Sustained Crossings

**Files:**
- Modify: `notebooks/semeval_boundary_utils.py`

**Interfaces:**
- Produces `predict_boundary_positions(..., smoothing_window: int = 5, sustained_length: int = 3) -> pd.DataFrame`.
- Keeps columns `model`, `window_length`, `text_id`, `predicted_token_boundary`, and `token_count` unchanged.

- [ ] **Step 1: Add parameter validation**

Reject non-positive or even `smoothing_window` values and non-positive `sustained_length` values with clear `ValueError`s.

- [ ] **Step 2: Implement smoothing and run detection**

For each grouped token score series, use `pd.Series(scores).rolling(smoothing_window, center=True, min_periods=1).median()`. Find the first index whose threshold mask contains `sustained_length` consecutive `True` values, and return that index. Return `token_count` if no run exists.

- [ ] **Step 3: Run the focused tests**

Run: `uv run pytest tests/test_semeval_boundary_utils.py -q`

Expected: all boundary tests pass.

- [ ] **Step 4: Commit the implementation**

Run: `git add notebooks/semeval_boundary_utils.py tests/test_semeval_boundary_utils.py; git commit -m "fix: use sustained smoothed SemEval crossings"`

### Task 3: Add Boundary Diagnostics and Notebook Configuration

**Files:**
- Modify: `notebooks/semeval_boundary_utils.py`
- Modify: `notebooks/bayesian_model_comparison.ipynb`
- Modify: `tests/test_bayesian_model_comparison_notebook.py`

**Interfaces:**
- `summarize_boundary_predictions` adds `early_rate`, `late_rate`, and `no_crossing_rate` while preserving existing columns.
- Notebook configuration sets `SEMEVAL_SMOOTHING_WINDOW = 5` and `SEMEVAL_SUSTAINED_LENGTH = 3`.

- [ ] **Step 1: Add diagnostic tests**

Test that errors below zero are impossible, positive/negative signed errors classify early/late correctly, and fallback predictions are counted as no crossing.

- [ ] **Step 2: Implement diagnostics**

Track signed error before absolute conversion, define early as predicted boundary below gold, late as above gold, and identify fallback rows where prediction equals `token_count` and no sustained run was found.

- [ ] **Step 3: Wire notebook parameters**

Pass the two configuration values into `predict_boundary_positions`, display the expanded summary, and add notebook contract fragments for the new parameters.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_semeval_boundary_utils.py tests/test_bayesian_model_comparison_notebook.py -q`

Expected: all focused tests pass and the notebook remains output-free in source form.

- [ ] **Step 5: Commit notebook wiring**

Run: `git add notebooks/semeval_boundary_utils.py notebooks/bayesian_model_comparison.ipynb tests/test_semeval_boundary_utils.py tests/test_bayesian_model_comparison_notebook.py; git commit -m "feat: report SemEval boundary trigger diagnostics"`

### Task 4: Execute and Compare SemEval Results

**Files:**
- No committed file changes; execute a temporary notebook copy under `C:\Users\Andrii\AppData\Local\Temp\opencode`.

- [ ] **Step 1: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output executed_bayesian_model_comparison.ipynb --output-dir C:\Users\Andrii\AppData\Local\Temp\opencode notebooks\\bayesian_model_comparison.ipynb`

- [ ] **Step 2: Extract results**

Read the SemEval summary output and record MAE, median error, exact accuracy, within-1/3/5 accuracy, early rate, late rate, no-crossing rate, and text coverage for every model/window combination.

- [ ] **Step 3: Compare against baseline**

Compare the new table against the previous first-crossing baseline: LR/32 MAE `54.465347`, exact accuracy `0.033663`, and within-5 accuracy `0.114851`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass, with no committed notebook outputs.

- [ ] **Step 5: Report findings**

Summarize whether sustained crossing improves MAE and tolerance accuracy, identify the best model/window, and call out any remaining early-trigger or coverage problems.
