# D-Score Notebook Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add D-score and D-score coverage to aggregate metric analysis and text-level HG/MG heatmap exploration without adding classifiers.

**Architecture:** Keep notebook cells as orchestration and place reusable calculations and plotting functions in the existing notebook utility modules. The metrics notebook reads the existing `d_score_top20`, `d_score_top20_coverage`, and `d_score_top20_num_tokens` columns; the heatmap utility derives token coverage from stage-1 candidates and uses the production D-score functions for passage summaries.

**Tech Stack:** Python 3.12+, pandas, NumPy, Matplotlib, Seaborn, Jupyter notebooks, pytest, `uv`.

## Global Constraints

- Do not add classifier or AUC evaluation.
- Keep raw and processed data outside the repository.
- Commit notebooks only without stored cell outputs.
- Reusable notebook logic belongs in version-controlled `*_utils.py` modules.
- Use `uv run ...` for Python commands.

---

### Task 1: Extend Aggregate Metric Discovery and HG/MG Analysis

**Files:**
- Modify: `notebooks/logprob_metrics_distributions_utils.py:10-128`
- Modify: `notebooks/logprob_metrics_distributions.ipynb` metric configuration, plotting, and summary cells
- Test: `tests/test_notebook_utils.py` (create if absent)

**Interfaces:**
- `DEFAULT_METRICS` includes `d_score_top20` and `d_score_top20_coverage`.
- `available_metric_columns()` returns both D-score columns when present.
- Add `plot_split_metric_comparison(data, metrics)` to render descriptive HG/MG distributions grouped by dataset/model.

- [ ] **Step 1: Add failing utility tests**

```python
def test_available_metric_columns_includes_d_score_fields():
    frame = pd.DataFrame({
        "d_score_top20": [0.1],
        "d_score_top20_coverage": [0.8],
    })
    assert "d_score_top20" in available_metric_columns(frame)
    assert "d_score_top20_coverage" in available_metric_columns(frame)


def test_split_metric_comparison_accepts_hg_and_mg_rows():
    frame = pd.DataFrame({
        "dataset": ["demo", "demo"],
        "model": ["model", "model"],
        "split": ["HG", "MG"],
        "d_score_top20": [1.0, 2.0],
        "d_score_top20_coverage": [0.8, 0.6],
    })
    summary = split_metric_summary(frame, ["d_score_top20", "d_score_top20_coverage"])
    assert set(summary["split"]) == {"HG", "MG"}
    assert summary.loc[summary["split"] == "MG", "d_score_top20_mean"].iat[0] == 2.0
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run pytest tests/test_notebook_utils.py -q`

Expected: FAIL because D-score fields and `split_metric_summary` are not yet implemented.

- [ ] **Step 3: Implement metric selection and descriptive summary**

Add the D-score fields to `DEFAULT_METRICS` and the candidate list in
`available_metric_columns`. Add:

```python
def split_metric_summary(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = data.groupby(["dataset", "model", "split"], dropna=False)[metrics]
    summary = grouped.agg(["count", "mean", "median", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary.sort_values(["dataset", "model", "split"])
```

Add a plotting helper that creates one row per metric and one facet per
dataset/model, using `split` as hue and the same HG/MG colors already used by
the notebook. The helper must skip missing metrics with a clear axis title.

- [ ] **Step 4: Update the metrics notebook orchestration**

Keep `selected_metrics` driven by `DEFAULT_METRICS`, so both D-score fields
flow through the existing distribution, per-experiment, and `.describe()`
cells. Add a markdown section and cells equivalent to:

```python
from logprob_metrics_distributions_utils import plot_split_metric_comparison, split_metric_summary

split_metric_summary(plot_df, ["d_score_top20", "d_score_top20_coverage"])
plot_split_metric_comparison(
    plot_df,
    ["d_score_top20", "d_score_top20_coverage"],
)
```

The notebook text must explain that coverage is the mean probability mass in
the available top-20 candidate list and that D-score is a passage-level
standardized contrast, not a token probability.

- [ ] **Step 5: Run focused tests and notebook JSON validation**

Run: `uv run pytest tests/test_notebook_utils.py -q`

Expected: PASS.

Run: `uv run python -c "import json; from pathlib import Path; p=Path('notebooks/logprob_metrics_distributions.ipynb'); d=json.loads(p.read_text()); assert all(not c.get('outputs') and c.get('execution_count') is None for c in d['cells'] if c['cell_type']=='code')"`

Expected: no output and exit code 0.

### Task 2: Add D-Score and Coverage to Token Heatmap Exploration

**Files:**
- Modify: `notebooks/token_candidate_similarity_heatmaps_utils.py`
- Modify: `notebooks/token_candidate_similarity_heatmaps.ipynb`
- Test: `tests/test_notebook_utils.py`

**Interfaces:**
- `build_text_heatmap_matrices()` returns the existing matrices plus a token coverage matrix and scales, or an equivalent backward-compatible structured result.
- Add `text_d_score_summary(logprob_row)` returning `d_score_top20`, `d_score_top20_coverage`, and `d_score_top20_num_tokens`.
- `visualize_text_similarities()` displays the summary and a coverage heatmap beside the existing two heatmaps.

- [ ] **Step 1: Add failing synthetic-row tests**

```python
def test_text_d_score_summary_matches_production_metrics():
    row = pd.Series({
        "tokens": [{
            "logprob": math.log(0.5),
            "top_logprobs": [{"logprob": math.log(0.5)}],
        }],
    })
    summary = text_d_score_summary(row)
    assert summary["d_score_top20_coverage"] == pytest.approx(0.5)
    assert summary["d_score_top20_num_tokens"] == 1


def test_token_coverage_matrix_contains_candidate_probability_mass():
    matrices = build_text_heatmap_matrices(logprob_row, similarity_row, top_k=1)
    coverage_matrix = matrices.coverage_matrix
    assert coverage_matrix[0, 0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `uv run pytest tests/test_notebook_utils.py -q`

Expected: FAIL because the summary and coverage matrix are not implemented.

- [ ] **Step 3: Implement production-aligned coverage and summary helpers**

Reuse `compute_top20_d_score` and `compute_top20_coverage` from
`scripts.logprob_metrics`. Derive each token's coverage as the sum of
`exp(candidate_logprob)` for valid candidates, with `NaN` for tokens lacking a
valid candidate list. Preserve the existing logprob and similarity matrix
semantics and add a non-negative coverage scale for the new matrix.

- [ ] **Step 4: Update the heatmap renderer and notebook cells**

Add a `COVERAGE_CMAP`, render the coverage matrix with a sequential scale, and
print a small summary table containing D-score, coverage, valid-token count,
and text id before the plots. Update notebook markdown to explain HG/MG
comparison and add a coverage heatmap to every existing example invocation.

- [ ] **Step 5: Run focused tests and output-free notebook validation**

Run: `uv run pytest tests/test_notebook_utils.py -q`

Expected: PASS.

Run: `uv run python -c "import json; from pathlib import Path; p=Path('notebooks/token_candidate_similarity_heatmaps.ipynb'); d=json.loads(p.read_text()); assert all(not c.get('outputs') and c.get('execution_count') is None for c in d['cells'] if c['cell_type']=='code')"`

Expected: no output and exit code 0.

### Task 3: Full Verification and Review

**Files:**
- Verify: `notebooks/logprob_metrics_distributions.ipynb`
- Verify: `notebooks/token_candidate_similarity_heatmaps.ipynb`
- Verify: `notebooks/*_utils.py`
- Verify: `tests/test_notebook_utils.py`

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run syntax/import checks without processing models**

Run: `uv run python -m compileall notebooks`

Expected: exit code 0.

- [ ] **Step 3: Verify notebook outputs and changed-file scope**

Run: `git diff --check; git status --short`

Then inspect both notebooks with a JSON check that every code cell has
`outputs == []` and `execution_count is None`. No generated data or notebook
outputs may be added.

- [ ] **Step 4: Commit the implementation**

After review, stage only the utility modules, notebooks, tests, and the design
and plan documents, then commit with:

```powershell
git add notebooks tests docs/superpowers/specs/2026-07-29-d-score-notebook-exploration-design.md docs/superpowers/plans/2026-07-29-d-score-notebook-exploration.md
git commit -m "Add D-score notebook exploration"
```
