# Processed Text Count Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a split-aware heatmap to the processing-monitoring notebook showing processed text counts for every dataset/model pair.

**Architecture:** Reuse the existing `readiness` DataFrame and its `stage1_rows` values. Build a dataset/model index and split-column matrix in the notebook, render it with Seaborn, and leave missing combinations blank. No reusable utility changes are needed.

**Tech Stack:** Jupyter Notebook JSON, Python, pandas, Matplotlib, Seaborn, pytest.

## Global Constraints

- Keep the notebook output-free.
- Keep the notebook read-only and do not process or rebuild artifacts.
- Preserve the existing readiness and artifact-count charts.
- Treat `stage1_rows` as the processed-text count.

---

### Task 1: Add the processed-text heatmap

**Files:**
- Modify: `notebooks/processing_monitoring.ipynb`, code cell `monitoring-charts`
- Test: `tests/test_processing_monitoring_notebook.py`

**Interfaces:**
- Consumes: `readiness` with `dataset`, `model`, `split`, and `stage1_rows` columns.
- Produces: `processed_texts` pivot and a Seaborn heatmap with annotated counts.

- [ ] **Step 1: Write the failing source-contract test**

Add a test requiring the notebook source to contain the processed-text pivot, `stage1_rows`, and heatmap annotation configuration:

```python
def test_monitoring_notebook_contains_processed_text_heatmap() -> None:
    source = _source()
    for fragment in (
        'values="stage1_rows"',
        'processed_texts = readiness.pivot_table',
        'annot=processed_texts',
        'Processed text count by dataset, model, and split',
    ):
        assert fragment in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_processing_monitoring_notebook.py::test_monitoring_notebook_contains_processed_text_heatmap -q`

Expected: FAIL because the notebook has no `processed_texts` heatmap.

- [ ] **Step 3: Add the minimal notebook chart**

In `monitoring-charts`, add:

```python
processed_texts = readiness.pivot_table(
    index=["dataset", "model"],
    columns="split",
    values="stage1_rows",
    aggfunc="sum",
).sort_index()
processed_texts.index = [f"{dataset} / {model}" for dataset, model in processed_texts.index]

plt.figure(figsize=(max(6, len(processed_texts.columns) * 2), max(4, len(processed_texts) * 0.45)))
sns.heatmap(
    processed_texts,
    annot=processed_texts,
    fmt="g",
    cmap="Blues",
    cbar_kws={"label": "Processed texts"},
    linewidths=0.5,
    linecolor="white",
)
plt.title("Processed text count by dataset, model, and split")
plt.xlabel("Split")
plt.ylabel("Dataset / model")
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/test_processing_monitoring_notebook.py -q`

Expected: all monitoring-notebook tests pass.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass and the notebook output-free contract remains valid.

- [ ] **Step 6: Commit the implementation**

```powershell
git add notebooks/processing_monitoring.ipynb tests/test_processing_monitoring_notebook.py
git commit -m "feat: add processed text count heatmap"
```
