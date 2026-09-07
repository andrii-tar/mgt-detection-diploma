# Classification Heatmaps and SemEval Boundary Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train/validation/test macro-F1 heatmaps and SemEval boundary visualizations to the existing output-free cross-dataset notebook without removing its result tables.

**Architecture:** Extend the existing scalar and local-evidence evaluation loops to retain train metrics in the same result frames as validation and test. Append one visualization cell that pivots those frames into per-LLM heatmaps and plots aggregate/sample SemEval boundary accuracy from existing boundary result frames. Keep all visualization logic notebook-local and make empty-frame behavior a no-op.

**Tech Stack:** Python 3.12, pandas, matplotlib, seaborn, Jupyter, pytest, nbformat.

## Global Constraints

- Preserve the existing result-table display cell.
- Render scalar and local-evidence macro-F1 for `train`, `validation`, and `test`.
- Heatmap rows are training dataset/regime and columns are evaluation dataset/regime.
- Render visualizations for every configured scorer LLM.
- Use SemEval `exact_accuracy`, `within_1_accuracy`, `within_3_accuracy`, and `within_5_accuracy` for aggregate boundary plots.
- Sample SemEval boundary plots must show predicted and gold boundaries over token positions and use deterministic sampling.
- Keep the notebook output-free and do not run stage 1/2 processing or large model jobs.

---

### Task 1: Evaluate All Three Partitions

**Files:**
- Modify: `notebooks/cross_dataset_classification_poc.ipynb`, scalar and local-evidence evaluation cells
- Modify: `tests/test_cross_dataset_classification_poc_notebook.py`

**Interfaces:**
- Consumes the existing `train_whole_text_models`, `evaluate_whole_text_datasets`, `train_local_evidence_models`, and `evaluate_local_evidence_datasets` APIs.
- Produces `whole_text_metrics` and `local_evidence_metrics` rows with `partition` equal to each of `train`, `validation`, and `test`.

- [ ] **Step 1: Write failing contract assertions**

Assert the notebook source contains `for partition in ("train", "validation", "test")` in both evaluation blocks and that the existing `matrix-displays` cell still calls `display(` with result frames.

- [ ] **Step 2: Run the focused test**

Run: `uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q`

Expected: the new partition assertion fails before the notebook change.

- [ ] **Step 3: Add train evaluation**

Change both existing partition loops from:

```python
for partition in ("validation", "test"):
```

to:

```python
for partition in ("train", "validation", "test"):
```

Use the existing partition lookup and labelling code unchanged so tables and metadata remain consistent.

- [ ] **Step 4: Run the focused test**

Run: `uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/cross_dataset_classification_poc.ipynb tests/test_cross_dataset_classification_poc_notebook.py
git commit -m "feat: expose train partition classification metrics"
```

### Task 2: Append Classification and SemEval Visualizations

**Files:**
- Modify: `notebooks/cross_dataset_classification_poc.ipynb`, append one code cell after `matrix-displays`
- Modify: `tests/test_cross_dataset_classification_poc_notebook.py`

**Interfaces:**
- Consumes `whole_text_metrics`, `local_evidence_metrics`, `semeval_boundary_summary`, and `boundary_predictions`.
- Produces displayed matplotlib figures only; it does not replace or mutate the existing result tables.

- [ ] **Step 1: Write failing visualization contract assertions**

Assert the final notebook cell is a code cell whose source contains all of the following literals: `train`, `validation`, `test`, `macro_f1`, `sns.heatmap`, `semeval_boundary_summary`, `exact_accuracy`, `within_1_accuracy`, `within_3_accuracy`, `within_5_accuracy`, `boundary_predictions`, `gold_token_boundary`, and `predicted_token_boundary`. Assert the result-table cell precedes it.

- [ ] **Step 2: Run the focused test**

Run: `uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q`

Expected: the new visualization assertions fail because no appended cell exists.

- [ ] **Step 3: Append the visualization cell**

Add one code cell after the existing result-table cell with this behavior:

```python
def _display_f1_heatmaps(frame, title_prefix):
    if frame.empty:
        return
    for llm in sorted(frame["llm"].dropna().unique()):
        llm_frame = frame.loc[frame["llm"].eq(llm)]
        for partition in ("train", "validation", "test"):
            selected = llm_frame.loc[llm_frame["partition"].eq(partition)]
            if selected.empty:
                continue
            matrix = selected.pivot_table(
                index="training_regime", columns="evaluation_regime",
                values="macro_f1", aggfunc="first",
            )
            if matrix.empty:
                continue
            plt.figure(figsize=(max(6, 1.5 * len(matrix.columns)), max(4, 1.0 * len(matrix.index))))
            sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis")
            plt.title(f"{title_prefix} macro F1 | {llm} | {partition}")
            plt.xlabel("Evaluation dataset")
            plt.ylabel("Training dataset")
            plt.tight_layout()
            plt.show()


_display_f1_heatmaps(whole_text_metrics, "Whole-text scalar metrics")
_display_f1_heatmaps(local_evidence_metrics, "Local evidence")

if not semeval_boundary_summary.empty:
    boundary_metrics = ["exact_accuracy", "within_1_accuracy", "within_3_accuracy", "within_5_accuracy"]
    boundary_plot = semeval_boundary_summary.melt(
        id_vars=["llm", "training_regime", "evaluation_dataset", "window_length"],
        value_vars=boundary_metrics, var_name="accuracy_tolerance", value_name="accuracy",
    )
    sns.catplot(
        data=boundary_plot, x="training_regime", y="accuracy", hue="accuracy_tolerance",
        col="evaluation_dataset", row="llm", kind="bar", sharey=True, height=4, aspect=1.4,
    )
    plt.show()

if not boundary_predictions.empty:
    sample_columns = {"llm", "training_regime", "evaluation_dataset", "window_length", "text_id", "predicted_token_boundary", "gold_token_boundary"}
    if sample_columns.issubset(boundary_predictions.columns):
        samples = boundary_predictions.sort_values(["llm", "evaluation_dataset", "text_id"]).groupby(
            ["llm", "evaluation_dataset"], sort=True, group_keys=False,
        ).head(3)
        for (llm, dataset), group in samples.groupby(["llm", "evaluation_dataset"], sort=True):
            figure, axes = plt.subplots(len(group), 1, figsize=(12, max(2.5, 2.5 * len(group))), squeeze=False)
            for axis, (_, row) in zip(axes[:, 0], group.iterrows()):
                axis.axvline(row["gold_token_boundary"], color="tab:green", label="gold")
                axis.axvline(row["predicted_token_boundary"], color="tab:red", linestyle="--", label="predicted")
                axis.set_xlim(0, row["token_count"])
                axis.set_yticks([])
                axis.set_title(f"{llm} | {dataset} | text {row['text_id']}")
                axis.legend(loc="upper right")
            figure.supxlabel("Token position")
            figure.tight_layout()
            plt.show()
```

The implementation may use equivalent formatting, but must preserve the exact data sources, partition set, metric names, deterministic sample ordering, and empty-data guards above.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/cross_dataset_classification_poc.ipynb tests/test_cross_dataset_classification_poc_notebook.py
git commit -m "feat: add classification and boundary visualizations"
```

### Task 3: Verify Output-Free Notebook and Full Suite

**Files:**
- Verify: `notebooks/cross_dataset_classification_poc.ipynb`
- Verify: all changed Python and test files

- [ ] **Step 1: Clear notebook outputs**

Run: `uv run jupyter nbconvert --clear-output --inplace notebooks/cross_dataset_classification_poc.ipynb`

- [ ] **Step 2: Verify notebook contract and output state**

Run: `uv run pytest tests/test_cross_dataset_classification_poc_notebook.py -q`

Expected: PASS, with every code cell having `execution_count` null and `outputs` empty.

- [ ] **Step 3: Run the complete suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Check repository diff**

Run: `git diff --check`.

Expected: no whitespace errors.

- [ ] **Step 5: Commit any output-clearing-only change**

If `nbconvert` changed the notebook after Task 2, run:

```bash
git add notebooks/cross_dataset_classification_poc.ipynb
git commit -m "chore: keep visualization notebook output-free"
```
