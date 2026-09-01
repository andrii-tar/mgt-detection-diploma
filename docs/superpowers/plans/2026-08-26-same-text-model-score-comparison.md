# Same-Text Model Score Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable token-position line charts that compare all available model `mg_score` curves for each seeded test text and configured window length.

**Architecture:** Add one focused plotting function beside the existing model-comparison plots. It filters `scored_windows`, reuses `expand_model_scores_to_tokens`, and returns a Matplotlib figure. The notebook selects seeded test texts and invokes the function once per text/window-length pair without changing classifier fitting or persisted data.

**Tech Stack:** Python 3.12+, pandas, NumPy, Matplotlib, pytest, Jupyter notebook JSON.

## Global Constraints

- Select a seeded random sample of texts from the existing `test` partition.
- Render one chart for every selected text and every configured `WINDOW_LENGTHS` value.
- Plot one token-position `mg_score` curve for each available model.
- Keep PyMC optional; charts include only models present in `AVAILABLE_MODELS`.
- Keep existing heatmap and normalized metric overlay behavior unchanged.
- Do not change classifier fitting, scoring, thresholding, or persisted data formats.
- Do not add notebook function definitions; reusable logic belongs in Python modules.

---

### Task 1: Add Same-Text Plot Utility

**Files:**
- Modify: `notebooks/local_evidence_comparison_utils.py` after `expand_model_scores_to_tokens`
- Test: `tests/test_local_evidence_comparison_utils.py`

**Interfaces:**
- Consumes: `scored_windows: pd.DataFrame`, `split: str`, `text_id: object`, `window_length: int`.
- Produces: `plot_same_text_model_scores(scored_windows, split, text_id, window_length) -> matplotlib.figure.Figure`.

- [ ] **Step 1: Write failing tests for filtering and model lines**

Add a fixture with two texts, two window lengths, and two models. Add this test:

```python
def test_same_text_model_scores_filters_text_and_length_and_draws_one_line_per_model():
    scored = pd.DataFrame([
        {"model": "lr", "split": "HG", "text_id": 1, "label": 0, "window_length": 4,
         "start_token": 0, "stop_token": 3, "mg_score": 0.2},
        {"model": "nb", "split": "HG", "text_id": 1, "label": 0, "window_length": 4,
         "start_token": 0, "stop_token": 3, "mg_score": 0.7},
        {"model": "other", "split": "HG", "text_id": 2, "label": 1, "window_length": 4,
         "start_token": 0, "stop_token": 3, "mg_score": 0.9},
        {"model": "lr", "split": "HG", "text_id": 1, "label": 0, "window_length": 8,
         "start_token": 0, "stop_token": 3, "mg_score": 0.4},
    ])

    figure = plot_same_text_model_scores(scored, "HG", 1, 4)
    axis = figure.axes[0]

    assert {line.get_label() for line in axis.lines} == {"lr", "nb"}
    assert axis.get_ylim() == (0.0, 1.0)
    assert "HG text 1" in axis.get_title()
    assert "window length=4" in axis.get_title()
    plt.close(figure)
```

Import the new function in the test module before running it.

- [ ] **Step 2: Write failing tests for invalid selections**

Add tests that remove one required column and that request a nonexistent text. Both must raise `ValueError`:

```python
def test_same_text_model_scores_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        plot_same_text_model_scores(pd.DataFrame({"model": ["lr"]}), "HG", 1, 4)


def test_same_text_model_scores_rejects_empty_selection():
    with pytest.raises(ValueError, match="no rows"):
        plot_same_text_model_scores(_scored_windows_with_identity_columns(), "MG", 99, 4)
```

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```powershell
$env:MPLBACKEND = "Agg"
uv run pytest tests/test_local_evidence_comparison_utils.py -q
```

Expected: the new tests fail because `plot_same_text_model_scores` is not defined.

- [ ] **Step 4: Implement the minimal plotting utility**

Add this implementation:

```python
def plot_same_text_model_scores(
    scored_windows: pd.DataFrame,
    split: str,
    text_id: object,
    window_length: int,
) -> plt.Figure:
    required = {
        "model", "split", "text_id", "label", "window_length",
        "start_token", "stop_token", "mg_score",
    }
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    selected = scored_windows.loc[
        scored_windows["split"].eq(split)
        & scored_windows["text_id"].eq(text_id)
        & scored_windows["window_length"].eq(window_length)
    ].copy()
    if selected.empty:
        raise ValueError("no rows match the requested text and window length")
    expanded = expand_model_scores_to_tokens(selected, int(selected["stop_token"].max()))
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for model in pd.unique(expanded["model"]):
        model_rows = expanded.loc[expanded["model"].eq(model)].sort_values("token_position")
        axis.plot(model_rows["token_position"], model_rows["mg_score"], linewidth=2.0, label=str(model))
    labels = pd.unique(selected["label"])
    label_text = f", label={labels[0]}" if len(labels) == 1 else ""
    axis.set_title(f"{split} text {text_id}{label_text} | window length={window_length}")
    axis.set_xlabel("Token position")
    axis.set_ylabel("MG score")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    return figure
```

- [ ] **Step 5: Run the focused tests and confirm success**

Run:

```powershell
$env:MPLBACKEND = "Agg"
uv run pytest tests/test_local_evidence_comparison_utils.py -q
```

Expected: all comparison utility tests pass.

### Task 2: Integrate Same-Text Charts Into the Notebook

**Files:**
- Modify: `notebooks/bayesian_model_comparison.ipynb` configuration and inspection code cells
- Modify: `tests/test_bayesian_model_comparison_notebook.py`

**Interfaces:**
- Consumes: `comparison_predictions`, `scored_windows`, `AVAILABLE_MODELS`, `WINDOW_LENGTHS`, `SEED`.
- Produces: one displayed Matplotlib figure per selected test text and window length.

- [ ] **Step 1: Add notebook contract assertions**

Extend `test_notebook_contains_comparison_contract` with:

```python
"plot_same_text_model_scores",
"select_comparison_texts(comparison_predictions",
"for selected_text in selected_comparison_texts.itertuples",
```

Add a structural test that confirms the notebook loops over `WINDOW_LENGTHS` and calls the new utility after selecting comparison texts.

- [ ] **Step 2: Run notebook contract tests and confirm failure**

Run:

```powershell
uv run pytest tests/test_bayesian_model_comparison_notebook.py -q
```

Expected: the new contract assertions fail because the notebook has not imported or called the utility.

- [ ] **Step 3: Import the utility in the notebook**

Add `plot_same_text_model_scores` to the existing `from local_evidence_comparison_utils import (...)` list. Keep the notebook output-free and do not add function definitions.

- [ ] **Step 4: Add seeded selection and chart loop**

After the existing comparison summary/inspection cells, add a code cell with this logic:

```python
selected_comparison_texts = select_comparison_texts(
    comparison_predictions,
    count=3,
    seed=SEED,
)
for selected_text in selected_comparison_texts.drop_duplicates(["split", "text_id"]).itertuples(index=False):
    for window_length in WINDOW_LENGTHS:
        figure = plot_same_text_model_scores(
            scored_windows,
            selected_text.split,
            selected_text.text_id,
            window_length,
        )
        display(figure)
        plt.close(figure)
```

Import `matplotlib.pyplot as plt` in the notebook configuration cell if it is not already available. The loop must use only test-derived selections from `select_comparison_texts`; it must not recompute model scores or select replacement texts when a requested length is missing.

- [ ] **Step 5: Keep the notebook output-free**

Run:

```powershell
uv run python -m nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/bayesian_model_comparison.ipynb
```

Then verify all code cells have `execution_count: null` and empty `outputs` through the existing test.

- [ ] **Step 6: Run notebook contract tests**

Run:

```powershell
uv run pytest tests/test_bayesian_model_comparison_notebook.py -q
```

Expected: all notebook contract tests pass.

### Task 3: Verify the Complete Change

**Files:**
- Inspect: `notebooks/local_evidence_comparison_utils.py`
- Inspect: `notebooks/bayesian_model_comparison.ipynb`
- Inspect: `tests/test_local_evidence_comparison_utils.py`
- Inspect: `tests/test_bayesian_model_comparison_notebook.py`

- [ ] **Step 1: Run focused utility and notebook tests**

Run:

```powershell
$env:MPLBACKEND = "Agg"
uv run pytest tests/test_local_evidence_comparison_utils.py tests/test_bayesian_model_comparison_notebook.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the broader local-evidence model tests**

Run:

```powershell
$env:MPLBACKEND = "Agg"
uv run pytest tests/test_local_evidence_models_utils.py tests/test_local_evidence_comparison_utils.py -q
```

Expected: all affected model and plotting tests pass without changing existing behavior.

- [ ] **Step 3: Check formatting and notebook output state**

Run:

```powershell
uv run pytest tests/test_bayesian_model_comparison_notebook.py::test_notebook_is_valid_output_free_nbformat -q
```

Expected: no whitespace errors and the notebook remains output-free.
