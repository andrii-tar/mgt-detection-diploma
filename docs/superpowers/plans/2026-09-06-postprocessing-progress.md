# Postprocessing Pair Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a split-count progress bar for every selected dataset/model pair processed by the postprocessing CLI.

**Architecture:** Keep all processing behavior in `run_postprocessing()`. After filters are applied, wrap that pair's split loop with `tqdm`, using the number of discovered split directories as the total and advancing once per split. Leave detailed split prints and summary accounting unchanged.

**Tech Stack:** Python 3.12+, `tqdm`, pytest, existing `uv` workflow.

## Global Constraints

- Use the existing `tqdm` dependency; do not add a new dependency.
- Preserve `--dataset`, `--model`, and `--split` filtering behavior.
- Preserve per-split status output and the final summary.
- Do not process or commit generated data artifacts.

---

### Task 1: Add Pair Progress Contract Test

**Files:**
- Modify: `tests/test_postprocess_pipeline.py`
- Read: `scripts/postprocess_pipeline.py`

**Interfaces:**
- Test the existing `run_postprocessing(processed_root: Path, dataset: str | None = None, model: str | None = None, split: str | None = None) -> int` function.
- Replace the module-level `tqdm` symbol with a recording iterator so the test does not depend on terminal rendering.

- [ ] **Step 1: Write the failing test**

Add a test that creates two valid split directories under one dataset/model pair, monkeypatches `_process_split` to return `("COMPLETE", comparison)`, and records the progress wrapper's `desc`, `total`, and each `update()` call. Assert the wrapper receives `desc="demo/model_a"`, `total=2`, and updates twice.

```python
def test_run_postprocessing_reports_progress_per_dataset_model_pair(tmp_path, monkeypatch, capsys):
    experiment = tmp_path / "demo__model_a"
    for split in ("HG", "MG"):
        (experiment / split / "data").mkdir(parents=True)

    progress = {}

    class RecordingBar:
        def __init__(self, iterable, **kwargs):
            progress.update(kwargs)
            self.iterable = iterable
            self.updates = 0
            progress["bar"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def update(self, amount):
            self.updates += amount

    monkeypatch.setattr("scripts.postprocess_pipeline.tqdm", RecordingBar)
    monkeypatch.setattr(
        "scripts.postprocess_pipeline._process_split",
        lambda split_dir: ("COMPLETE", compare_ids([1], [1])),
    )

    assert run_postprocessing(tmp_path) == 0
    assert {key: progress[key] for key in ("desc", "total", "unit")} == {
        "desc": "demo/model_a",
        "total": 2,
        "unit": "split",
    }
    assert progress["bar"].updates == 2
    assert "SUMMARY: splits=2 complete=2 invalid=0 local_evidence=0" in capsys.readouterr().out
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run pytest tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair -q`

Expected: FAIL because `scripts.postprocess_pipeline` has no `tqdm` symbol and `run_postprocessing()` does not create or update a progress bar.

### Task 2: Implement Pair Progress

**Files:**
- Modify: `scripts/postprocess_pipeline.py:3-15,203-233`
- Test: `tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair`

**Interfaces:**
- Add the module import `from tqdm import tqdm`.
- Keep `run_postprocessing()`'s signature and return value unchanged.

- [ ] **Step 1: Wrap each filtered pair's split loop**

After applying dataset/model filters, assign `split_dirs = _split_dirs(experiment_dir)`. Apply the existing split filter to that list, then process it in:

```python
with tqdm(
    split_dirs,
    desc=f"{dataset_name}/{model_name}",
    unit="split",
) as progress:
    for split_dir in progress:
        status, _ = _process_split(split_dir)
        total += 1
        if status == "COMPLETE":
            complete += 1
            valid_splits.add(split_dir.name)
        else:
            invalid += 1
```

The iterator itself advances the bar once per split; do not call `update()` separately. Keep local-evidence rebuilding after the context manager and retain the existing summary and return code.

- [ ] **Step 2: Run the focused test and verify it passes**

Run: `uv run pytest tests/test_postprocess_pipeline.py -q`

Expected: PASS, including the new progress contract test.

### Task 3: Regression Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run all postprocessing and monitoring tests**

Run: `uv run pytest tests/test_postprocess_pipeline.py tests/test_processing_monitoring_utils.py tests/test_processing_monitoring_notebook.py -q`

Expected: PASS.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 3: Inspect the final diff and status**

Run: `git diff -- scripts/postprocess_pipeline.py tests/test_postprocess_pipeline.py` and `git status --short`.

Confirm only the progress import, pair-level wrapper, and its test are changed. Do not stage generated `datasets.json` or `processing-states.json`.

- [ ] **Step 4: Commit the implementation**

```powershell
git add scripts/postprocess_pipeline.py tests/test_postprocess_pipeline.py
git commit -m "feat: show postprocessing pair progress"
```
