# Postprocessing Phase Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each dataset/model progress bar advance through meaningful postprocessing phases rather than only after each split.

**Architecture:** Add an optional phase callback to the private split and local-evidence functions. `run_postprocessing()` owns the pair-level `tqdm` bar and passes a callback that advances it and updates its postfix. Existing processing decisions, output writes, filtering, summary accounting, and return codes remain unchanged.

**Tech Stack:** Python 3.12+, `tqdm`, pandas, pytest, existing `uv` workflow.

## Global Constraints

- Keep one progress bar per selected dataset/model pair, with an open-ended
  phase count so the bar does not claim a percentage based on guessed work.
- Report ID comparison, Stage 1 canonicalization, metrics, Stage 2 canonicalization, similarity statistics, and applicable local-evidence phases.
- Advance the pair bar after each completed phase and show the current phase in postfix text.
- Preserve existing split status lines, summary accounting, filters, and return codes.
- Do not add dependencies or row-level progress overhead.
- Do not process or commit generated data artifacts.

---

### Task 1: Add Failing Phase-Progress Tests

**Files:**
- Modify: `tests/test_postprocess_pipeline.py`
- Read: `scripts/postprocess_pipeline.py`

**Interfaces:**
- Exercise private `_process_split(split_dir, report_phase=...)` and `run_postprocessing(processed_root)` contracts.
- Use recording callbacks/bars; do not render a real terminal progress bar.

- [ ] **Step 1: Write a failing split-phase test**

Add a test with valid Stage 1 and Stage 2 batch files, monkeypatch `_write_metrics` and `_write_similarity_stats`, and call `_process_split` with a callback that records phase names. Assert the callback receives, in order, `read_ids`, `stage1`, `metrics`, `stage2`, and `similarity_stats`, and that the split remains `COMPLETE`.

```python
def test_process_split_reports_completed_phases(tmp_path, monkeypatch):
    split_dir = tmp_path / "demo__model_a" / "HG"
    _write_ids(split_dir / "data" / "batch.parquet", [0, 1])
    _write_ids(split_dir / "token_similarities" / "batch.parquet", [0, 1])
    monkeypatch.setattr("scripts.postprocess_pipeline._write_metrics", lambda *_: None)
    monkeypatch.setattr("scripts.postprocess_pipeline._write_similarity_stats", lambda *_: None)
    phases = []

    status, _ = _process_split(split_dir, report_phase=phases.append)

    assert status == "COMPLETE"
    assert phases == ["read_ids", "stage1", "metrics", "stage2", "similarity_stats"]
```

- [ ] **Step 2: Write a failing pair-bar phase test**

Extend the existing recording bar test so its fake bar records `set_postfix_str()` calls and the mocked `_process_split` invokes the supplied callback with two phases. Assert the pair bar receives the phase postfixes and advances once for each callback phase, not once merely for the split.

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `uv run pytest tests/test_postprocess_pipeline.py -q`

Expected: FAIL because `_process_split()` has no phase callback parameter and the pair loop does not provide or consume phase updates.

### Task 2: Implement Phase Callbacks

**Files:**
- Modify: `scripts/postprocess_pipeline.py:161-200,204-241`
- Test: `tests/test_postprocess_pipeline.py`

**Interfaces:**
- Add private callback signature `report_phase: Callable[[str], None] | None = None` to `_process_split()`.
- Add private callback signature `report_phase: Callable[[str], None] | None = None` to `_rebuild_local_evidence()`.
- Keep `run_postprocessing()`'s public signature unchanged.

- [ ] **Step 1: Implement phase reporting in `_process_split()`**

Use a local helper that calls `report_phase(name)` only when provided. Report `read_ids` immediately after ID comparison, `stage1` after Stage 1 consolidation, `metrics` after metrics, `stage2` after Stage 2 consolidation, and `similarity_stats` after statistics. For missing or invalid inputs, report only phases that actually complete.

```python
def report(name: str) -> None:
    if report_phase is not None:
        report_phase(name)
```

- [ ] **Step 2: Implement pair-bar callback wiring**

Create the pair bar with `total=None` and `unit="phase"`. Pass a callback into each `_process_split()` that calls `progress.update(1)` and `progress.set_postfix_str(phase)`. Iterate the filtered split list directly, not through `tqdm`, so each increment corresponds to a completed phase rather than a split.

- [ ] **Step 3: Report local-evidence phases**

Pass the same callback into `_rebuild_local_evidence()` and report `local_evidence` after the rebuild completes. Keep the local-evidence call inside the pair bar context so it uses the same detailed phase count and closes only after all pair work is done.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `uv run pytest tests/test_postprocess_pipeline.py -q`

Expected: PASS, including the phase ordering and pair-bar callback tests.

### Task 3: Regression Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run related tests**

Run: `uv run pytest tests/test_postprocess_pipeline.py tests/test_processing_monitoring_utils.py tests/test_processing_monitoring_notebook.py -q`

Expected: PASS.

- [ ] **Step 2: Run the complete suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Check the diff and generated files**

Run: `git diff --check` and `git status --short`.

Confirm only the postprocessing implementation/tests are staged. Leave `datasets.json` and `processing-states.json` untracked.

- [ ] **Step 4: Commit the implementation**

```powershell
git add scripts/postprocess_pipeline.py tests/test_postprocess_pipeline.py
git commit -m "feat: detail postprocessing progress"
```
