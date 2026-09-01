# Persisted Local-Evidence Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist fitted local-evidence classifiers outside git and reuse only compatible cache entries on later notebook runs.

**Architecture:** Add a small model-cache API to `notebooks/local_evidence_models_utils.py` that serializes each fitted `(model, window_length)` entry plus a deterministic manifest. The comparison function will optionally load/save through that API while retaining the in-memory registry used by SemEval scoring; the notebook will expose opt-in save/reuse flags.

**Tech Stack:** Python 3.12, pandas, scikit-learn, PyMC optional, pickle, JSON, pytest, Jupyter notebooks.

## Global Constraints

- Store caches under `TEXT_DETECTION_DATA/processed/{dataset}__{model}/local_evidence/models/`.
- Cache reuse requires exact identity/configuration matching and otherwise retrains.
- Save generated model artifacts outside the repository and never commit them.
- Preserve existing classifier output schemas and in-memory `fitted_models` behavior.
- Keep reusable logic in Python modules rather than notebook cells.

---

### Task 1: Define Cache API and Serialization Tests

**Files:**
- Modify: `tests/test_local_evidence_models_utils.py`

**Interfaces:**
- Tests target `save_fitted_models(cache_root, fitted_models, manifest)` and `load_fitted_models(cache_root, manifest)`.

- [ ] **Step 1: Write failing tests**

Cover deterministic manifest matching, estimator round-trip prediction equality, incompatible manifest returning an empty registry, malformed cache fallback, and atomic replacement behavior.

- [ ] **Step 2: Run focused tests**

Run: `uv run pytest tests/test_local_evidence_models_utils.py -q`

Expected: failures because the cache API does not exist.

- [ ] **Step 3: Commit red tests**

Run: `git add tests/test_local_evidence_models_utils.py; git commit -m "test: specify persisted model cache"`

### Task 2: Implement Safe Model Persistence

**Files:**
- Modify: `notebooks/local_evidence_models_utils.py`

**Interfaces:**
- `save_fitted_models(cache_root: Path, fitted_models: Mapping[tuple[str, int], _FittedClassifier], manifest: Mapping[str, object]) -> None`.
- `load_fitted_models(cache_root: Path, manifest: Mapping[str, object]) -> dict[tuple[str, int], _FittedClassifier]`.

- [ ] **Step 1: Validate manifest identity**

Canonicalize manifest JSON with sorted keys and compare it to the stored manifest before loading any estimator.

- [ ] **Step 2: Implement atomic save**

Write one temporary pickle containing the registry and canonical manifest, replace the final cache file with `Path.replace`, and create the parent directory as needed.

- [ ] **Step 3: Implement defensive load**

Return `{}` for missing, malformed, incompatible, or structurally invalid caches; never allow a cache failure to stop retraining.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_local_evidence_models_utils.py -q`

Expected: all model-cache and existing model tests pass.

- [ ] **Step 5: Commit persistence API**

Run: `git add notebooks/local_evidence_models_utils.py tests/test_local_evidence_models_utils.py; git commit -m "feat: persist local evidence models"`

### Task 3: Integrate Cache With Comparison Notebook

**Files:**
- Modify: `notebooks/local_evidence_models_utils.py`
- Modify: `notebooks/bayesian_model_comparison.ipynb`
- Modify: `tests/test_bayesian_model_comparison_notebook.py`

**Interfaces:**
- `run_local_evidence_model_comparison` accepts optional cache root/manifest and save/reuse booleans without changing its existing return tuple.

- [ ] **Step 1: Add failing integration tests**

Verify that the notebook defines `SAVE_FITTED_MODELS` and `REUSE_FITTED_MODELS`, builds a cache path below `PROCESSED_ROOT`, and passes the cache options to the comparison call.

- [ ] **Step 2: Build a complete manifest**

Include dataset, model, feature columns, seed, window lengths, model names, Bayesian configuration, and Gaussian-mixture configuration in the notebook-provided manifest.

- [ ] **Step 3: Load before fitting and save after fitting**

When reuse is enabled, prepopulate `fitted_models` from a matching cache; when save is enabled, persist the registry after successful comparison training. Existing entries remain usable for same-run SemEval scoring.

- [ ] **Step 4: Run focused tests and clear notebook outputs**

Run: `uv run pytest tests/test_local_evidence_models_utils.py tests/test_bayesian_model_comparison_notebook.py -q` and clear outputs with `uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/bayesian_model_comparison.ipynb`.

- [ ] **Step 5: Commit integration**

Run: `git add notebooks/local_evidence_models_utils.py notebooks/bayesian_model_comparison.ipynb tests/test_bayesian_model_comparison_notebook.py; git commit -m "feat: reuse persisted notebook models"`

### Task 4: Verify Cache Reuse and Full Regression Suite

**Files:**
- No generated cache files committed.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Verify notebook source hygiene**

Parse the notebook as JSON and assert every code cell has `outputs == []` and `execution_count is None`.

- [ ] **Step 3: Run a cache smoke test**

Use a temporary cache root and a deterministic classifier fixture to save, load, and compare predictions; verify that a manifest change returns an empty cache.

- [ ] **Step 4: Review final status**

Run `git status --short`, `git diff --check`, and confirm only intended source/test/docs commits contain changes; confirm no generated cache is tracked.
