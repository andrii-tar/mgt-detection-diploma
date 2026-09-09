# Local-Evidence Notebook Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable local-evidence classification artifacts and two focused notebooks while retaining the all-in-one POC unchanged.

**Architecture:** A new artifact utility writes trusted fitted-model caches, bundle reconstruction metadata, decision rules, and classification result tables below `TEXT_DETECTION_DATA`. The classification notebook produces one named artifact; the SemEval-only segmentation notebook loads that artifact and scores current SemEval windows without fitting or threshold selection.

**Tech Stack:** Python 3.12, pandas, PyArrow Parquet, scikit-learn, Jupyter, pytest, uv.

## Global Constraints

- Keep `notebooks/cross_dataset_classification_poc.ipynb` unchanged.
- Keep both new notebooks output-free in Git and place reusable code in modules.
- Preserve `Mixed` and `Mixed+BBC`, BBC HG-only loading, and disjoint train/validation/test cohorts.
- Persist artifacts only below `TEXT_DETECTION_DATA/experiment-results/local-evidence-classification/<run-id>/`.
- Require `trust_artifact=True` before unpickling a persisted model cache.
- Segmentation is SemEval-only and must not fit models or select thresholds.
- Do not run Stage 1 or Stage 2 processing from notebooks or automated tests.

---

### Task 1: Persist and Restore Local-Evidence Artifacts

**Files:**
- Create: `notebooks/local_evidence_artifact_utils.py`
- Modify: `notebooks/local_evidence_training_utils.py`
- Create: `tests/test_local_evidence_artifact_utils.py`
- Modify: `tests/test_local_evidence_training_utils.py`

**Interfaces:**
- Produces `save_local_evidence_artifact(data_root, run_id, result, *, metadata, evaluation_results)` and `load_local_evidence_artifact(data_root, run_id, *, expected_metadata, trust_artifact)`.
- Produces a `LocalEvidenceArtifact` containing a reconstructed `LocalEvidenceModelBundle`, metadata, and saved result frames.
- Extends `LocalEvidenceTrainingResult` with `training_input_fingerprint: str` computed from the training and validation windows and parent cohorts.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_trusted_artifact_round_trip_restores_bundle_and_results(tmp_path: Path) -> None:
    saved = save_local_evidence_artifact(tmp_path, "run", trained, metadata=metadata, evaluation_results=results)
    loaded = load_local_evidence_artifact(tmp_path, "run", expected_metadata=metadata, trust_artifact=True)
    assert loaded.path == saved
    assert loaded.model.decision_rules.equals(trained.model.decision_rules)
    assert loaded.evaluation_results["metrics"].equals(results["metrics"])


def test_artifact_loading_requires_explicit_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_local_evidence_artifact(tmp_path, "run", trained, metadata=metadata, evaluation_results=results)
    monkeypatch.setattr(models_utils.pickle, "load", lambda _: pytest.fail("artifact must not be unpickled"))
    with pytest.raises(PermissionError, match="trust_artifact"):
        load_local_evidence_artifact(tmp_path, "run", expected_metadata=metadata)
```

- [ ] **Step 2: Run the new tests to verify failure**

Run: `uv run pytest -q tests/test_local_evidence_artifact_utils.py`

Expected: import failure because the artifact utility does not exist.

- [ ] **Step 3: Add the training fingerprint**

Compute the existing stable `training_input_fingerprint` after loading the namespaced train and validation windows, and expose it on `LocalEvidenceTrainingResult`.

```python
combined_windows = pd.concat([train_windows, validation_windows], ignore_index=True)
fingerprint = training_input_fingerprint(
    combined_windows,
    {"train": pd.concat(train_text_ids.values(), ignore_index=True),
     "validation": pd.concat(validation_text_ids.values(), ignore_index=True)},
)
```

- [ ] **Step 4: Implement trusted artifact persistence**

Write `metadata.json`, `decision_rules.parquet`, `training_metrics.parquet`, `training_coverage.parquet`, `evaluation_metrics.parquet`, `evaluation_predictions.parquet`, and `evaluation_coverage.parquet`. Use `save_fitted_models` and `load_fitted_models` with an artifact-local cache manifest containing the schema version, bundle configuration, and fingerprint. Reject an empty untrusted registry before any unpickle attempt, and reject metadata mismatch before reconstructing the bundle.

- [ ] **Step 5: Run focused artifact and training tests**

Run: `uv run pytest -q tests/test_local_evidence_artifact_utils.py tests/test_local_evidence_training_utils.py`

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add notebooks/local_evidence_artifact_utils.py notebooks/local_evidence_training_utils.py tests/test_local_evidence_artifact_utils.py tests/test_local_evidence_training_utils.py
git commit -m "feat: persist local evidence artifacts"
```

### Task 2: Add the Focused Classification Notebook

**Files:**
- Create: `notebooks/local_evidence_classification.ipynb`
- Modify: `tests/test_cross_dataset_classification_poc_notebook.py`

**Interfaces:**
- Consumes shared cohorts, `train_local_evidence_models`, `evaluate_local_evidence_datasets`, and `save_local_evidence_artifact`.
- Produces one chosen run ID with whole-text local-evidence classification result frames and a reusable bundle.

- [ ] **Step 1: Add failing notebook contracts**

```python
def test_local_evidence_classification_notebook_saves_artifact_without_semeval_segmentation() -> None:
    source = _source("local_evidence_classification.ipynb")
    assert "save_local_evidence_artifact(" in source
    assert "evaluate_local_evidence_semeval(" not in source
    assert "predict_mg_regions(" not in source
```

- [ ] **Step 2: Run the notebook contract test to verify failure**

Run: `uv run pytest -q tests/test_cross_dataset_classification_poc_notebook.py -k local_evidence_classification`

Expected: failure because the focused notebook does not exist.

- [ ] **Step 3: Create output-free orchestration cells**

Configure `RUN_ID`, data root, local models, dataset aliases, BBC HG-only splits, classifier settings, and shared capped cohorts. Train only logistic regression, evaluate source train/validation/test partitions, generate document and window-score tables plus BBC false-positive summaries, save the artifact, and display classification/token-score visualizations.

- [ ] **Step 4: Verify focused notebook contracts**

Run: `uv run pytest -q tests/test_cross_dataset_classification_poc_notebook.py`

Expected: all notebook contracts pass and the POC remains unchanged.

- [ ] **Step 5: Commit**

```powershell
git add notebooks/local_evidence_classification.ipynb tests/test_cross_dataset_classification_poc_notebook.py
git commit -m "feat: add local evidence classification notebook"
```

### Task 3: Add the SemEval Segmentation Notebook

**Files:**
- Create: `notebooks/local_evidence_segmentation.ipynb`
- Modify: `tests/test_cross_dataset_classification_poc_notebook.py`

**Interfaces:**
- Consumes `load_local_evidence_artifact`, `evaluate_local_evidence_semeval`, and SemEval annotations/current local-evidence windows.
- Produces SemEval region metrics, boundary metrics, predictions, exclusions, and token-score visualizations without creating or changing an artifact.

- [ ] **Step 1: Add failing notebook contracts**

```python
def test_local_evidence_segmentation_notebook_loads_trusted_artifact_without_training() -> None:
    source = _source("local_evidence_segmentation.ipynb")
    assert "load_local_evidence_artifact(" in source
    assert "trust_artifact=True" in source
    assert "evaluate_local_evidence_semeval(" in source
    assert "train_local_evidence_models(" not in source
```

- [ ] **Step 2: Run the contract test to verify failure**

Run: `uv run pytest -q tests/test_cross_dataset_classification_poc_notebook.py -k local_evidence_segmentation`

Expected: failure because the segmentation notebook does not exist.

- [ ] **Step 3: Create output-free SemEval-only orchestration cells**

Configure `RUN_ID` and require `trust_artifact=True`. Load the bundle, read SemEval train/dev annotations, call `evaluate_local_evidence_semeval`, retain exclusions, and display region/boundary/token-score visualizations. Do not call training, document evaluation, or artifact save functions.

- [ ] **Step 4: Run all focused tests**

Run: `uv run pytest -q tests/test_local_evidence_artifact_utils.py tests/test_local_evidence_training_utils.py tests/test_local_evidence_evaluation_utils.py tests/test_cross_dataset_classification_poc_notebook.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add notebooks/local_evidence_segmentation.ipynb tests/test_cross_dataset_classification_poc_notebook.py
git commit -m "feat: add local evidence segmentation notebook"
```

### Task 4: Verify the Complete Split

**Files:**
- Read: `notebooks/cross_dataset_classification_poc.ipynb`
- Read: `notebooks/local_evidence_classification.ipynb`
- Read: `notebooks/local_evidence_segmentation.ipynb`

**Interfaces:**
- Confirms that source notebooks are output-free and automated checks do not process data or load local LMs.

- [ ] **Step 1: Clear new notebook outputs**

Run: `uv run jupyter nbconvert --clear-output --inplace notebooks\local_evidence_classification.ipynb notebooks\local_evidence_segmentation.ipynb`

Expected: source remains with null execution counts and empty outputs.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`

Expected: zero failures.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff HEAD --check` and `git status --short`

Expected: no whitespace errors and no raw/processed data staged.
