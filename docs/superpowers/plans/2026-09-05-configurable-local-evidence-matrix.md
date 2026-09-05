# Configurable Local-Evidence Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the cross-dataset notebook to run the complete Kaggle/Outfox/Mixed train/evaluation matrix with SmolLM and a configurable classifier list, then evaluate every trained model on SemEval.

**Architecture:** Keep the notebook as orchestration only. Reuse the existing local-evidence training and dataset/SemEval evaluation utilities, adding one small aggregation helper for combining document-level predictions into a Mixed evaluation regime. Build deterministic per-dataset cohorts once, derive regime unions from those cohorts, and attach explicit LLM, training-regime, and evaluation-regime labels to all outputs.

**Tech Stack:** Python 3.12, Jupyter Notebook JSON, pandas, scikit-learn-backed local-evidence models, pytest, `uv`.

## Global Constraints

- Initial LLM configuration is exactly `("smollm2_360m",)`.
- Initial source datasets are exactly `kaggle_train` and `outfox_chatgpt_train`.
- Kaggle and Outfox are split independently; Mixed is the union of matching partitions.
- Default classifier configuration is exactly `("logistic_regression",)` and remains editable.
- Do not run stage 1/2 processing or large model jobs during verification.
- Keep notebook code cells output-free and execution-count-free.
- Do not commit raw data, processed Parquet files, model weights, secrets, or generated monitoring output.

---

### Task 1: Specify Mixed Evaluation Aggregation

**Files:**
- Modify: `notebooks/local_evidence_evaluation_utils.py:54-110`
- Test: `tests/test_local_evidence_evaluation_utils.py`

**Interfaces:**
- Consumes: `LocalEvidenceModelBundle`, per-dataset test cohorts, and existing `evaluate_local_evidence_datasets` outputs.
- Produces: a small public helper that combines document-level evaluation results for a regime without refitting or recalibrating thresholds.

- [ ] **Step 1: Write the failing test**

Add a test with two explicit `LocalEvidenceDatasetEvaluationResult` instances. Each result should contain one document prediction with the same rule metadata, a `label`, an `mg_score`, and a namespaced `text_id`; combine them as Mixed and assert that the combined metrics have one row per classifier/window/aggregation rule, that `evaluation_regime` is `Mixed`, and that both prediction IDs are retained.

```python
import pandas as pd

from notebooks.local_evidence_evaluation_utils import (
    LocalEvidenceDatasetEvaluationResult,
    combine_local_evidence_dataset_results,
)


def _evaluation_result(text_id: str, label: int, score: float) -> LocalEvidenceDatasetEvaluationResult:
    predictions = pd.DataFrame([{
        "text_id": text_id,
        "label": label,
        "mg_score": score,
        "model": "logistic_regression",
        "window_length": 32,
        "aggregation": "maximum_window",
        "window_threshold": float("nan"),
        "document_threshold": 0.5,
        "score_direction": "higher_is_mg",
        "prediction": int(score >= 0.5),
    }])
    metrics = pd.DataFrame([{
        "model": "logistic_regression",
        "window_length": 32,
        "aggregation": "maximum_window",
        "window_threshold": float("nan"),
        "document_threshold": 0.5,
        "score_direction": "higher_is_mg",
    }])
    return LocalEvidenceDatasetEvaluationResult(
        metrics=metrics,
        predictions=predictions,
        scored_windows=pd.DataFrame(),
        coverage=pd.DataFrame([{"evaluation_dataset": text_id.split("::")[0]}]),
    )


def test_combined_dataset_result_preserves_documents_and_uses_existing_rules() -> None:
    combined = combine_local_evidence_dataset_results(
        [
            _evaluation_result("kaggle::HG::2", 0, 0.1),
            _evaluation_result("outfox::MG::5", 1, 0.9),
        ],
        evaluation_regime="Mixed",
    )
    assert set(combined.predictions["text_id"]) == {
        "kaggle::HG::2", "outfox::MG::5",
    }
    assert set(combined.metrics["evaluation_regime"]) == {"Mixed"}
    assert len(combined.metrics) == 1
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
uv run pytest -q tests/test_local_evidence_evaluation_utils.py::test_combined_dataset_result_preserves_documents_and_uses_existing_rules
```

Expected: collection failure because `combine_local_evidence_dataset_results` does not exist.

- [ ] **Step 3: Implement the minimal helper**

Implement `combine_local_evidence_dataset_results(results, *, evaluation_regime)` in `notebooks/local_evidence_evaluation_utils.py`:

1. Concatenate each result’s `predictions` and `scored_windows`.
2. Group predictions by `model`, `window_length`, `aggregation`, `window_threshold`, `document_threshold`, and `score_direction`.
3. Recompute metrics from the concatenated document `label` and `mg_score`, negating scores for `higher_is_hg` exactly as the existing evaluator does.
4. Add `evaluation_regime` to metrics and predictions.
5. Concatenate coverage and add the same regime label.
6. Return `LocalEvidenceDatasetEvaluationResult` without changing the existing per-dataset evaluator contract.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
uv run pytest -q tests/test_local_evidence_evaluation_utils.py
```

Expected: all evaluation utility tests pass.

- [ ] **Step 5: Commit the utility change**

```powershell
git add tests/test_local_evidence_evaluation_utils.py notebooks/local_evidence_evaluation_utils.py
git commit -m "feat: support mixed local evidence evaluation"
```

### Task 2: Refactor the Cross-Dataset Notebook Matrix

**Files:**
- Modify: `notebooks/cross_dataset_classification_poc.ipynb`
- Test: `tests/test_cross_dataset_classification_poc_notebook.py`

**Interfaces:**
- Consumes: `train_local_evidence_models`, `evaluate_local_evidence_datasets`, `combine_local_evidence_dataset_results`, `evaluate_local_evidence_semeval`, `load_or_build_local_evidence_windows`, and `split_local_evidence_texts`.
- Produces: output-free notebook code that executes the 3-by-3 train/evaluation matrix for every configured LLM and evaluates SemEval for each training regime.

- [ ] **Step 1: Replace the notebook contract tests first**

Update the notebook tests so the source must contain:

```python
'LLM_MODELS = ["smollm2_360m"]'
'DATASETS = {"Kaggle": "kaggle_train", "Outfox": "outfox_chatgpt_train"}'
'CLASSIFIER_NAMES = ("logistic_regression",)'
'TRAINING_REGIMES'
'EVALUATION_REGIMES'
'combine_local_evidence_dataset_results('
'evaluate_local_evidence_semeval('
'"training_regime"'
'"evaluation_regime"'
```

Also assert that `gemma_3_1b_it`, `mage_train`, `gaussian_nb`, `gaussian_mixture_nb`, `pymc_bayesian_logistic`, PyMC configuration, stage-processing entry points, and fitted-model-cache flags are absent. Preserve the output-free assertions.

- [ ] **Step 2: Run the notebook contract tests and verify they fail**

Run:

```powershell
uv run pytest -q tests/test_cross_dataset_classification_poc_notebook.py
```

Expected: failures because the current notebook still contains Gemma, MAGE, and the old matrix.

- [ ] **Step 3: Rewrite the notebook configuration and imports**

Keep the existing environment-root setup and replace the parameter cell with:

```python
DATASETS = {"Kaggle": "kaggle_train", "Outfox": "outfox_chatgpt_train"}
LLM_MODELS = ["smollm2_360m"]
TRAINING_REGIMES = {
    "Kaggle": ("Kaggle",),
    "Outfox": ("Outfox",),
    "Mixed": ("Kaggle", "Outfox"),
}
EVALUATION_REGIMES = TRAINING_REGIMES
CLASSIFIER_NAMES = ("logistic_regression",)
WINDOW_LENGTHS = (32, 64, 128)
STRIDE = 16
TOP_K = 20
SEED = 42
RECOMPUTE_WINDOWS = False
```

Import only the reusable training/evaluation APIs, local-evidence constants, and plotting/display dependencies. Remove Bayesian/Gaussian-mixture settings and PyMC detection.

- [ ] **Step 4: Rewrite the experiment cell**

Implement one loop over `LLM_MODELS` that:

1. Loads each configured source dataset’s complete local-evidence cache and calls `split_local_evidence_texts(..., seed=SEED)` once.
2. Stores `train`, `validation`, and `test` cohorts by display name.
3. For each `training_regime`, maps display names to dataset aliases and passes the corresponding cohort mappings to `train_local_evidence_models(..., classifier_names=CLASSIFIER_NAMES, ...)`.
4. Evaluates each trained bundle on Kaggle and Outfox separately with `evaluate_local_evidence_datasets`.
5. Combines the two per-dataset results with `combine_local_evidence_dataset_results(..., evaluation_regime="Mixed")`.
6. Adds `llm` and `training_regime` columns to dataset metrics, predictions, scored windows, and coverage.
7. Loads the prepared SemEval annotations once and calls `evaluate_local_evidence_semeval` once per training regime, adding `llm` and `training_regime` to summary/evaluation/prediction outputs.
8. Displays coverage, classification metrics, and SemEval summaries without writing files.

Skip only unavailable source caches with a displayed message. Do not construct a training regime if one of its required source datasets is unavailable.

- [ ] **Step 5: Add matrix visualizations**

Create heatmaps from `classification_results` filtered to the configured aggregation, with rows indexed by `training_regime`, `classifier`, and `window_length`, columns indexed by `evaluation_regime`, and values `macro_f1`. Use `mask=matrix.isna()` and keep the figure output generated only when the notebook is explicitly executed.

- [ ] **Step 6: Run contract tests and validate notebook structure**

Run:

```powershell
uv run pytest -q tests/test_cross_dataset_classification_poc_notebook.py
uv run python -c "import json; from pathlib import Path; p=Path('notebooks/cross_dataset_classification_poc.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); assert all(c['execution_count'] is None and c['outputs'] == [] for c in n['cells'] if c['cell_type'] == 'code')"
```

Expected: all notebook contract tests pass and the JSON/output assertion exits successfully.

- [ ] **Step 7: Commit the notebook change**

```powershell
git add tests/test_cross_dataset_classification_poc_notebook.py notebooks/cross_dataset_classification_poc.ipynb
git commit -m "refactor: configure local evidence experiment matrix"
```

### Task 3: Run Full Verification

**Files:**
- Verify: `notebooks/cross_dataset_classification_poc.ipynb`, `notebooks/local_evidence_evaluation_utils.py`, and the two test files above.

**Interfaces:**
- Consumes: the committed implementation from Tasks 1 and 2.
- Produces: verified passing test suite and clean notebook/diff checks.

- [ ] **Step 1: Run focused local-evidence tests**

```powershell
uv run pytest -q tests/test_local_evidence_training_utils.py tests/test_local_evidence_evaluation_utils.py tests/test_cross_dataset_classification_poc_notebook.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full test suite**

```powershell
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and diff checks**

```powershell
uv run python -m compileall -q notebooks tests
git diff --check HEAD~2..HEAD
git status --short
```

Expected: compilation succeeds, diff check has no whitespace errors, and the working tree is clean after the implementation commits.

- [ ] **Step 4: Review the final matrix contract**

Confirm the committed notebook contains exactly one initial LLM (`smollm2_360m`), exactly Kaggle and Outfox source datasets, three training regimes, three evaluation regimes, configurable `CLASSIFIER_NAMES`, and SemEval evaluation for each trained regime.
