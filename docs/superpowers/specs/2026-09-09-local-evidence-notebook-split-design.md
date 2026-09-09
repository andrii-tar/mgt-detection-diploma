# Local-Evidence Notebook Split Design

## Goal

Split the local-evidence workflow from the existing cross-dataset proof of
concept into a reusable whole-text classification notebook and a separate,
deferred SemEval segmentation notebook.

## Scope

- Keep `notebooks/cross_dataset_classification_poc.ipynb` unchanged as the
  historical all-in-one and scalar-classification reference.
- Add exactly two output-free notebooks:
  - `notebooks/local_evidence_classification.ipynb`
  - `notebooks/local_evidence_segmentation.ipynb`
- Preserve the current `Mixed` and `Mixed+BBC` training regimes.
- Preserve BBC as HG-only and keep its train, validation, and test partitions
  disjoint.
- Do not run Stage 1 or Stage 2 processing from either notebook.

## Notebook Responsibilities

### Local-Evidence Classification

The classification notebook builds shared capped cohorts, trains local-evidence
window classifiers, selects document aggregation thresholds from validation
partitions only, and evaluates document classification on train, validation,
and test partitions. It retains the whole-text local-evidence outputs:

- document metrics, predictions, and coverage;
- window-level MG scores for inspection and token-score visualizations;
- BBC false-positive summaries.

It saves a reusable fitted bundle after the run completes. It does not create
SemEval region or boundary labels or metrics.

### Local-Evidence Segmentation

The segmentation notebook is SemEval-only. It takes an explicit artifact run
ID, loads its trusted fitted bundle, and rescores the current SemEval train and
dev MG local-evidence windows. It expands the fresh window scores to token
scores, derives HG/MG regions, and evaluates region and boundary metrics.

It never trains models, selects thresholds, modifies the source artifact, or
uses SemEval data for fitting or threshold selection.

## Artifact Contract

Artifacts live outside Git below:

```text
TEXT_DETECTION_DATA/experiment-results/local-evidence-classification/<run-id>/
  metadata.json
  model-cache/
  training_metrics.parquet
  training_coverage.parquet
  evaluation_metrics.parquet
  evaluation_predictions.parquet
  evaluation_coverage.parquet
```

`metadata.json` includes a schema version, scorer LLM, training regime and
source aliases, cohort seed and cap, classifier names, window lengths, stride,
top-k, feature columns, and a training-input fingerprint. The model cache uses
the existing validated fitted-model serialization and must not be unpickled
unless the caller explicitly passes `trust_artifact=True`.

Loading validates the schema and the requested artifact configuration before
reconstructing the `LocalEvidenceModelBundle`. Missing, corrupt, untrusted, or
incompatible artifacts raise clear errors rather than silently retraining or
loading stale models.

## Implementation Boundaries

Create `notebooks/local_evidence_artifact_utils.py` for artifact persistence
and loading. It owns external path construction, metadata validation, bundle
reconstruction, and Parquet result persistence. It reuses the existing
fitted-model cache validation instead of adding an unchecked serializer.

The notebooks remain orchestration and visualization only; reusable behavior
lives in Python modules. Existing local-evidence training, document evaluation,
and SemEval evaluation utilities remain their respective sources of truth.

## Testing

- Test trusted artifact round-trip reconstruction of classifiers and decision
  rules.
- Test that loading without trust does not unpickle the saved model.
- Test schema/configuration/fingerprint mismatch rejection.
- Add notebook-source contracts for output-free state, focused notebook roles,
  no segmentation in classification, and no training in segmentation.
- Run focused artifact/notebook tests and the full pytest suite.

## Non-Goals

- Do not modify or remove the existing cross-dataset POC.
- Do not generalize segmentation to datasets other than SemEval.
- Do not persist SemEval scored-window caches; segmentation rescores current
  windows with the saved bundle.
- Do not add Random Forest to the scalar classifier in this change.
