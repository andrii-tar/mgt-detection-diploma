# Bounded-Memory Classifier Evaluation Design

## Goal

Evaluate arbitrarily large prepared feature partitions without loading a complete
dataset or the accumulated detailed prediction history into memory. Preserve one
exact metric and coverage row per evaluation artifact, dataset, and partition.

## Scope

This changes feature-vector classifier evaluation. Model training remains
in-memory because the supported estimators train on the complete selected cohort.

## Prediction Artifacts

Detailed predictions become the primary, partitioned evaluation output. A
completed artifact is stored at:

```text
experiment-results/classification/evaluations/{run_id}/predictions/
  {artifact_id}/{evaluation_regime}/{dataset}/{partition}/
    data.parquet
    metadata.json
```

`artifact_id` is a deterministic, filesystem-safe identifier for the model
artifact. Metadata records the complete evaluation key, source artifact path,
model metadata required by existing exports, and the persisted row count.

`evaluation_predictions.csv` is no longer the primary detailed-result store.
Analysis code discovers partitioned prediction artifacts only when detailed rows
are needed. Existing compact `evaluation_metrics.csv`, `evaluation_coverage.csv`,
and `run_summary.csv` exports remain.

## Score Phase

For each eligible model artifact, evaluation regime, dataset, and partition:

1. Resolve and validate prepared-partition metadata without reading its complete
   `data.parquet` file.
2. Read the required Parquet columns using `pyarrow.parquet.ParquetFile`
   record batches. `EVALUATION_BATCH_SIZE` controls the maximum source rows in a
   batch.
3. Filter each batch to the selected cohort, score it, attach evaluation
   dimensions, and write it through a temporary `ParquetWriter`.
4. Release the source and scored batch before reading the next batch.
5. Validate the written row count and atomically publish `data.parquet` with
   `metadata.json` only after all batches succeed.

The published prediction artifact is the score-phase completion marker. A
temporary or incomplete directory is never considered complete and is removed
before retrying the same evaluation key.

## Metric Phase

After a prediction artifact is published:

1. Scan only `label` and `mg_score` from that artifact.
2. Persist those values in a temporary on-disk SQLite table for the current
   evaluation key.
3. Derive threshold metrics from aggregate counts and derive exact ROC-AUC and
   average precision from disk-backed score ordering, including tied scores.
4. Write one metrics row and one coverage row after aggregation succeeds.
5. Drop the temporary table in `finally`.

Metric calculations must match the current `evaluate_scores` behavior for the
same labels, scores, and threshold. A source batch with one label is valid; the
complete artifact determines whether AUC metrics are defined.

## Resume And Failure Behavior

- Score resume checks for a valid prediction artifact and matching metadata.
- Metric resume checks compact metric and coverage exports for the evaluation
  key.
- A failed score phase publishes no prediction artifact and emits the existing
  unavailable run-summary record.
- A failed metric phase leaves the valid prediction artifact available for a
  later metric-only retry and emits an unavailable run-summary record.
- No partial detailed predictions are appended to final result exports.

## Testing

Tests must verify that:

- Multiple record batches yield the same detailed predictions, metrics, and
  coverage as current in-memory evaluation.
- Tied scores and source batches with only one label preserve exact
  dataset-level ROC-AUC and average precision behavior.
- Incomplete or mismatched prediction artifacts are ignored by resume checks.
- The streaming reader does not call full-partition `pd.read_parquet`.
- Notebook structure uses the score phase, metric phase, and explicit batch
  release rather than an append-only detailed-prediction CSV.
