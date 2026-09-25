# Classifier Run ID Design

## Goal

Give the controlled classifier experiment a new run ID so its outputs cannot be
confused with artifacts produced by the earlier classifier configuration.

## Design

Use `feature-vector-classification-2026-09-25-logreg-random-forest-max600-min100`
for training and standard evaluation. This identifies the experiment date, the
active model families, the 600-text cap, and the 100-token minimum.

Use that value as `SOURCE_RUN_ID` for bounded-prefix evaluation, while keeping a
separate report run ID: `bounded-prefix-classifier-evaluation-window32-2026-09-25`.
Update downstream analysis notebooks that load the classifier results to use the
new source run ID.

## Verification

Run the notebook contract tests covering training, evaluation, bounded-prefix,
and downstream analysis run-ID references. Review the resulting diff and commit
the design note and notebook updates together.
