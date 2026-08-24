# Kaggle Baseline Classification Design

## Goal

Validate the repository's processed-metrics-to-classification path with one
simple, reproducible HG/MG baseline on `kaggle_train`.

The baseline uses existing processed metric files only. It does not run model
inference, train a language model, or persist experiment outputs.

## Scope

Add:

- `notebooks/classification_utils.py` for reusable loading, validation,
  splitting, fitting, and evaluation functions.
- `notebooks/text_classification.ipynb` as a thin, output-free notebook that
  configures and runs the baseline, displays result tables, and renders plots.
- Focused pytest coverage for the reusable helper functions.
- `scikit-learn` as a project dependency required by the helper and notebook.

Do not add Random Forest, SVM, token-similarity features, cross-validation,
multi-dataset evaluation, persisted results, or a command-line interface in
this iteration.

## Data Selection

The helper discovers eligible `kaggle_train` metric exports under:

```text
processed/{dataset}__{model}/{split}/metrics/metrics.parquet
```

It selects the scorer model with the greatest balanced raw coverage: maximize
the smaller of its HG and MG row counts, then total row count, then model alias
for a deterministic tie break. The notebook displays the selected model alias
and raw HG/MG counts. A notebook parameter can override the dataset or
scorer-model selection for later experiments.

Metric rows use their existing `split` values as labels:

- `HG = 0`
- `MG = 1`

The initial feature candidates are:

- `prompt_mu_score`
- `prompt_sigma_score`
- `perplexity`
- `d_score_top20`
- `d_score_top20_coverage`
- available text-length controls

No token-similarity aggregate feature is loaded in this baseline. It is
deferred until the scalar-metric path is validated.

## Experiments

After selecting the scorer model, the helper builds one canonical cohort by
retaining rows with finite values for every available scalar feature. It reports
raw and retained HG/MG counts. All baseline experiments use this same cohort
so their test results are comparable.

The helper creates reproducible stratified train/validation/test partitions of
60/20/20 with `random_state=42`.

The notebook runs two baseline experiment types:

1. Single-feature threshold baselines. Each available scalar feature is used
   alone. The helper tests both score directions, selects the direction and
   threshold that maximize validation-set macro-F1, then freezes them.
2. A multi-feature Logistic Regression baseline. It uses all available scalar
   features, median imputation, standard scaling, `class_weight="balanced"`,
   and a fixed random seed. Its operating threshold is 0.5.

All Logistic Regression preprocessing is fit from training data only. Threshold
direction and value are fit from validation scores only. The test partition is
not used to select a threshold, model, or feature set; the notebook compares
the predeclared baseline experiments descriptively rather than claiming that
the highest test score is selected for deployment.

## Evaluation And Visuals

Each frozen experiment is evaluated once on the untouched test partition.
Results include:

- ROC-AUC and PR-AUC using the continuous MG score
- accuracy and balanced accuracy
- macro-F1 and weighted F1
- HG and MG precision, recall, F1, and support
- confusion-matrix counts
- selected threshold and score direction where applicable

The notebook displays a tidy test-results table, a per-experiment confusion
matrix, and combined ROC and precision-recall curves. Every output labels MG
as the positive class (`1`).

## Interfaces And Errors

Helper functions accept a processed-data root, dataset/model selection, and
feature columns. They raise clear errors when no metric exports exist, the
requested dataset/model has no HG/MG coverage, no requested features are
available, or either label lacks sufficient rows for a stratified split.

The notebook exposes parameters for the data root, dataset alias,
model-selection override, partition fractions, and seed. It fails before
fitting if the selected dataset/model cannot support the requested stratified
partitions.

## Tests

Tests use temporary Parquet fixtures and synthetic feature frames to cover:

- scorer-model balanced-coverage discovery and deterministic selection;
- label mapping and canonical-cohort validation;
- train/validation/test split reproducibility and label preservation;
- direction-aware threshold fitting using validation data only;
- Logistic Regression baseline output shape and required evaluation fields;
- metric behavior for known HG/MG predictions.

The implementation is verified with the focused new tests and the repository
test suite. Notebook cells remain output-free in version control.
