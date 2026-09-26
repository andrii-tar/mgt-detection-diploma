# GPT-who-Style Baseline Notebook Design

## Goal

Create an isolated exploratory notebook that tests whether GPT-who-style
surprisal and local-uniformity features provide a useful binary human-versus-
machine baseline when calculated from the project's existing small-LM Stage 1
logprob outputs.

The experiment must not modify the production classifier-training notebook or
register model artifacts in the main classification pipeline. Successful
results can later be integrated in a separate change.

## Notebook

Create `notebooks/try_gpt_who_baseline.ipynb`.

The notebook will:

1. Read configuration from `TEXT_DETECTION_DATA`, with the repository's local
   default used by existing notebooks.
2. Load Stage 1 token-logprob Parquet rows from the configured dataset and
   scorer model. The scorer model is configurable and defaults to
   `smollm2_360m`.
3. Derive a compact GPT-who-style feature vector from each text.
4. Fit balanced logistic regression on the derived features.
5. Evaluate held-out data using the project's binary-classification metrics.
6. Save tabular results and plots below the external data root.

All code cells must be cleared of stored outputs and execution counts before
committing the notebook.

## Feature Definition

For each text, use finite observed-token log probabilities and define

`surprisal = -observed_logprob`.

The feature vector will include:

- mean surprisal;
- surprisal variance;
- mean absolute difference between consecutive surprisals;
- mean squared difference between consecutive surprisals;
- minimum and maximum 20-token-window surprisal variance;
- minimum and maximum 20-token-window mean absolute consecutive difference;
- minimum and maximum 20-token-window mean squared consecutive difference;
- valid token count.

Window statistics use a 20-token window, matching the GPT-who local-span
design. Texts with fewer than 20 valid tokens will not be silently padded or
treated as equivalent evidence. They will be excluded from the window-based
experiment and reported in a coverage table. Invalid or missing logprob rows
will be reported and excluded rather than imputed at the token level.

The feature-column order will be fixed and recorded in the result metadata.
This is an adapted feature set, not a claim to reproduce GPT-who's exact 44
features.

## Data Splits and Evaluation

The notebook will use the configured `kaggle_train` dataset by default and
preserve parent-text identity when constructing cohorts. The scorer model can
be changed without modifying notebook logic. Each parent text contributes one
feature row; the 20-token windows are used only to calculate features, not as
separate training examples. It will use deterministic seeding and avoid
placing the same parent text in both training and evaluation partitions.

The primary classifier is balanced `LogisticRegression`, with the same binary
label convention already used by the project: HG = 0 and MG = 1.

Reported metrics will include:

- sample count and class counts;
- accuracy;
- balanced accuracy;
- macro-F1;
- ROC-AUC when both classes have valid scores;
- confusion matrices;
- per-dataset and per-scorer coverage.

The first pass will evaluate only the adapted GPT-who-style feature vector.
Comparison with `scalar_compact` is deferred until the baseline has produced
valid results. This keeps the experiment focused and avoids mixing cache
availability problems with baseline validation.

## Outputs

Write experiment results below an external-data results directory, not inside
the repository's tracked data area. The outputs will include:

- feature coverage and exclusion summary;
- per-scorer and per-dataset evaluation metrics;
- fitted-model coefficients with feature names;
- confusion-matrix and score-distribution plots;
- a run configuration containing scorer, dataset, window size, seed, and
  feature-column order.

The notebook will not download models, run the Stage 1 pipeline, or require
new raw data. It will fail with a clear availability message when the
requested Stage 1 Parquet inputs are absent.

## Testing and Validation

Add focused tests for the reusable UID feature calculation logic, preferably
in a version-controlled utility module rather than embedding non-trivial
functions in notebook cells. Tests will cover:

- exact feature values on a small synthetic token sequence;
- 20-token window boundary behavior;
- short-text exclusion;
- invalid/missing logprob handling;
- deterministic feature-column order;
- both HG and MG labels in the classifier input.

Validate the notebook with JSON/code syntax checks, repository notebook
contract checks, `git diff --check`, and a small smoke run when external Stage
1 data are available.

## Non-Goals

- Reproducing GPT-who's original GPT2-XL experiment exactly.
- Comparing against `scalar_compact` or other existing feature sets in this
  first notebook.
- Adding `gpt_who_uid` to the production feature-set registry.
- Changing `prepare_and_train_classifiers.ipynb` or existing evaluation
  notebooks.
- Adding a new model alias or downloading model weights.
- Claiming that the adapted features are equivalent to the published 44-feature
  implementation.
