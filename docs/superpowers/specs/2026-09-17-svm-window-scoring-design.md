# SVM Window Scoring Design

## Goal

Add support-vector-machine classifiers to the feature-vector window scoring
experiment so their results can be compared with logistic regression and random
forest classifiers without changing existing model behavior.

## Model Families

The model-family registry will contain these additional explicit names:

- `svm_linear`
- `svm_rbf`
- `svm_poly`
- `svm_sigmoid`

Each name identifies one independently trained and persisted classifier. It is
recorded in artifact metadata and in every evaluation result, allowing existing
artifact discovery, metric grouping, and result analysis to distinguish kernel
results without a new metadata field.

## Training And Scoring

Each SVM model uses a sklearn `Pipeline` containing:

1. Median `SimpleImputer` with `keep_empty_features=True`.
2. `StandardScaler`.
3. `sklearn.svm.SVC` configured with the corresponding kernel,
   `probability=True`, and `class_weight="balanced"`.

All other `SVC` settings retain sklearn defaults. In particular, `C=1.0`,
`gamma="scale"`, polynomial `degree=3`, and the default coefficient settings
are not tuned or overridden. `probability=True` is required because the shared
window scoring interface emits an MG probability through `predict_proba` and
applies its existing `0.5` prediction threshold.

Logistic regression remains imputed and scaled. Random forest remains imputed
without scaling. No change is made to feature validation, label validation,
artifact format, or score thresholding.

## Experiment Integration

The training notebook takes its model families from the shared registry, so it
will train all six registered classifier families. The evaluation notebook also
uses the shared registry, so it will discover, evaluate, and export each SVM
family using the existing metrics, predictions, coverage, run-summary, and
SemEval output formats.

Each SVM artifact is stored under its own `models/{model_family}/...` directory.
No large training or evaluation run is part of this implementation.

## Error Handling

Unknown model-family names continue to raise `ValueError` before fitting or
loading artifacts. Existing metadata validation accepts only registered model
families. The existing imputation, numeric-feature validation, exact feature
schema checks, and trusted-artifact loading requirements apply to SVMs unchanged.

## Tests And Verification

Extend model-training tests to cover each SVM family. Tests will verify that:

- each family maps to an `SVC` with the expected kernel;
- the estimator retains `probability=True` and `class_weight="balanced"`;
- the SVM pipeline includes imputation and standard scaling;
- scoring produces MG probabilities in `[0, 1]` and binary predictions;
- an SVM artifact round-trip preserves its family and estimator type;
- unknown families remain rejected.

Verification will run the focused model-training test module, the full pytest
suite, and a source-level notebook configuration check. It will not execute
large model training or evaluation jobs.
