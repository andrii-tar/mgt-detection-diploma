# Random Forest Classifier Comparison Design

## Goal

Add a random-forest classifier as a peer to the existing feature-vector logistic-regression classifier. The training notebook will train, persist, and evaluate both model families for every configured feature set and window configuration.

## Scope

- Keep `logistic_regression` unchanged. It continues to use scikit-learn's default L2 regularization.
- Add `random_forest` to the supported model-family registry and notebook model-family selection.
- Use identical prepared features, parent-text cohorts, train/validation partitions, metrics, artifact metadata, and downstream artifact discovery for both families.
- Use a deterministic `RandomForestClassifier` with the configured cohort seed and balanced class weights.
- Apply median imputation to random-forest inputs, but do not standardize them.
- Preserve the existing artifact version and schema because model-family-specific pipeline contents remain encapsulated in the saved estimator.

## Implementation Boundaries

`notebooks/utils/model_training_utils.py` remains the single interface for fitting and scoring feature-vector classifiers. It will select the preprocessing and estimator pipeline by `model_family`:

- Logistic regression: median imputation, standard scaling, existing balanced L2 logistic regression.
- Random forest: median imputation and a balanced random forest without scaling.

`notebooks/prepare_and_train_classifiers.ipynb` will derive its `MODEL_FAMILIES` from the utility registry so the new family participates in the existing training/resume/export loop. No separate notebook path or result schema is required.

`notebooks/evaluate_classifiers.ipynb` will recognize `random_forest` through the same model-family registry, allowing saved random-forest artifacts to be evaluated by the existing workflow.

## Data Flow

For each scorer model, feature set, and training window, the existing notebook loads prepared feature vectors and fixed cohorts. It fits both model families against the same training frame, scores the same validation frame, writes the same metrics, and saves separate artifacts under model-family-specific paths. Downstream evaluation discovers each artifact and reports its results with `model_family=random_forest` or `model_family=logistic_regression`.

## Error Handling

The current validation for supported model families, required feature columns, numeric features, binary labels, artifact metadata, and trusted artifact loading remains unchanged. Unknown model families continue to fail before fitting. Existing logistic-regression artifacts remain valid and loadable.

## Verification

Add or extend unit tests for feature-vector training utilities to confirm that random forest:

- Is an accepted model family.
- Fits and produces bounded MG probabilities and binary predictions.
- Uses imputation and does not include a scaling pipeline step.
- Can be saved and loaded with metadata validation.

Run the targeted tests and the relevant classifier utility test suite. Do not run data processing or model inference jobs.
