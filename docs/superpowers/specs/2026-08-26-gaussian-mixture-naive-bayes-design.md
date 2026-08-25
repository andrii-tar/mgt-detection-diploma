# Gaussian Mixture Naive Bayes

## Goal

Add a `gaussian_mixture_nb` classifier that produces one window-level `mg_score` probability from the 36 local-evidence features by modeling each feature independently with class-conditional Gaussian mixtures.

## Model

For every feature and class (`HG` and `MG`), fit a one-dimensional Gaussian mixture with a configurable number of components. The default is two components per class.

For a window with feature vector `x`, calculate:

```text
log p(x | class) = sum_feature log p(x_feature | class)
```

Using class priors estimated from the training labels:

```text
mg_score = exp(log p(x | MG) + log p(MG))
           / [exp(log p(x | MG) + log p(MG))
              + exp(log p(x | HG) + log p(HG))]
```

The implementation must use numerically stable log-sum-exp normalization. The result is a probability in `[0, 1]`, compatible with the existing scoring and comparison pipeline.

## Configuration

Add a configuration dataclass or fields for:

- `components`: default `2` Gaussian components per class and feature.
- `random_seed`: inherited from the model comparison seed.
- `reg_covar`: positive covariance regularization passed to each `GaussianMixture`.

The default configuration must be deterministic and robust on small feature/class samples. Invalid component counts or covariance regularization values raise `ValueError` before fitting.

## Integration

- Add `gaussian_mixture_nb` to `MODEL_NAMES`.
- Keep `logistic_regression`, `gaussian_nb`, and optional `pymc_bayesian_logistic` behavior unchanged.
- Fit only rows in the training partition when partitioned input is supplied.
- Impute missing feature values using the existing training-only median imputer before fitting mixtures.
- Store one fitted mixture per `(feature, class)` and the training class log priors.
- Score windows with the same output schema: `model`, `mg_score`, `mg_score_lower`, and `mg_score_upper`. Deterministic mixture scores have NaN uncertainty bounds.
- No new dependency is required; use `sklearn.mixture.GaussianMixture` and existing NumPy/scikit-learn dependencies.

## Behavior on Small Classes

If a class has fewer training rows than the requested component count, fit the maximum feasible number of components for that class, with at least one component. This keeps the default model usable on small smoke-test fixtures while preserving two components when sufficient data exists. The effective component count should be inspectable on the fitted estimator.

## Testing

Add tests covering:

- `gaussian_mixture_nb` is part of the model registry.
- Fitting and scoring produce bounded window-level MG probabilities and the common output schema.
- The model fits one-dimensional mixtures per feature and class, with two components when enough training rows exist.
- Small class samples reduce effective components without failing.
- Scores are deterministic for a fixed seed.
- Training-only imputation and fitting are preserved when validation/test rows are mutated.
- Invalid configuration is rejected.
- Existing model tests and model comparison output continue to pass with the expanded registry.

## Out of Scope

- Full-covariance multivariate QDA in this change.
- Combining the 36 feature scores with a second-stage regression.
- Per-feature chart changes; the existing same-text plotting work can display the resulting window-level scores.
- Persisting fitted mixture models.
