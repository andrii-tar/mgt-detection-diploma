# Bayesian Model Comparison Design

## Goal

Add a comparison workflow that evaluates the existing logistic-regression local-evidence classifier alongside Gaussian Naive Bayes and Bayesian logistic regression implemented with PyMC. The comparison must use the same local-evidence features, parent-text partitions, validation threshold selection, test data, and document-level aggregations for every model.

The existing `local_evidence_classification.ipynb` workflow remains intact. A new comparison notebook will make it possible to compare the models on the complete test partition and inspect several individual texts with the existing heatmaps and combined local-evidence plots.

## Scope

In scope:

- A reusable model-adapter layer for deterministic and Bayesian classifiers.
- Logistic regression as the current baseline.
- GaussianNB as the scikit-learn comparison model.
- PyMC Bayesian logistic regression using ADVI as the default inference method.
- Optional PyMC dependency and runtime availability handling.
- Shared window scoring, threshold selection, aggregation, accuracy, and macro F1 evaluation.
- A new output-free comparison notebook.
- Regression and smoke tests for the adapters, evaluation contract, and notebook structure.

Out of scope:

- Replacing logistic regression in the existing notebook.
- Hierarchical models across texts or window lengths.
- NUTS as the default inference method.
- New raw-data processing or model-weight generation.
- Large production-data runs as part of automated tests.

## Current Context

`notebooks/local_evidence_utils.py` builds 36 continuous local-evidence features from four metrics and nine window statistics. `fit_local_evidence_model` currently fits an imputed, standardized, class-balanced logistic regression for each window length. `run_local_evidence_experiments` scores validation and test windows, selects document thresholds on validation data, evaluates maximum-window and evidence-fraction aggregations, and returns summaries, predictions, and scored windows.

The comparison should reuse this data and leakage boundary rather than create a second feature-building or partitioning implementation.

## Model Adapters

Create a focused reusable module, `notebooks/local_evidence_models_utils.py`, containing a common adapter contract and model-specific implementations. Keep feature construction, parent partitioning, aggregation, and threshold selection in the existing local-evidence utilities unless a small shared helper is needed to support the adapters.

Each adapter is fit independently for one window length and must support:

```text
fit(train_windows, feature_columns, seed) -> fitted model
predict(fitted model, windows) -> scored windows
```

The scored output must contain the original window identity columns plus:

- `mg_score`: probability of the MG class, bounded to [0, 1].
- `model`: stable model identifier.
- `mg_score_lower` and `mg_score_upper` when posterior predictive uncertainty is available; otherwise missing values.

The adapter must not fit on validation or test rows. Any preprocessing state must be learned from training rows only.

### Logistic Regression

Retain the existing behavior as the deterministic baseline:

- Median imputation.
- StandardScaler.
- Class-balanced `LogisticRegression` with the existing seed and iteration settings.
- `predict_proba()[:, 1]` as `mg_score`.

The comparison adapter may wrap or reuse the current implementation, but the existing local-evidence behavior and output contracts must not regress.

### GaussianNB

Use scikit-learn `GaussianNB` for the continuous real-valued window statistics:

- Median imputation learned from the training partition.
- No artificial non-negative transformation and no multinomial-count assumption.
- Fit `GaussianNB` on the imputed training features.
- Use `predict_proba()[:, 1]` as `mg_score`.

`MultinomialNB` is deliberately excluded because the local features include signed and continuous values rather than counts or frequencies.

### Bayesian Logistic Regression

Use PyMC to define a binary logistic model over the standardized local-evidence features:

```text
coefficient_j ~ Normal(0, prior_scale)
intercept ~ Normal(0, intercept_scale)
mg_probability = sigmoid(intercept + X @ coefficients)
label ~ Bernoulli(mg_probability)
```

Use `Normal(0, 1)` priors for standardized feature coefficients and `Normal(0, 2.5)` for the intercept. These defaults must be explicit, stable, and configurable in one place. The default inference method is ADVI with a fixed random seed and 10,000 iterations, both configurable for notebook runs and smoke tests. The fitted adapter will retain the posterior approximation and standardized preprocessing state.

Prediction behavior:

- Draw 1,000 posterior parameter samples, or a configured smaller number for smoke tests, and evaluate the latent MG probability for each scored window.
- Use the posterior mean MG probability as `mg_score`.
- Use the 2.5th and 97.5th quantiles of those latent probabilities for `mg_score_lower` and `mg_score_upper`.
- Keep posterior coefficient summaries available for the comparison notebook, including posterior mean, 95% credible interval, and posterior probability of a positive coefficient.

PyMC must be imported lazily. If it is not installed, LR and GaussianNB remain usable. The comparison notebook must show a clear availability message and omit the PyMC rows rather than failing during import. A PyMC-specific smoke test may be skipped with an explicit reason when the optional dependency is unavailable.

## Shared Evaluation

For every model and configured window length:

1. Split parent texts once with `split_local_evidence_texts`.
2. Fit only on the training parent-text partition.
3. Score validation and test windows.
4. Select document thresholds and score direction using validation predictions only.
5. Evaluate test predictions using the existing maximum-window and evidence-fraction aggregations.
6. Record accuracy and macro F1 for each model, window length, and aggregation.

The common summary schema should include model identity, window length, aggregation, threshold information, support counts, accuracy, macro F1, and any validation selection metrics. The test result table must remain long-form so it can be filtered and pivoted by model and window length.

Overall comparison means metrics over all eligible test texts/windows represented by each document-level aggregation, not metrics averaged from selected individual examples. Per-text results must remain available for visualization and error inspection.

## Comparison Notebook

Add an output-free notebook named `notebooks/bayesian_model_comparison.ipynb`. Reusable logic must remain in version-controlled Python modules; notebook cells should configure parameters, run the comparison, display tables, and create plots.

The notebook will:

- Load or build the same cached local-evidence windows used by the existing workflow.
- Configure dataset, causal-LM model, window lengths, seed, and PyMC ADVI settings.
- Run LR and GaussianNB unconditionally.
- Run PyMC when the optional dependency is available.
- Display overall test accuracy and macro F1 tables grouped by model, window length, and aggregation.
- Display per-text prediction/error tables for selected models and window lengths.
- Select several configurable texts from the test partition without changing the training split.
- Reuse the existing token logprob, candidate-similarity, and coverage heatmaps for each selected text.
- Add a model-score heatmap showing model/window scores across token positions or local windows.
- Add a reusable combined normalized plot with the existing token metrics and one local-evidence score line per classifier.
- Show PyMC posterior predictive bands when available.
- Keep every code cell output-free before commit.

The notebook must not run large model inference or raw-data processing implicitly. It may build the local-evidence cache when explicitly configured to recompute, matching the existing notebook behavior.

## Testing

Add focused tests for:

- Adapter output schema and probability bounds.
- Training-only preprocessing and parent-text leakage prevention.
- GaussianNB handling of signed continuous features.
- Shared evaluation producing comparable accuracy and macro F1 rows for all available models.
- PyMC ADVI smoke fitting on a small synthetic/local fixture when PyMC is installed.
- Optional PyMC behavior when PyMC is unavailable.
- Posterior interval columns and valid interval ordering when Bayesian predictions are present.
- Comparison notebook imports, required model names, required metric/plot calls, and empty outputs.

Use small fixtures and low ADVI iteration counts in tests. Do not run production data or large PyMC inference in the test suite.

## Dependency Decision

PyMC should be added under a separate optional `bayesian` dependency extra rather than the base runtime or default notebook dependencies. The extra should declare a supported PyMC 5 release. The default LR and GaussianNB comparison must remain runnable without installing this extra.

## Success Criteria

- One comparison notebook can run LR and GaussianNB without PyMC installed.
- When PyMC is installed, the same notebook compares all three models using the same partitions and evaluation protocol.
- Whole-test accuracy and macro F1 are directly comparable by model, window length, and aggregation.
- Several selected texts can be inspected through existing heatmaps and combined model plots.
- Bayesian coefficient and prediction uncertainty is visible without changing deterministic model scores.
- Existing local-evidence tests and behavior remain passing.
- The committed notebook contains no stored outputs.
