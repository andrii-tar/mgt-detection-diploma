# Mixed Local-Evidence Cross-Dataset Evaluation

## Goal

Adapt `notebooks/cross_dataset_classification_poc.ipynb` to use the local-evidence approach from `notebooks/bayesian_model_comparison.ipynb`. For each processing model, train classifiers on a mixed training pool containing Kaggle, Outfox, and MAGE, then evaluate on each dataset's isolated test partition.

The processing models are the canonical aliases `gemma_3_1b_it` and `smollm2_360m`.

## Scope

- Modify only the cross-dataset classification notebook and its notebook contract tests.
- Read existing processed artifacts; do not run stage 1 or stage 2 processing.
- Use the existing local-evidence window cache and model utilities.
- Keep notebook cells output-free and do not commit generated data or model caches.
- Evaluate four available classifiers: logistic regression, Gaussian NB, Gaussian Mixture NB, and PyMC Bayesian logistic when PyMC is installed.

## Data Flow

1. For each processing model and source dataset, load or build local-evidence windows with the configured window lengths, stride, and top-k settings.
2. Split each dataset's windows into deterministic train, validation, and test partitions using the existing local-evidence splitter.
3. Concatenate the three dataset-specific training partitions into a `Mixed` training partition. Validation data remains available for future model selection but is not mixed into test evaluation.
4. Compute a training-input fingerprint for the mixed windows and partitions, allowing the existing fitted-model cache contract to distinguish experiments.
5. Fit each available local-evidence classifier separately for each processing model and window length on the mixed training partition.
6. Score each fitted classifier on the untouched test partition of Kaggle, Outfox, and MAGE independently.

The processing models are treated as separate feature-generation backends. Their windows and fitted classifiers are never combined with one another.

## Notebook Outputs

The notebook will display:

- Artifact coverage by processing model, dataset, and split.
- Mixed training/test partition sizes.
- A result table containing processing model, classifier, window length, evaluation dataset, training rows, evaluation rows, accuracy, and macro-F1.
- One heatmap per processing model, with classifier/window length on rows and evaluation dataset on columns, showing macro-F1.

The test partitions remain dataset-specific so the results measure cross-dataset transfer from the mixed training pool rather than performance on a pooled test set.

## Error Handling

- Skip dataset/model combinations that lack both labels or cannot produce stratified train/validation/test partitions.
- Skip a processing model if no dataset has valid local-evidence windows.
- Raise a clear error if no mixed training partition can be formed or if no evaluation results are produced.
- Exclude PyMC from the available classifier list when the optional dependency is not installed, matching the Bayesian notebook behavior.

## Testing

Add static notebook contract assertions that verify:

- Both canonical processing model aliases are configured.
- Kaggle, Outfox, and MAGE are loaded.
- Local-evidence loading, partitioning, model comparison, and model caching are used.
- The three training partitions are concatenated into `Mixed`.
- Evaluation is performed separately for each dataset's test partition.
- All supported classifier names and macro-F1 heatmap output are present.
- The notebook remains output-free and does not invoke processing entry points.

No large model or data-processing run is required for verification. Run the notebook contract tests and the full pytest suite.
