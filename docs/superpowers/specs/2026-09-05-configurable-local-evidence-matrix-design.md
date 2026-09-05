# Configurable Local-Evidence Experiment Matrix

## Goal

Refactor `notebooks/cross_dataset_classification_poc.ipynb` into a configurable
experiment notebook that evaluates local-evidence classification for one or
more LLM processing models. The initial configuration uses only
`smollm2_360m`, Kaggle, and Outfox, while keeping the classifier selection
extensible.

## Experiment Matrix

The notebook will define these source datasets:

- Kaggle: `kaggle_train`
- Outfox: `outfox_chatgpt_train`

Each dataset is split independently and deterministically into train,
validation, and test text cohorts. The `Mixed` regime is the union of the
corresponding Kaggle and Outfox cohorts; it is not re-split.

For every configured LLM, the notebook trains three models:

- `Kaggle`: Kaggle train and validation cohorts
- `Outfox`: Outfox train and validation cohorts
- `Mixed`: the union of Kaggle and Outfox train and validation cohorts

Each trained model is evaluated in three regimes:

- `Kaggle`: Kaggle test cohort
- `Outfox`: Outfox test cohort
- `Mixed`: the union of Kaggle and Outfox test cohorts

This produces the complete 3-by-3 train/evaluation transfer matrix.

## Model Configuration

The default classifier configuration is:

```python
CLASSIFIER_NAMES = ("logistic_regression",)
```

The notebook passes this tuple to the existing reusable training function, so
future classifiers can be enabled by changing configuration rather than
rewriting the experiment loop. Existing local-evidence feature columns,
window lengths, stride, top-k, seed, and cache recomputation controls remain
configurable.

Validation cohorts determine document decision rules. Test cohorts are used
only for final evaluation.

## Architecture

The notebook remains an orchestration layer. It will reuse:

- `train_local_evidence_models` for fitting classifiers and calibrating rules.
- `evaluate_local_evidence_datasets` for document-level classification.
- `evaluate_local_evidence_semeval` for SemEval region evaluation.
- `load_or_build_local_evidence_windows` and
  `split_local_evidence_texts` for cache access and deterministic cohorts.

The evaluation boundary will receive only the minimal support needed to
combine the two independently loaded dataset results under an explicit
`evaluation_regime="Mixed"` label while retaining dataset-specific results.

## Outputs

Classification results will include:

- `llm`
- `training_regime`
- `evaluation_regime`
- `classifier`
- `window_length`
- aggregation and threshold settings
- classification metrics

SemEval results will include the LLM, training regime, classifier, window
length, and region-level metrics. The notebook will display result tables and
heatmaps grouped by training and evaluation regime.

The notebook will remove the current Gemma, MAGE, Bayesian, Gaussian-mixture,
and fitted-model-cache configuration because these are outside the initial
experiment scope.

## Failure Handling

Unavailable local-evidence caches will skip only the affected dataset/model
combination where possible. A training regime with no available source data
will be reported and excluded from evaluation. The notebook will not run stage
1 or stage 2 processing jobs.

## Testing

Update notebook contract tests to verify:

- only `smollm2_360m` is configured;
- Kaggle and Outfox are the source datasets;
- all three training and evaluation regimes are represented;
- logistic regression is configurable through a classifier tuple;
- SemEval is evaluated for each training regime;
- removed models and processing entry points are absent;
- notebook code cells have no stored outputs or execution counts.

Run focused local-evidence tests and the full `uv run pytest -q`, followed by
notebook JSON/compile checks and `git diff --check`.
