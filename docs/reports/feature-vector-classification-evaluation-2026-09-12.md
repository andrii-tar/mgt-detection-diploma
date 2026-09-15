# Feature-Vector Classification Evaluation Report

## Executive Summary

This report analyses run `feature-vector-classification-2026-09-12` from
`notebooks/evaluate_classifiers.ipynb`. The experiment compares four feature
sets, four scorer LLMs, training windows of 32/64/128 tokens, evaluation windows
of 32/64/128 tokens plus whole-text evaluation, and the configured classifier
family.

The strongest overall result is the `simple_similarity` feature set. Under
Mixed training and Mixed evaluation, its mean test macro F1 across Kaggle,
MAGE, and Outfox is **0.793**, compared with **0.661** for `extended`, **0.525**
for `scalar_all`, and **0.477** for `scalar_compact`. This is a substantial
advantage for the similarity-only representation in the current experiment.

The best scorer by the same aggregate is `smollm2_360m` at **0.649** mean test
macro F1, followed by `littlelamb` at **0.607**, `gemma_3_270m_it` at **0.605**,
and `gemma_3_1b_it` at **0.595**. The difference between scorer LLMs is smaller
than the difference between feature sets.

Longer training windows help: mean test macro F1 rises from **0.533** at 32
tokens to **0.596** at 64 and **0.653** at 128. For evaluation, 128-token
windows are best on average for `simple_similarity` (**0.803**), while
`extended` peaks at 64 (**0.719**). Window length is therefore not a neutral
parameter and should be selected separately for each representation.

The best observed Mixed-evaluation configuration is `smollm2_360m` with
`extended`, trained at 64 tokens and evaluated at 128 tokens: **0.843** mean
test macro F1 on the three source datasets. Its source-specific scores are
0.981 on Kaggle, 0.679 on MAGE, and 0.870 on Outfox. The best same-length
configuration is `littlelamb` or `smollm2_360m` with `extended`, trained and
evaluated at 128 tokens: **0.837**.

SemEval region localization is led by `littlelamb` with `extended` features,
trained and evaluated at 64 tokens: **0.767 MG-region F1**, **0.707 accuracy**,
and **62.81-token mean boundary error**. The best boundary result is not the
same as the best document-classification result, so model selection should use
the target task rather than a single global score.

## Evidence And Integrity

The evaluated artifact metadata identifies the classifier as
`logistic_regression` for all saved configurations. Consequently, this run does
not support claims comparing logistic regression with random forests, SVMs,
gradient boosting, neural classifiers, or other estimators.

The run was generated before the CSV placeholder-header correction. The initial
schema-only headers have fewer columns than the real frames, and the SemEval
detail files have no header. The numeric comparisons in this report were
reconstructed from the stable positional row layout and filtered to artifacts
whose artifact run ID is `feature-vector-classification-2026-09-12`. The raw
result files should be regenerated once with the corrected notebook before
using them as a final publication artifact.

The report excludes old `2026-09-11` artifacts that were also discovered by the
notebook. Unavailable configurations are not treated as zero performance.

## Experimental Configuration

| Dimension | Values |
| --- | --- |
| Training regime | `Mixed`: Kaggle + MAGE + Outfox |
| Evaluation regimes | Kaggle, MAGE, Outfox, Mixed, Mixed+BBC, BBC where available |
| Scorer LLMs | `gemma_3_1b_it`, `gemma_3_270m_it`, `littlelamb`, `smollm2_360m` |
| Feature sets | `scalar_all`, `scalar_compact`, `simple_similarity`, `extended` |
| Training windows | 32, 64, 128 tokens |
| Evaluation windows | whole text, 32, 64, 128 tokens |
| Classifier family | `logistic_regression` only |
| Class weighting | `balanced` |
| Optimizer limit | `max_iter=1000` |
| Random seed | 42 |
| Local-evidence stride | 16 tokens |
| Local-evidence top-k | 20 |
| Decision threshold | 0.5 for document scores; local operating points selected on validation data |

Macro F1 is the primary document metric because it weights HG and MG equally.
ROC-AUC is used as a ranking diagnostic. Dataset-level values below are means
over the saved source-specific rows, not a recomputation from concatenated raw
predictions.

## Feature-Set Comparison

Mixed training and Mixed evaluation, test partition, averaged over Kaggle,
MAGE, Outfox, scorer LLMs, training windows, and evaluation windows:

| Feature set | Mean macro F1 | Mean ROC-AUC | Mean balanced accuracy |
| --- | ---: | ---: | ---: |
| `simple_similarity` | 0.793 | 0.852 | 0.811 |
| `extended` | 0.661 | 0.823 | 0.712 |
| `scalar_all` | 0.525 | 0.724 | 0.617 |
| `scalar_compact` | 0.477 | 0.712 | 0.587 |

The similarity representation improves mean macro F1 by 0.132 over `extended`
and by 0.316 over `scalar_compact`. Adding scalar features to similarity in
`extended` does not improve the aggregate result; in this run it reduces macro
F1. This may indicate feature-scale, redundancy, or distribution-shift effects,
not that scalar features are intrinsically uninformative.

### Evaluation-window effect

| Feature set | 32 tokens | 64 tokens | 128 tokens |
| --- | ---: | ---: | ---: |
| `simple_similarity` | 0.736 | 0.764 | **0.803** |
| `extended` | 0.698 | **0.719** | 0.662 |
| `scalar_all` | 0.609 | 0.573 | 0.494 |
| `scalar_compact` | 0.533 | 0.502 | 0.459 |

The two scalar-only sets degrade as evaluation windows become longer. Similarity
features behave differently: 128-token evaluation is strongest. `extended`
peaks at 64 tokens and then falls at 128, suggesting that the added scalar
features interfere with long-window aggregation.

### Training-window effect

| Feature set | 32-token training | 64-token training | 128-token training |
| --- | ---: | ---: | ---: |
| `simple_similarity` | 0.798 | **0.800** | **0.800** |
| `extended` | 0.546 | 0.631 | **0.697** |
| `scalar_all` | 0.393 | 0.471 | **0.553** |
| `scalar_compact` | 0.396 | 0.482 | **0.561** |

Similarity is comparatively insensitive to training window. The other feature
sets benefit strongly from 128-token training, especially `extended`.

## Scorer LLM Comparison

| Scorer LLM | Mean macro F1 | Mean ROC-AUC | Mean balanced accuracy |
| --- | ---: | ---: | ---: |
| `smollm2_360m` | **0.649** | 0.775 | 0.695 |
| `littlelamb` | 0.607 | **0.778** | 0.686 |
| `gemma_3_270m_it` | 0.605 | 0.789 | 0.676 |
| `gemma_3_1b_it` | 0.595 | 0.769 | 0.670 |

`smollm2_360m` has the highest thresholded classification score, while
`gemma_3_270m_it` has the highest mean ROC-AUC. This difference means that
ranking quality and the fixed operating point do not identify exactly the same
winner. Threshold calibration should be reported alongside AUC.

Under Mixed evaluation, the strongest source-specific mean test results are:

| Evaluation source | Best configuration | Mean macro F1 |
| --- | --- | ---: |
| Kaggle | `smollm2_360m` + `extended`, train 64, eval 128 | **0.981** |
| MAGE | `littlelamb` + `extended`, train 64, eval 128 | **0.766** |
| Outfox | `gemma_3_270m_it` + `simple_similarity`, train 128, eval 128 | **0.934** |

MAGE is the difficult source. Even the best Mixed-trained configuration is
substantially below the corresponding Kaggle and Outfox results. This is the
main dataset-specific generalization risk in the run.

## Best Mixed-Evaluation Configurations

The following table ranks selected configurations by mean test macro F1 across
Kaggle, MAGE, and Outfox under Mixed evaluation:

| Scorer | Features | Train window | Eval window | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `smollm2_360m` | `extended` | 64 | 128 | **0.843** | 0.853 |
| `littlelamb` | `extended` | 128 | 128 | 0.837 | **0.932** |
| `smollm2_360m` | `extended` | 128 | 128 | 0.837 | 0.807 |
| `smollm2_360m` | `scalar_all` | 128 | 128 | 0.833 | 0.802 |
| `smollm2_360m` | `simple_similarity` | 64 | 128 | 0.832 | 0.800 |
| `smollm2_360m` | `scalar_compact` | 128 | 128 | 0.831 | 0.807 |
| `gemma_3_270m_it` | `extended` | 128 | 128 | 0.827 | 0.874 |
| `littlelamb` | `simple_similarity` | 32 | 128 | 0.824 | 0.917 |

The top result is not `simple_similarity`, despite that set winning the broad
average. This is a reminder to distinguish a robust average from the best
single operating configuration.

## SemEval Region Localization

SemEval results are evaluated on the train split of the available prepared
SemEval dataset. The best retained region configurations are:

| Scorer | Features | Train/eval window | Accuracy | MG precision | MG recall | MG F1 | Boundary error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `littlelamb` | `extended` | 64/64 | 0.707 | 0.761 | 0.810 | **0.767** | 62.81 |
| `littlelamb` | `extended` | 128/128 | 0.717 | 0.744 | 0.830 | 0.763 | 76.04 |
| `littlelamb` | `scalar_all` | 128/128 | 0.704 | 0.737 | 0.816 | 0.750 | 72.82 |
| `littlelamb` | `simple_similarity` | 64/64 | 0.681 | 0.736 | 0.791 | 0.745 | 73.86 |
| `gemma_3_1b_it` | `extended` | 64/64 | 0.592 | 0.661 | 0.650 | 0.627 | 80.05 |

The best boundary error among the leading retained rows is **45.83 tokens**
for `littlelamb` + `scalar_all`, train/eval 32/32, but its region F1 is only
0.714. Region detection and boundary localization therefore have a clear
trade-off. A future selection rule should report both metrics, or use a
multi-objective criterion.

## Classifier Parameters And Scope

All classifier results use scikit-learn logistic regression wrapped in a
pipeline with median imputation and standard scaling. Class balancing is
enabled and the maximum iteration count is 1,000. The experiment uses seed 42.
No regularization sweep, class-threshold sweep, alternative solver comparison,
or alternative classifier family comparison was performed in this run.

For local evidence, 32/64/128-token windows use stride 16 and top-k 20 token
candidate features. Local document decisions use aggregation rules and
thresholds selected from validation outputs. Those operating points should not
be compared as if they were identical to the whole-text model's fixed 0.5
threshold.

## Conclusions And Recommendations

1. Use `simple_similarity` as the primary baseline for document classification.
2. Use `extended` when SemEval localization is the priority, especially with
   `littlelamb` and 64-token windows.
3. Prefer 128-token training for scalar and extended feature sets.
4. Treat MAGE as the main generalization bottleneck and report it separately.
5. Select scorer LLMs by task: `smollm2_360m` for thresholded document F1,
   `gemma_3_270m_it` for ranking quality, and `littlelamb` for SemEval region
   localization.
6. Regenerate the CSV exports after the saving-flow fix before publication.
7. Add non-logistic classifier families and repeated seeds before making claims
   about classifier type or statistical significance.
