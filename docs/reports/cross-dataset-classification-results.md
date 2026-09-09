# Cross-Dataset Classification Results

## Executive Summary

This report analyses the saved execution of
`notebooks/cross_dataset_classification_poc.ipynb`. The notebook was **not
rerun**. Consequently, every quantitative claim below is limited to the
tables retained in the notebook output; unavailable or formatter-truncated
results are identified rather than reconstructed.

The main document-classification result is clear: whole-text statistical
logistic regression is highly accurate in distribution but its single-source
cross-dataset performance is strongly source dependent. Training on all three
available binary sources (Kaggle, MAGE, and Outfox) gives the most consistently
high test macro F1 across those sources: 0.945-0.995 for the three fully
available scorer LLMs. In contrast, a source-specific detector may collapse to
approximately chance-level macro F1 (about 0.33) on a different source.

`littlelamb` has the strongest single-source transfer from Outfox to MAGE
(0.864 test macro F1), while `gemma_3_1b_it` has the strongest observed
Outfox-to-Kaggle transfer (0.670). Neither exception removes the overall case
for mixed-source training. `gemma_3_270m_it` gives the best mixed-training
whole-text test macro F1 on Kaggle (0.995) and on the pooled Mixed evaluation
(0.968); `littlelamb` is best on MAGE (0.975) and `gemma_3_1b_it` is best on
Outfox (0.965, tied with its own Outfox-only model).

The notebook does **not** compare multiple learning algorithms: both the
whole-text and local-evidence blocks use logistic regression. It does compare
two classifier families: a document-level model over scalar log-probability
features and a local-evidence model that scores token windows before
document-level aggregation. Therefore, claims about "classification model
type" in this report mean family/feature representation and aggregation, not
a comparison of logistic regression against alternative estimators.

For SemEval MG-region localization, `littlelamb` trained on MAGE at window
length 64 is the best retained configuration by MG-region F1 (0.778). Boundary
localization remains difficult: the best mean absolute token error is 46.47
tokens (`gemma_3_270m_it`, MAGE, 32-token windows) and exact-boundary accuracy
does not exceed 5.41% in the saved SemEval-train results.

## Experimental Design and Evidence Scope

### Data and regimes

The notebook defines four named source datasets:

| Role | Dataset | Alias |
| --- | --- | --- |
| Binary source | Kaggle | `kaggle_train` |
| Binary source | MAGE | `mage_train` |
| Binary source | Outfox | `outfox_chatgpt_train` |
| HG-only holdout | BBC | `bbc_news_content` |

Training regimes are Kaggle-only, MAGE-only, Outfox-only, and Mixed
(Kaggle + MAGE + Outfox). Evaluation regimes include those same binary sources,
the pooled Mixed evaluation, BBC, and SemEval 2024 Task 8 Subtask C. The
notebook builds deterministic parent cohorts from the intersection of scalar
and local-evidence availability, caps each HG/MG source split at 500 parents,
and partitions with seed 42. The intended document partitions are train,
validation, and test.

### Models and settings

Four scorer LLM names are configured: `gemma_3_1b_it`, `gemma_3_270m_it`,
`littlelamb`, and `smollm2_360m`. The saved results show usable full
whole-text matrices for the first three. `smollm2_360m` has MAGE and Outfox
coverage in saved tables but lacks Kaggle coverage, so Kaggle and Mixed results
for that scorer are undefined rather than low-scoring.

Both classifier families use logistic regression:

| Family | Input representation | Decision level |
| --- | --- | --- |
| Whole text | Scalar log-probability-derived features, including prompt summary scores and perplexity-derived values | One probability per document, threshold 0.5 |
| Local evidence | Window features derived from token-level evidence | Score 32, 64, or 128-token windows, then aggregate with `maximum_window` or `evidence_fraction` |

Local evidence uses stride 16 and top-k 20. The local-evidence threshold and,
when applicable, evidence-fraction window threshold are selected on validation
data per scorer/training/window/aggregation configuration. This is not the
same operating point as the fixed 0.5 document threshold of the whole-text
model.

### Metrics and interpretation

Binary document classification is primarily compared with macro F1, which
equally weights HG and MG F1. ROC-AUC and PR-AUC are retained by the notebook
but should be interpreted only where both classes are present. BBC is HG-only:
the relevant outcome would be HG texts falsely predicted as MG, not binary
ranking metrics. SemEval region quality is measured by MG-region F1; boundary
quality includes absolute token error and exact/within-tolerance accuracy.

## Whole-Text Statistical Classification

The following tables reproduce the exact **test macro F1** values retained in
the notebook's whole-text pivot. Rows identify the source data used for
training and columns identify the binary evaluation source. `Mixed` pools the
three binary evaluation sources before metrics are calculated.

### `gemma_3_1b_it`

| Training source | Kaggle | MAGE | Outfox | Mixed |
| --- | ---: | ---: | ---: | ---: |
| Kaggle | 0.995 | 0.444 | 0.476 | 0.673 |
| MAGE | 0.333 | 0.970 | 0.344 | 0.619 |
| Outfox | 0.670 | 0.663 | 0.965 | 0.774 |
| Mixed | 0.990 | 0.905 | 0.965 | 0.953 |

The model is near perfect in distribution for Kaggle and Outfox. Kaggle-only
training transfers weakly to MAGE and Outfox; MAGE-only training is especially
poor on Kaggle and Outfox. Mixed training increases the weakest cross-source
result substantially, particularly on MAGE (0.905 versus 0.444 for Kaggle-only
and 0.663 for Outfox-only).

### `gemma_3_270m_it`

| Training source | Kaggle | MAGE | Outfox | Mixed |
| --- | ---: | ---: | ---: | ---: |
| Kaggle | 0.990 | 0.344 | 0.331 | 0.623 |
| MAGE | 0.333 | 0.970 | 0.344 | 0.619 |
| Outfox | 0.507 | 0.608 | 0.985 | 0.726 |
| Mixed | 0.995 | 0.950 | 0.960 | 0.968 |

Mixed training is decisively strongest for this scorer. It is the best saved
mixed configuration on Kaggle and the pooled Mixed evaluation, and its
worst-source score under mixed training is 0.950. This contrasts with
Kaggle-only transfer to Outfox (0.331) and MAGE-only transfer to Kaggle
(0.333).

### `littlelamb`

| Training source | Kaggle | MAGE | Outfox | Mixed |
| --- | ---: | ---: | ---: | ---: |
| Kaggle | 0.965 | 0.388 | 0.335 | 0.621 |
| MAGE | 0.333 | 0.985 | 0.344 | 0.623 |
| Outfox | 0.498 | 0.864 | 0.930 | 0.782 |
| Mixed | 0.980 | 0.975 | 0.945 | 0.967 |

`littlelamb` provides the strongest single-source MAGE transfer: an
Outfox-trained model reaches 0.864 on MAGE. It still has poor Kaggle and Outfox
transfer from MAGE-only data, and the mixed model is the only configuration
that remains high on every available target source.

### Cross-scorer comparison under mixed training

| Scorer LLM | Kaggle | MAGE | Outfox | Mixed |
| --- | ---: | ---: | ---: | ---: |
| `gemma_3_1b_it` | 0.990 | 0.905 | 0.965 | 0.953 |
| `gemma_3_270m_it` | 0.995 | 0.950 | 0.960 | 0.968 |
| `littlelamb` | 0.980 | 0.975 | 0.945 | 0.967 |

The three scorers differ by at most 0.025 macro F1 under mixed training for
any individual source. Thus, expanding training-source diversity matters more
than choosing among these three scorer LLMs for whole-text test performance.
`gemma_3_270m_it` is the best all-round mixed scorer by pooled test F1;
`littlelamb` is preferable if MAGE is the primary target; and
`gemma_3_1b_it` is preferable if Outfox is the primary target.

## Local-Evidence Classification

The local-evidence block evaluates the same logistic-regression estimator on
window-level features. It tests two document aggregation strategies at window
lengths 32, 64, and 128. The saved local-evidence matrix contains 252
configuration rows across source/regime/partition/aggregation/window
combinations, demonstrating that the comparison was evaluated broadly.

The directly retained portions of the pivot show two useful properties:

| Saved example | Partition and macro F1 |
| --- | ---: |
| `gemma_3_1b_it`, Kaggle-trained, Kaggle-evaluated, evidence fraction, 32 | 0.980 |
| Same, 64-token windows | 1.000 |
| Same, 128-token windows | 0.995 |
| `gemma_3_1b_it`, Kaggle-trained, Kaggle-evaluated, maximum window, 32 | 0.985 |
| `smollm2_360m`, Outfox-trained, Outfox-evaluated, evidence fraction, 64, validation | 1.000 |
| `smollm2_360m`, Outfox-trained, Outfox-evaluated, maximum window, 64, validation | 0.980 |

These examples establish high in-distribution local-evidence performance and
show that aggregation/window choice can change results even with the same
underlying scorer and training source. They are not sufficient to make a
numerically defensible global family ranking: the notebook's saved
`text/plain` rendering elides the middle rows of the 252-row local matrix, and
no separate machine-readable result export is present in the notebook. The
plots remain valuable qualitative evidence but should not be converted into
unverified exact values.

The appropriate next comparison, once an explicit result export is added, is
paired test macro F1 for each `(LLM, training regime, evaluation regime)`
against the whole-text model, selecting local aggregation/window by validation
performance only. That comparison cannot be reconstructed from the saved
truncated display alone.

## Scorer LLM and Training-Data Effects

### Training-source diversity dominates isolated transfer

The whole-text tables demonstrate a recurring pattern:

- In-distribution test macro F1 is 0.930-0.995 for the fully available
  single-source configurations, except that the specific score depends on the
  source and scorer.
- Cross-source transfer ranges from 0.331 to 0.864. Scores close to 0.333
  indicate a detector that is not separating one of the binary sources
  reliably at its selected operating point.
- Mixed training attains 0.945-0.995 on every available individual source and
  0.953-0.968 on pooled Mixed evaluation.

The practical implication is that a detector trained on one data-generation
process should not be presented as dataset-general without cross-source
validation. In this saved experiment, mixed-source training is the robust
baseline.

### Scorer availability is part of the result

Availability differs by scorer and source. For example, saved coverage shows
1,000, 2,080, and 1,120 scalar rows for Kaggle, MAGE, and Outfox respectively
under `gemma_3_1b_it`, while BBC fails during cohort construction because of a
missing token-similarity row. The coverage table should be consulted before
comparing scores: unavailable configurations must not be interpreted as zero
performance.

`smollm2_360m` illustrates this constraint. Its saved whole-text MAGE and
Outfox values are present, but Kaggle and all Mixed metrics are `NaN` because
the shared scalar/local parent cohort cannot be built for Kaggle. It is not a
valid candidate for a four-source ranking in this notebook state.

## Text-Length Robustness

For Mixed-trained models, the notebook evaluates validation prefixes of 50,
100, 200, and 400 words alongside the original text. Both classifier families
are included: whole-text statistical and local-evidence aggregate MG score.

The saved comparison frame has 129 rows, covering scorer/dataset and
aggregation/window variants. It confirms that this is a shared-cohort,
controlled prefix experiment rather than a different training split. However,
the saved rendered table elides the macro-F1 column for the individual rows.
The associated bar plots are retained but are not a reliable source for
transcribing exact values. This report therefore records the analysis as
executed but does not claim a numerical robustness ordering.

The design is sound for a subsequent report revision: each model is trained
once on the full Mixed training cohort, then evaluated on the same validation
parents after deterministic first-N-word truncation. A future notebook revision
should write the 129-row frame to a versioned CSV or Parquet summary so that
length effects can be reported exactly.

## BBC False-Positive Evaluation

BBC is a deliberate HG-only evaluation. A correct report would measure the
false-HG-as-MG count and error rate per configuration. In the saved execution,
`bbc_false_positive_summary` is an empty table because BBC is unavailable in
the shared source data for the evaluated scorer LLMs. The reported cause in
coverage is a missing token-similarity row, for example:

```text
No token similarity row for text id 1600
```

Therefore, this notebook state does **not** provide evidence about false
accusations on BBC text. It is incorrect to infer either a zero false-positive
rate or a favourable BBC result from the empty summary. Restoring complete BBC
stage-2/token-similarity coverage is a prerequisite for this analysis.

## SemEval Region and Boundary Localization

### Evaluation availability

Exact SemEval summary values are retained only for
`semeval2024_task8_subtask_c_train` and three scorer LLMs. The saved exclusion
frame contains 31,416 rows, principally missing processed token labels and
missing Stage-1/local-evidence data; it includes unsuccessful attempts for the
SemEval development split and for unavailable scorer/regime combinations.
Consequently, results below are SemEval-train-only and should not be read as
held-out development performance.

### Best MG-region F1 by scorer and training regime

Each cell is the best MG-region F1 across the evaluated 32, 64, and 128-token
windows. The selected window is in parentheses.

| Scorer LLM | Kaggle | MAGE | Outfox | Mixed |
| --- | ---: | ---: | ---: | ---: |
| `gemma_3_1b_it` | 0.713 (64) | 0.772 (64) | 0.340 (32) | 0.537 (32) |
| `gemma_3_270m_it` | 0.566 (32) | 0.710 (64) | 0.329 (32) | 0.438 (32) |
| `littlelamb` | 0.687 (32) | 0.778 (64) | 0.564 (64) | 0.728 (64) |

MAGE training is the strongest SemEval region-localization source for all
three scorers. `littlelamb` is the only scorer for which Mixed training
approaches the best MAGE-only result (0.728 versus 0.778). Outfox-only training
is consistently weaker, especially for the Gemma scorers.

### Boundary localization

Boundary localization is markedly weaker than region detection. The most
favourable retained configurations are:

| Objective | Configuration | Result |
| --- | --- | ---: |
| Lowest mean absolute token error | `gemma_3_270m_it`, MAGE, 32 tokens | 46.47 tokens |
| Highest exact-boundary accuracy | `littlelamb`, MAGE, 32 tokens | 5.41% |
| Highest within-5-token accuracy | `gemma_3_270m_it`, MAGE, 64 tokens | 11.64% |
| Lowest median absolute token error | `gemma_3_270m_it`, MAGE, 64 tokens | 35.0 tokens |

The metrics capture different failure modes. For example,
`gemma_3_270m_it` trained on MAGE at 64 tokens has the lowest median error but
not the lowest mean error, suggesting a long error tail. Its 64-token
configuration also has 54.6% early and 41.8% late boundary
predictions, plus a 4.9% no-crossing rate. Boundary estimates should therefore
be treated as coarse localization signals rather than precise edit boundaries.

## Limitations and Reproducibility

### Limitations

- Results come from the saved notebook state only; the notebook was not rerun.
- BBC is unavailable, so no human-text-only false-positive conclusion is
  available.
- `smollm2_360m` lacks Kaggle shared-cohort coverage and cannot be fairly
  ranked against the three full-coverage scorers.
- SemEval results are retained for the train split only; the development split
  is unavailable in the saved state.
- The saved local-evidence and text-length display tables are formatter-truncated.
  Their full quantitative ranking cannot be derived without an explicit export
  or a new evaluation execution.
- Thresholds differ by family: whole-text uses a fixed 0.5 document threshold,
  while local evidence tunes document rules on validation data. Raw error-rate
  comparisons require this distinction.
- The parent cohorts are capped at 500 per source HG/MG split. The experiment
  compares controlled cohorts, not each dataset's full available population.

### Reproduction procedure

The report source is the notebook configuration and its saved execution
outputs. The configuration specifies:

```text
LLMs: gemma_3_1b_it, gemma_3_270m_it, littlelamb, smollm2_360m
Classifier: logistic_regression
Seed: 42
Maximum parents per HG/MG split: 500
Window lengths: 32, 64, 128
Stride: 16
Top-k: 20
Validation word limits: 50, 100, 200, 400
SemEval sustained length: 3
```

To refresh the report quantitatively, first repair the incomplete processed
coverage, then rerun the notebook deliberately and export the result frames
(`whole_text_metrics`, `local_evidence_metrics`,
`short_validation_comparison`, `semeval_region_summary`, and
`semeval_boundary_summary`) in a machine-readable form. That action is outside
the scope of this report.

## Saved-Output Inventory

The report used the following saved notebook artifacts:

| Notebook cell | Artifact | Use in this report |
| --- | --- | --- |
| Configuration | Dataset/model regimes and fixed settings | Experimental design |
| Cohorts | Source coverage and shared-parent construction | Availability caveats |
| Block 1 | `whole_text_metrics`, predictions, and coverage | Exact document-level test macro F1 tables |
| Block 2 | `local_evidence_metrics`, predictions, and coverage | Family structure and directly retained examples |
| Block 3 | SemEval region, boundary, and exclusion frames | Exact SemEval-train region and boundary findings |
| Matrix displays | Whole-text, local-evidence, SemEval pivots | Cross-source comparison and retained exact values |
| Additional visualizations | Prefix-length, mixed-scorer, BBC, and boundary plots | Confirmation of executed analyses; no unverified value transcription |

The saved notebook is the authoritative record for this report. An empty,
`NaN`, unavailable, or formatter-elided value is kept distinct from a measured
low score throughout.
