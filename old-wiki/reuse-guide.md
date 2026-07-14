# Reuse Guide For A New Project

## What To Reuse

Reuse the **experimental structure**, not trained artifacts or raw thresholds:

```text
provenance-aware labeled data
  + scorer adapter
  + reproducible token-evidence store
  + likelihood/discrepancy feature functions
  + split-safe classifier pipeline
  + in-domain and cross-domain evaluation
```

Do not reuse old model weights, feature normalization statistics, outlier limits, decision thresholds, sample manifests, or reported scores. All of them depend on the original corpus and scorer configuration.

## Implementation Order

### Phase 1: Define The Detection Target

- Write down exactly what `MG` means: fully generated text, assisted writing, paraphrased text, translated text, or another condition.
- Decide the target language, length range, genre, and time period.
- Define the action taken for a positive prediction and its tolerated HG false-positive rate.
- Record known generators and prompt families separately from the binary label.

Deliverable: versioned `texts` data contract with source provenance.

### Phase 2: Build Leakage-Resistant Splits

- Deduplicate exact and near-duplicate texts.
- Group source documents, prompts, paraphrase families, and chunks.
- Reserve a final in-domain test set before feature investigation.
- Reserve at least one external-domain/generator set before modeling.

Deliverable: immutable train/validation/test manifests containing text IDs and group IDs.

### Phase 3: Implement The Scorer Adapter

- Choose one causal LM as the initial scorer.
- Pin model revision, tokenizer revision, inference library, device type, and endpoint settings.
- Choose `full` vocabulary mode when infrastructure permits; otherwise use explicitly recorded `top_k` mode.
- Preserve raw observed log probabilities, candidate distributions, token IDs, and failures.
- Add tests for empty input, one-token input, candidate alignment, and repeated deterministic scoring where applicable.

Deliverable: token-evidence artifact per `(text_id, scorer_id)`.

### Phase 4: Implement Feature Extraction

- Use the pure functions from [Features](features.md).
- Exclude the initial non-predictive autoregressive position.
- Use `ddof=0` or `ddof=1` deliberately and record the choice.
- Include length controls but do not assume they are semantic detection features.
- Version the feature schema and validate finite values.

Deliverable: a scorer-specific feature table plus a data-quality report.

### Phase 5: Establish Classifier Baselines

- Begin with a scaled RBF SVM and constrained Random Forest.
- Add linear SVM and XGBoost only as controlled comparisons.
- Fit all transformations inside pipelines and cross-validation folds.
- Tune hyperparameters against validation/CV macro F1, plus the class-specific recall constraint required by the application.

Deliverable: cross-validation summary and selected pipeline configuration.

### Phase 6: Calibrate And Evaluate

- Choose an operating threshold only on validation predictions.
- Freeze the configuration and threshold.
- Report in-domain test performance once.
- Report external-domain performance separately, including failure analysis.
- Inspect the highest-confidence HG positives and MG negatives manually, subject to privacy policy.

Deliverable: immutable predictions and report bundle for every test domain.

## Baseline Configuration Template

Use a human-readable config file such as this one. It deliberately does not depend on this repository.

```yaml
run_id: baseline-001
data:
  label_mapping:
    HG: 0
    MG: 1
  group_column: split_group
  language: en
  min_sentences: null

scorer:
  id: scorer-model-id
  revision: pinned-revision-or-sha
  mode: full # full or top_k
  top_k: null
  sample_count: 1000
  seed: 42

features:
  version: likelihood-discrepancy-v1
  columns:
    - perplexity
    - prompt_mu_score
    - prompt_sigma_score
    - d_score
    - d_mu_score
    - d_sigma_score
    - num_tokens

evaluation:
  split_seed: 42
  cv_folds: 5
  primary_metric: macro_f1
  threshold_objective: macro_f1
  positive_label: 1
```

## Minimum Test Suite

| Test | Expected assertion |
| --- | --- |
| Perplexity identity | Constant log probability `log(p)` gives `1 / p`. |
| Prompt summary | Mean and population standard deviation match NumPy. |
| Discrepancy shape | One observed candidate index exists for every candidate row. |
| Missing observed token | Top-`k` feature extraction fails explicitly rather than guessing. |
| Split isolation | No `text_id` or `split_group` appears in multiple partitions. |
| Pipeline isolation | Feature scaler/imputer is fitted only on each training fold. |
| Label mapping | HG always maps to `0`; MG always maps to `1`. |
| Reproducibility | Fixed seed produces the same feature samples and split manifest. |
| Artifact completeness | Every prediction can be linked to source text ID, scorer config, model config, and threshold. |

## Reproducibility Checklist

Before publishing or relying on a result, record all items below.

- [ ] Dataset sources, license/consent status, collection dates, and preprocessing version.
- [ ] Exact label definition and class counts in every split/domain.
- [ ] Grouping/deduplication procedure and split manifests.
- [ ] Scorer model, tokenizer, revision, vocabulary mode, endpoint/library, hardware, and decoding settings.
- [ ] Definition and code version of every scalar feature.
- [ ] Candidate `top_k`, sample count, RNG seed, and invalid-token policy.
- [ ] Classifier family, hyperparameters, feature columns, scaling/imputation, and fitted-artifact checksum.
- [ ] Validation-only threshold selection rule and frozen threshold.
- [ ] Per-class test metrics, confusion matrices, ROC-AUC, PR-AUC, and calibration evidence.
- [ ] Cross-domain results with support counts and a qualitative false-positive/false-negative review.

## Decision Rules

Use the following stop conditions instead of pushing a weak detector toward deployment:

| Condition | Action |
| --- | --- |
| Cross-domain macro F1 is materially lower than in-domain macro F1 | Treat the detector as domain-specific; collect representative data or retrain per domain. |
| HG recall is below the application requirement | Do not use MG-positive predictions for enforcement; adjust threshold only through validation. |
| A feature separates labels mainly by text length or source | Remove/match that confounder and repeat the split. |
| A scorer fails more often for one class/domain | Investigate scoring behavior before training. |
| Results vary strongly by RNG seed or split | Report uncertainty and expand the dataset; do not claim stable performance. |
| Generator, language, or topic changes | Re-run full calibration and external evaluation. |

## Practical Deployment Posture

Treat the detector as a ranking or review-assistance tool unless it has passed representative, current, cross-domain validation for the exact decision context. Present confidence, scorer configuration, and limitations to downstream users. Never present a single binary detection result as proof of authorship or misconduct.
