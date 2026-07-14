# Machine-Generated Text Detection Baseline

This wiki is a self-contained technical reference for rebuilding the scoring and classification approach explored in `scripts/demo_3.ipynb` and evaluated visually in `scripts/demo_3_visual.ipynb`. It is intended for a new text-detection project, not as a claim that a trained model or threshold from this repository will transfer unchanged.

## Goal

Given labeled human-generated (HG) and machine-generated (MG) texts, score every text with one or more causal language models, derive likelihood and sampling-discrepancy features, then train and validate a binary classifier. The core premise is that a scorer model may assign measurably different likelihood patterns to human and generated text.

The output is a probability or label for the **origin of the entire input text**. It is not authorship attribution, factuality assessment, plagiarism detection, or a reliable detector for every generator and domain.

## Contents

- [Pipeline](pipeline.md): data contracts, scorer modes, storage, transformations, and artifacts.
- [Features](features.md): precise metric definitions, formulas, and standalone NumPy reference code.
- [Evaluation](evaluation.md): classifiers, splits, reports, diagnostics, and cross-domain validation.
- [Reuse Guide](reuse-guide.md): a practical implementation sequence and reproducibility checklist.

## Baseline In One View

```text
immutable text records with HG/MG labels
    -> score each text with each selected causal LM
    -> retain token likelihoods and candidate distributions
    -> calculate likelihood, discrepancy, and length features
    -> one feature row per (text, scorer model)
    -> train a binary classifier per scorer configuration
    -> validate in-domain and cross-domain
    -> calibrate a final operating threshold only on validation data
```

## Terminology

| Term | Meaning |
| --- | --- |
| HG | Human-generated / human-written text. The repository stores this as `origin == "HG"`. |
| MG | Machine-generated text. The repository stores this as `origin == "MG"`. |
| scorer | A causal language model used to compute features. It is not necessarily the model that generated MG text. |
| generation model | The unknown or known model that created a machine-generated example. |
| scorer configuration | Scorer identity, tokenizer/revision, vocabulary mode, `top_k`, sampling count, and preprocessing version. |
| feature row | Features for one text scored by one scorer configuration. |
| in-domain evaluation | Training and testing within the same dataset/domain after a held-out split. |
| cross-domain evaluation | Training on one source/domain and testing on another. This is the important portability test. |

## Minimal Data Contract

Build a table with one immutable row per text before any scoring. Do not let labels, deduplication decisions, or text normalization depend on the train/test split.

| Field | Type | Requirement |
| --- | --- | --- |
| `text_id` | string or integer | Stable unique identifier. |
| `text` | string | Original scoring input, with a documented normalization version. |
| `label` | integer | `0` for HG and `1` for MG. |
| `origin` | string | Optional readable form: `HG` or `MG`. |
| `dataset_id` | string | Source corpus/domain identifier. |
| `generator_id` | string or null | Known MG generator, if available. Never infer it from the detection label. |
| `split_group` | string or null | Group ID used to keep related samples together during splitting. |
| `metadata` | mapping | Source, prompt family, language, date, and other provenance needed for audits. |

Example:

```python
texts = pd.DataFrame(
    [
        {
            "text_id": "news-hg-0001",
            "text": "...",
            "label": 0,
            "origin": "HG",
            "dataset_id": "news-v1",
            "generator_id": None,
            "split_group": "article-0001",
        },
        {
            "text_id": "news-mg-0001",
            "text": "...",
            "label": 1,
            "origin": "MG",
            "dataset_id": "news-v1",
            "generator_id": "generator-x",
            "split_group": "prompt-0042",
        },
    ]
)
```

## Recommended Default Baseline

Use this as the first reproducible experiment in a new project:

1. Freeze a labeled corpus and group related/near-duplicate texts before splitting.
2. Define a development dataset, a held-out in-domain test set, and at least one external-domain test set.
3. Score all splits with one small causal LM using full-vocabulary access when feasible; otherwise record the precise top-`k` approximation.
4. Extract `perplexity`, `prompt_mu_score`, `prompt_sigma_score`, `d_score`, `d_mu_score`, `d_sigma_score`, and text-length controls.
5. Train a scaled RBF SVM and a tree ensemble as complementary baselines.
6. Select hyperparameters and any decision threshold using training folds and validation data only.
7. Report per-class precision, recall, F1, macro F1, confusion matrix, ROC-AUC, PR-AUC, and calibration diagnostics.
8. Report cross-domain results separately. Do not merge them into an in-domain average.

## Repository Traceability

The wiki is independent of these files, but they establish the historical implementation:

| Source | Historical responsibility |
| --- | --- |
| `scripts/demo_3.ipynb` | Data selection, metric distributions, classifier experiments, custom-text and threshold explorations. |
| `scripts/demo_3_visual.ipynb` | CSV result normalization, classifier comparison plots, heatmaps, and custom-text report aggregation. |
| `scripts/process_texts.py` | Database-backed text scoring and post-processing of stored metrics. |
| `scripts/fast_dgpt_mini.py` | Top-`k` probability conversion and Monte-Carlo discrepancy score. |
| `scripts/fast_detect_gpt.py` | Full-vocabulary Fast-DetectGPT-style discrepancy calculation. |
| `scripts/tools.py` | Database extraction, text controls, outlier filtering, distributions, and Random Forest helpers. |
| `scripts/database/db_models.py` | Historical `texts` and `processed_texts` schemas. |

## Important Boundary

The source notebooks contain exploratory cells, repeated experiments, hard-coded paths, and some code that should not be reproduced as baseline practice. This wiki names those behaviors where relevant and provides a safer default. Historical notebook outputs and saved scores are deliberately not presented as portable performance claims.
