---
type: Domain Model
title: Domain Model and Data Layout
description: Canonical explanation of the repository's dataset aliases, model aliases, experiment naming, split conventions, and processed output layouts. This page is the best reference for the on-disk data model used by the pipeline and monitoring jobs.
tags: [domain, data-model, datasets, state]
resource: /scripts/experiment_config.py
---

# Domain model and data layout

This repository uses a small, explicit domain model built around datasets, models, experiments, and split-based processed outputs. The authoritative definitions live in [`scripts/experiment_config.py`](../scripts/experiment_config.py), and the downstream stages all depend on those names staying stable.

## Core domain terms

- `HG`: human-generated text.
- `MG`: machine-generated text.
- Dataset alias: the short key used on the command line and in tests.
- Model alias: the short key that maps to a Hugging Face model ID.
- Experiment name: `{dataset}__{model}` after path-safe normalization.

The older documentation mentions `HW` in one place, but the current code and file names use `HG` and `MG` consistently.

## Configuration source of truth

[`scripts/experiment_config.py`](../scripts/experiment_config.py) defines:

- `DATA_ROOT`, loaded from `TEXT_DETECTION_DATA`.
- `MODEL_IDS`, the alias-to-model-ID map.
- `DATASETS`, the dataset alias-to-input-path map.
- `resolve_model_id`, `resolve_dataset_name`, and `resolve_dataset_splits`.

If you change dataset or model naming, update this file first and then update pipeline commands, monitoring variables, and tests that assume the old names.

## Input data expectations

The pipeline expects CSV split files with stable `id` and `text` columns. The configured input paths vary by dataset, but the command-line contract is always the same: the chosen dataset alias must resolve to a set of split files that later stages can sort and resume over by `id`.

## Processed stage 1 layout

Stage 1 writes to:

```text
processed/{dataset}__{model}/{split}/data/*.parquet
processed/{dataset}__{model}/{split}/state.json
```

Each stage 1 row stores the original text, the token-by-token score output, and the experiment identity fields needed by later stages and metrics exports.

## Processed stage 2 layout

Stage 2 writes to:

```text
processed/{dataset}__{model}/{split}/token_similarities/*.parquet
processed/{dataset}__{model}/{split}/similarity_state.json
```

The stage 2 rows keep the same experiment identity fields and add similarity arrays for each token's top-logprob candidates.

## Scalar metric exports

Two downstream reductions are important:

- [`scripts/logprob_metrics.py`](../scripts/logprob_metrics.py) derives `perplexity`, prompt score summaries, text-length controls, and top-20 candidate metrics from stage 1 rows.
- [`scripts/export_token_similarity_stats.py`](../scripts/export_token_similarity_stats.py) derives summary statistics from stage 2 rows.

The stage 1 metrics output now uses `metrics_version = logprob_metrics_v3` and includes:

- `d_score_top20`: a passage-level standardized difference between the actual token logprob and the weighted mean of finite top-20 candidate logprobs; it is `NaN` when no valid candidate distribution exists or its aggregate variance is non-positive.
- `d_score_top20_coverage`: the mean finite candidate probability mass represented by the supplied top-20 logprobs, or `NaN` when no valid candidate mass exists.
- `d_score_top20_num_tokens`: the number of eligible tokens with at least one candidate logprob.

These exports are useful because they convert nested token structures into flat tables for analysis and dashboards. Their progress metadata is written to `metrics_state.json` with the same `metrics_version`.

The metric calculations and missing-candidate behavior are covered by [`tests/test_logprob_metrics.py`](../tests/test_logprob_metrics.py) and summarized in the [pipeline workflow](../workflows/pipeline.md).

## State-file contract

Both `state.json` and `similarity_state.json` include the same core identity fields plus progress tracking:

- `dataset`
- `dataset_name`
- `model`
- `model_id`
- `experiment_name`
- `experiment_params`
- `split`
- `last_id`
- `processed_count`
- `updated_at`

Stage 2 state files also include `stage = token_similarity` and `similarity_method = input_embedding_cosine`.

## Relationships to other pages

- [`Pipeline workflows`](../workflows/pipeline.md) explains how the files are produced.
- [`Architecture overview`](../architecture/overview.md) explains why the split/resume model exists.
- [`Operations and monitoring`](../operations/monitoring.md) explains how the state files are consumed.
- [`Testing guidance`](../testing.md) explains which tests should change when the data model changes.
