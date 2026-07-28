---
type: Workflow Guide
title: Pipeline Workflows
description: Operational guide for the staged text-detection pipeline. Covers dataset preparation, stage 1 logprob processing, stage 2 similarity computation, resume behavior, and the most important CLI arguments.
tags: [workflow, pipeline, cli, resume]
resource: /scripts/process_pipeline.py
---

# Pipeline workflows

The repository's main workflow is a staged pipeline that can be resumed from the last processed `id`. The stages are designed to be rerunnable without clobbering existing data unless a command explicitly asks to recreate output.

## Stage 0: dataset preparation

Stage 0 converts raw inputs into the CSV split files consumed by later stages. The clearest example is Outfox ChatGPT preparation, implemented by [`scripts/extract_data/prepare_outfox_chatgpt.py`](../scripts/extract_data/prepare_outfox_chatgpt.py) and described in the existing knowledge docs under `docs/knowledge/processing-pipeline.md`.

That preparation step matters because stage 1 and stage 2 both expect stable split files with `id` and `text` columns.

## Stage 1: logprob scoring

[`scripts/process_pipeline.py`](../scripts/process_pipeline.py) is the primary entrypoint. It:

- resolves the selected dataset and model aliases from [`scripts/experiment_config.py`](../scripts/experiment_config.py),
- loads a local Hugging Face causal LM through [`scripts/openai_logprobs.py`](../scripts/openai_logprobs.py),
- scores text rows in bounded batches,
- writes Parquet files under `processed/{dataset}__{model}/{split}/data`, and
- updates `state.json` after each written batch.

Important CLI behavior:

- `--count` controls how many new rows to process per split.
- `--dataset` and `--model` pick registered aliases; `--model-id` overrides the alias with a raw Hugging Face ID.
- `--top-n` controls the number of top alternatives stored per token.
- `--inference-batch-size` and `--write-batch-size` keep memory and file sizes bounded.
- Stage 2 token similarity runs automatically after stage 1 by default; `--no-stage-2` disables it when stage 1 output is required without downstream similarity processing.
- When stage 2 runs through this command, its requested count is `count * len(selected_splits)`, so the request scales with the number of selected splits.

The stage 1 tests verify exactly these behaviors: chunk sizes, temp-file cleanup, and `state.json` updates.

## Stage 2: token similarity

[`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py) reads the stage 1 outputs and computes cosine similarity between the embedding for the ground-truth token and the embeddings for each top-logprob candidate token. It writes batch Parquet files under `token_similarities/` and updates `similarity_state.json` after each batch.

Stage 2 depends on stage 1 data already existing in `processed/{dataset}__{model}/{split}/data`. It resumes from the latest `last_id` in `similarity_state.json` or, if the state file is missing, from the max `id` found in existing similarity Parquet files.

## Derived metric exports

After stage 1 and stage 2, the repository has two useful reduction paths:

- [`scripts/logprob_metrics.py`](../scripts/logprob_metrics.py) computes scalar logprob metrics from stage 1 outputs, including the top-20 D-score, average top-20 candidate probability-mass coverage, and the count of tokens with candidate logprobs. Its output and state contract is `metrics_version = logprob_metrics_v3`.
- [`scripts/export_token_similarity_stats.py`](../scripts/export_token_similarity_stats.py) computes summary statistics from stage 2 outputs.

The top-20 D-score uses a truncated candidate distribution per token, aggregates the candidate variance across valid tokens, and returns `NaN` when no valid candidates exist or the aggregate variance is non-positive. Coverage averages the unnormalized probability mass represented by finite candidate logprobs and also returns `NaN` when no candidates are valid.

These export steps are not the main pipeline itself, but they are part of the downstream analysis workflow and should be updated when output schemas change. The metrics schema is summarized in the [domain model and data layout](../domain/data-model.md), and the calculation contract is protected by [testing guidance](../testing.md).

## What to watch out for

- Keep `TEXT_DETECTION_DATA` set before running any stage.
- The code loads models locally and may require cached weights or offline model flags.
- Resuming is based on `id`, so stable input ordering matters.
- If you change dataset aliases or split names, update the monitoring dashboard selection logic too.

## Related pages

- [`Architecture overview`](../architecture/overview.md) for the system-wide design.
- [`Domain model and data layout`](../domain/data-model.md) for output schemas and state-file fields.
- [`Testing guidance`](../testing.md) for the tests that protect these workflows.
