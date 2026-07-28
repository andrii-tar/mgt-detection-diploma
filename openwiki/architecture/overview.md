---
type: Architecture Overview
title: Architecture Overview
description: High-level architecture for the text-detection repository. Explains the staged processing pipeline, the external data root, the resumable state files, and the monitoring/export path that turns processing state into Grafana-visible data.
tags: [architecture, pipeline, monitoring, state]
resource: /scripts/process_pipeline.py
---

# Architecture overview

The repository is organized around a staged, resumable research pipeline rather than a single monolithic job. The key architectural idea is that each stage writes compact, append-friendly artifacts under `TEXT_DETECTION_DATA`, and later stages and monitoring jobs read those artifacts instead of recomputing from scratch.

## Core flow

`[raw CSV inputs] -> [stage 1 logprob batches] -> [stage 2 similarity batches] -> [metric exports] -> [monitoring/state export]`

The orchestration path is implemented by [`scripts/process_pipeline.py`](../scripts/process_pipeline.py) and [`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py). The code uses the alias and split definitions from [`scripts/experiment_config.py`](../scripts/experiment_config.py), which is the source of truth for dataset names, model aliases, and input CSV locations.

## Why the design looks this way

The current codebase favors resumability and reproducibility over transient in-memory processing:

- Stage 1 writes one Parquet batch at a time and tracks progress in `state.json`.
- Stage 2 reads stage 1 outputs, writes its own batch files, and tracks progress in `similarity_state.json`.
- Scalar metric exports and monitoring jobs operate on those persisted artifacts.

That structure is visible in the tests as well: the pipeline tests assert batch sizing, resume updates, and state-file evolution rather than only checking return values.

## Main architectural components

### Configuration and naming

[`scripts/experiment_config.py`](../scripts/experiment_config.py) resolves `TEXT_DETECTION_DATA`, maps short model aliases to Hugging Face IDs, and defines dataset split paths. The output directory name is derived from `{dataset}__{model}` after path-safe normalization, so every downstream component uses the same experiment namespace.

### Stage 1 processing

[`scripts/process_pipeline.py`](../scripts/process_pipeline.py) loads a local causal LM through [`scripts/openai_logprobs.py`](../scripts/openai_logprobs.py), scores text rows, writes token-level logprob Parquet batches, and stores progress in `state.json`.

### Stage 2 similarity computation

[`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py) loads the same model's input embeddings, normalizes them, and computes cosine similarity for the ground-truth token against each top-logprob candidate. It writes the similarity batches and `similarity_state.json`.

### Derived metric exports

[`scripts/logprob_metrics.py`](../scripts/logprob_metrics.py) converts stage 1 token outputs into scalar metrics such as perplexity and text-length controls. [`scripts/export_token_similarity_stats.py`](../scripts/export_token_similarity_stats.py) reduces token similarity batches into aggregate statistics for analysis or downstream plotting.

### Monitoring path

The monitoring stack in `monitoring/` reads the `state.json` and `similarity_state.json` files and exposes them through a small HTTP service for Grafana. The dashboard dataset selector is kept in sync with `scripts/experiment_config.py`, which is why dataset alias changes must touch both the pipeline and monitoring layers.

## Cross-page relationships

- [`Pipeline workflows`](../workflows/pipeline.md) explains the operational meaning of this architecture.
- [`Domain model and data layout`](../domain/data-model.md) explains the experiment names, state fields, and file layout that the architecture depends on.
- [`Operations and monitoring`](../operations/monitoring.md) explains how the persisted state becomes a visible dashboard.
- [`Testing guidance`](../testing.md) explains which contract tests protect the architecture.
