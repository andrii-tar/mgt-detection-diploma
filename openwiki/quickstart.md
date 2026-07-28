---
type: Repository Wiki
title: Text Detection OpenWiki
description: Entry point for the repository code wiki. Summarizes the text-detection pipeline, the major documentation sections, the data and model conventions, monitoring, testing, and integration points.
tags: [openwiki, quickstart, text-detection, pipeline]
resource: /README.md
---

# Text Detection OpenWiki

This wiki documents a text-detection research repository that scores human-generated (`HG`) and machine-generated (`MG`) text with local Hugging Face causal language models, then derives downstream similarity and summary metrics from the processed outputs.

Start here, then follow the section pages for deeper detail:

- [Architecture overview](architecture/overview.md) — how the staged pipeline fits together and how stateful outputs are organized.
- [Pipeline workflows](workflows/pipeline.md) — stage 0/1/2 commands, resume behavior, and what each script writes.
- [Domain model and data layout](domain/data-model.md) — dataset aliases, model aliases, experiment naming, and processed file structure.
- [Operations and monitoring](operations/monitoring.md) — Grafana, the state exporter, and the Docker Compose monitoring stack.
- [Testing guidance](testing.md) — what the tests cover and which behavior changes should trigger new coverage.
- [Integrations](integrations.md) — `TEXT_DETECTION_DATA`, local model loading, and other environment-backed dependencies.
- [Source map](source-map.md) — quick pointers from concepts to source files.

## What this repository does

The core pipeline is staged so each step can resume from state files and write results outside the repository under `TEXT_DETECTION_DATA`:

1. Stage 0 prepares dataset CSV inputs, including Outfox ChatGPT conversion from raw inputs.
2. Stage 1 runs local causal LM scoring and writes token-level logprob Parquet batches plus `state.json`.
3. Stage 2 reads the stage 1 output and computes token embedding similarity features into `token_similarities/*.parquet` plus `similarity_state.json`.
4. Monitoring exports the state files for Grafana dashboards.

The main workflow and naming rules are defined in [`scripts/experiment_config.py`](../scripts/experiment_config.py), [`scripts/process_pipeline.py`](../scripts/process_pipeline.py), and [`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py).

## Where to start when changing the repo

- Change aliases, datasets, or split paths in [`scripts/experiment_config.py`](../scripts/experiment_config.py) first.
- Change stage 1 processing in [`scripts/process_pipeline.py`](../scripts/process_pipeline.py); it owns Parquet batch writing and `state.json` updates.
- Change stage 2 similarity logic in [`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py); it depends on stage 1 outputs.
- Change scalar metric export in [`scripts/logprob_metrics.py`](../scripts/logprob_metrics.py) and [`scripts/export_token_similarity_stats.py`](../scripts/export_token_similarity_stats.py).
- Change dashboards or state export behavior in `monitoring/`.
- Change notebook helpers in the `notebooks/*_utils.py` modules rather than inside notebooks.

## Important conventions

- Keep raw datasets and generated Parquet outputs outside the repository, under `TEXT_DETECTION_DATA`.
- The current code uses `HG` for human-generated text and `MG` for machine-generated text.
- Experiment output directories are named `{dataset}__{model}` after path-safe normalization.
- Stage outputs are resumable and keyed by `last_id` in state files.
- Local model loading uses Hugging Face IDs and can run offline when `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set.

## If you are updating code

Use the documentation section that matches the change:

- Pipeline or data-flow change: [Pipeline workflows](workflows/pipeline.md)
- Alias, dataset, or path change: [Domain model and data layout](domain/data-model.md)
- Grafana or exporter change: [Operations and monitoring](operations/monitoring.md)
- CLI or environment behavior change: [Integrations](integrations.md)
- Behavior or contract change: [Testing guidance](testing.md)

## Backlog

- [Notebooks and exploratory analysis](#backlog-notebooks) — exploratory notebooks exist and likely matter, but the current evidence does not yet justify a dedicated notebook section page.
- [Utility entrypoints](#backlog-utils) — `scripts/clean_cuda.py`, `scripts/jupyter_lab.py`, and `scripts/interactive_cli.py` are useful helpers but are secondary to the staged pipeline.
