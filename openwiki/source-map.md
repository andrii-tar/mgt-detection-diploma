---
type: Source Map
title: Source Map
description: Quick mapping from wiki concepts to the repository files that implement them. Useful for future agents that need to jump from a conceptual page back to the authoritative source.
tags: [source-map, navigation, code-map]
resource: /scripts
---

# Source map

This page points from the wiki concepts back to the most important source files.

## Configuration and naming

- [`scripts/experiment_config.py`](../scripts/experiment_config.py) — dataset aliases, model aliases, split paths, `TEXT_DETECTION_DATA` resolution.
- [`tests/test_experiment_config.py`](../tests/test_experiment_config.py) — alias resolution coverage.
- [`tests/test_script_env_loading.py`](../tests/test_script_env_loading.py) — dotenv-backed environment loading.

## Stage 1 logprob processing

- [`scripts/process_pipeline.py`](../scripts/process_pipeline.py) — main stage 1 CLI and resume logic.
- [`scripts/openai_logprobs.py`](../scripts/openai_logprobs.py) — local causal LM scoring helper.
- [`tests/test_process_pipeline_memory.py`](../tests/test_process_pipeline_memory.py) — batch writing and state updates.
- [`tests/test_openai_logprobs.py`](../tests/test_openai_logprobs.py) — scoring helper coverage.

## Stage 2 token similarity processing

- [`scripts/token_similarity_stage2.py`](../scripts/token_similarity_stage2.py) — cosine similarity computation over top-logprob candidates.
- [`tests/test_token_similarity_stage2_progress.py`](../tests/test_token_similarity_stage2_progress.py) — state update cadence.

## Derived metrics and exports

- [`scripts/logprob_metrics.py`](../scripts/logprob_metrics.py) — scalar metrics from stage 1 outputs.
- [`scripts/export_token_similarity_stats.py`](../scripts/export_token_similarity_stats.py) — aggregate stats from stage 2 outputs.
- [`tests/test_logprob_metrics.py`](../tests/test_logprob_metrics.py) — metrics contract.
- [`tests/test_export_token_similarity_stats.py`](../tests/test_export_token_similarity_stats.py) — similarity export contract.

## Dataset preparation

- [`scripts/extract_data/prepare_outfox_chatgpt.py`](../scripts/extract_data/prepare_outfox_chatgpt.py) — Outfox pickle-to-CSV conversion.
- [`scripts/extract_data/prepare_bbc_news_archive.py`](../scripts/extract_data/prepare_bbc_news_archive.py) — BBC preparation.
- [`scripts/extract_data/prepare_mage_train.py`](../scripts/extract_data/prepare_mage_train.py) — MAGE preparation.

## Monitoring

- [`monitoring/scripts/export_processing_states.py`](../monitoring/scripts/export_processing_states.py) — state aggregation.
- [`monitoring/grafana/dashboards/processing-progress.json`](../monitoring/grafana/dashboards/processing-progress.json) — Grafana dashboard definition.
- [`tests/test_monitoring_config.py`](../tests/test_monitoring_config.py) — dashboard and URL contract.

## Notebook helpers

- [`notebooks/logprob_metrics_distributions_utils.py`](../notebooks/logprob_metrics_distributions_utils.py) — reusable notebook logic for logprob distribution analysis.
- [`notebooks/token_candidate_similarity_heatmaps_utils.py`](../notebooks/token_candidate_similarity_heatmaps_utils.py) — reusable notebook logic for token similarity visualizations.
