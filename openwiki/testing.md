---
type: Testing Guide
title: Testing Guidance
description: Practical testing guide for the repository's pipeline and monitoring contracts. Summarizes the existing unittest coverage, the kinds of behavior that should be tested when changing scripts, and the standard test command.
tags: [testing, pytest, unittest, contracts]
resource: /tests
---

# Testing guidance

The repository's tests are mostly contract tests: they verify CLI/environment loading, data-layout assumptions, batch sizing, resume behavior, and monitoring configuration.

## Test command

The README and knowledge docs use `uv run python -m unittest discover -s tests -v` as the standard repository-wide test command.

## What the tests cover

### Configuration and environment loading

- [`tests/test_experiment_config.py`](../tests/test_experiment_config.py) checks model alias resolution.
- [`tests/test_script_env_loading.py`](../tests/test_script_env_loading.py) checks that `experiment_config` loads `TEXT_DETECTION_DATA` via dotenv.

### Stage 1 processing

- [`tests/test_process_pipeline_memory.py`](../tests/test_process_pipeline_memory.py) checks that stage 1 writes chunked Parquet batches, cleans up temp files, and updates `state.json` correctly.
- [`tests/test_logprob_metrics.py`](../tests/test_logprob_metrics.py) and [`tests/test_logprob_metrics_distributions_utils.py`](../tests/test_logprob_metrics_distributions_utils.py) protect downstream metric logic.

### Stage 2 processing

- [`tests/test_token_similarity_stage2_progress.py`](../tests/test_token_similarity_stage2_progress.py) checks that stage 2 updates `similarity_state.json` after each similarity batch.

### Metrics export and utilities

- [`tests/test_logprob_metrics.py`](../tests/test_logprob_metrics.py) covers scalar metrics, including the top-20 D-score's truncated weighted distribution, passage-level variance aggregation, `NaN` behavior without valid candidates, top-20 probability-mass coverage, and the `d_score_top20_num_tokens` count.
- [`tests/test_export_token_similarity_stats.py`](../tests/test_export_token_similarity_stats.py) covers the token similarity summary export path.
- [`tests/test_openai_logprobs.py`](../tests/test_openai_logprobs.py) covers local scoring behavior in the logprob helper.

### Monitoring

- [`tests/test_monitoring_config.py`](../tests/test_monitoring_config.py) checks the dashboard URL, HTTP scheme, and dataset variable contents.
- [`tests/test_export_processing_states.py`](../tests/test_export_processing_states.py) covers the state export behavior used by monitoring.

### Dataset preparation

- [`tests/test_prepare_bbc_news_archive.py`](../tests/test_prepare_bbc_news_archive.py), [`tests/test_prepare_mage_train.py`](../tests/test_prepare_mage_train.py), and [`tests/test_prepare_outfox_chatgpt.py`](../tests/test_prepare_outfox_chatgpt.py) protect input preparation jobs.

## When to add or change tests

Add or update tests whenever you change any of the following:

- dataset aliases, model aliases, or split names,
- state file fields or resume logic,
- batch sizing or output file naming,
- metric column definitions,
- monitoring URLs, dashboard variables, or exporter paths,
- dotenv or environment-variable loading,
- notebook helper functions that are imported from tests or scripts.

The tests are especially valuable for anything that could silently change experiment outputs.

## Fast verification mindset

The repository favors small smoke tests over large model runs. For behavior changes, prefer:

1. unit tests for the affected script or helper,
2. a tiny end-to-end smoke run only when necessary,
3. larger processing runs only when the user explicitly requests them.

## Related pages

- [`Architecture overview`](../architecture/overview.md) for the system view.
- [`Pipeline workflows`](../workflows/pipeline.md) for the main behavior being tested.
- [`Operations and monitoring`](../operations/monitoring.md) for dashboard contracts.
