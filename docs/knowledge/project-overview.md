# Project Overview

This repository contains utilities for text-detection research. The current pipeline processes human-generated (`HG`) and machine-generated (`MG`) text with local Hugging Face causal language models, writes token logprob outputs, then computes token embedding similarity features.

## Main Concepts

- `HG`: human-generated or human-written text in the current code and file names.
- `MG`: machine-generated text.
- Dataset alias: the key used by commands, defined in `scripts/experiment_config.py`.
- Model alias: the short model key used by commands, defined in `scripts/experiment_config.py`.
- Experiment name: `{dataset}__{model}` after path-safe normalization.

## Pipeline idea
The pipeline is staged so each step can resume from state files and write outputs under `TEXT_DETECTION_DATA` instead of the repository.

- Stage 0: prepare dataset CSV inputs, currently including Outfox ChatGPT conversion from pickle files.
- Stage 1: score text with a causal LM and write token logprob Parquet batches plus `state.json`.
- Stage 2: compute token embedding similarity features from stage 1 outputs and write `similarity_state.json`.
- Monitoring: aggregate stage state files for Grafana.

## Common Commands

Run all tests:

```powershell
uv run python -m unittest discover -s tests -v
```

Prepare Outfox ChatGPT CSV inputs:

```powershell
uv run python scripts\prepare_outfox_chatgpt.py
```

Run stage 1 logprob processing:

```powershell
uv run python scripts\process_pipeline.py --count 10 --model opt_350m --dataset outfox_chatgpt_train --top-n 20 --inference-batch-size 1
```

Run stage 2 token similarity processing after stage 1 exists:

```powershell
uv run python scripts\token_similarity_stage2.py --count 10 --model opt_350m --dataset outfox_chatgpt_train
```

Start monitoring:

```powershell
docker compose up -d
```

## Code Map

- `scripts/experiment_config.py`: source of truth for model aliases, dataset aliases, and input split paths.
- `scripts/prepare_outfox_chatgpt.py`: converts Outfox pickle files into project CSV split files.
- `scripts/process_pipeline.py`: stage 1 logprob processing into `data/*.parquet` plus `state.json`.
- `scripts/token_similarity_stage2.py`: stage 2 token embedding similarity processing into `token_similarities/*.parquet` plus `similarity_state.json`.
- `monitoring/scripts/export_processing_states.py`: aggregates processing state files for Grafana.
- `monitoring/grafana/`: Grafana dashboard, datasource provisioning, and monitoring documentation.
- `tests/`: unittest-based regression tests.
