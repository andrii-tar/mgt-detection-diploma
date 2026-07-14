# Project Overview

This project supports text-detection experiments for comparing human-generated and machine-generated text. It uses local Hugging Face causal language models to produce token-level logprob data and derived token similarity features.

## Main Concepts

- `HG`: human-generated or human-written text in the current code and file names.
- `MG`: machine-generated text.
- Dataset alias: the key used by commands, defined in `scripts/experiment_config.py`.
- Model alias: the short model key used by commands, defined in `scripts/experiment_config.py`.
- Experiment name: `{dataset}__{model}` after path-safe normalization.

## Current Pipeline

1. Prepare raw dataset CSV files under `TEXT_DETECTION_DATA/raw`.
2. Run stage 1 to score texts with a local language model and write token logprob Parquet batches.
3. Run stage 2 to compute token embedding cosine similarities for stage 1 outputs.
4. Export processing state files for Grafana monitoring.

## Important Boundaries

- Code lives in this repository.
- Large data lives outside the repository under `TEXT_DETECTION_DATA`.
- Model weights are loaded through Hugging Face tooling and are not stored in this repository.
- Monitoring reads state files only; it does not process model outputs directly.
