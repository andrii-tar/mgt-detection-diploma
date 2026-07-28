---
type: Integration Guide
title: Integrations
description: Environment and runtime integration notes for the repository. Covers TEXT_DETECTION_DATA, dotenv loading, Hugging Face model loading, offline flags, and the Docker-based monitoring integration.
tags: [integrations, environment, huggingface, docker]
resource: /README.md
---

# Integrations

This repository integrates with a small set of external runtime dependencies: an external data root, Hugging Face models, dotenv-based local configuration, and Docker-based monitoring.

## Data root integration

The most important integration is the external data directory referenced by `TEXT_DETECTION_DATA`. `scripts/experiment_config.py` reads this variable at import time and turns it into `DATA_ROOT`, so the environment must be configured before importing or running the pipeline.

The repository README documents the expected local paths for Windows and Linux/Vast environments.

## Hugging Face model loading

[`scripts/openai_logprobs.py`](../scripts/openai_logprobs.py) loads a local causal language model with `transformers.AutoTokenizer` and `transformers.AutoModelForCausalLM`. It selects CUDA when available and otherwise falls back to CPU.

The helper also supports offline execution via `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`. When either environment variable is set to a true-ish value, the code uses `local_files_only=True` so it will not attempt to fetch from the network.

## Pipeline and monitoring integration

The stage 1 and stage 2 scripts write their outputs under `TEXT_DETECTION_DATA`, and the monitoring stack reads those outputs through Docker Compose services. This means the runtime integration is mostly file-based rather than service-to-service API calls.

The dashboard depends on the same dataset aliases that power the pipeline, so alias changes have to propagate across code and monitoring config together.

## Local development helpers

`uv run ...` is the expected Python invocation style throughout the repo, and `pyproject.toml` declares the `j-lab` entrypoint for launching the notebook environment helper.

The existing root docs also describe Mutagen sync usage for local-to-remote file synchronization. Even though that setup is environment-specific, the important integration rule is the same: keep data and processed outputs outside the repository and do not rely on checking those artifacts into git.

## Environment checks

Relevant tests verify that:

- `experiment_config` loads `TEXT_DETECTION_DATA` through dotenv,
- the repository honors alias-to-model resolution,
- monitoring URLs and dataset variables remain aligned.

## Related pages

- [`Architecture overview`](architecture/overview.md) for how these integrations fit into the system.
- [`Pipeline workflows`](workflows/pipeline.md) for the commands that use them.
- [`Operations and monitoring`](operations/monitoring.md) for the Docker-based runtime.
