# AGENTS.md

## Project Summary

This repository contains utilities for text-detection research. The current pipeline processes human-generated (`HG`) and machine-generated (`MG`) text with local Hugging Face causal language models, writes token logprob outputs, then computes token embedding similarity features.

## Environment

- Python: `>=3.12`.
- Package runner: use `uv run ...` for Python commands.
- Required data root variable: `TEXT_DETECTION_DATA`.
- Local Windows data root: `D:/Projects/master_diploma/text-detection-data`.
- Linux/Vast data root: `/workspace/text-detection/data`.

PowerShell setup example:

```powershell
$env:TEXT_DETECTION_DATA = "D:/Projects/master_diploma/text-detection-data"
```

## Safety Rules

- Do not commit raw data, processed Parquet files, model weights, `.env` secrets, or generated monitoring output.
- Do not run Docker builds, data processing jobs, or git commits unless the user explicitly asks for that exact action.
- Do not start, stop, or inspect Mutagen sessions; the user runs all Mutagen commands manually.
- Do not run large model processing unless the user explicitly requests it.
- Prefer small smoke runs before large runs: use low `--count` values and `--inference-batch-size 1` when memory is uncertain.
- Do not run Mutagen upload and download data sync sessions together unless the user explicitly wants bidirectional activity.
- Before changing dataset or model aliases, inspect `scripts/experiment_config.py`.
- After adding datasets or changing dataset names, model names, stages, or processing state fields, check and update monitoring docs, dashboard variables, and state export logic as needed.

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
uv run python scripts\kaggle_process_pipeline.py --count 10 --model opt_350m --dataset outfox_chatgpt_train --top-n 20 --inference-batch-size 1
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
- `scripts/kaggle_process_pipeline.py`: stage 1 logprob processing into `data/*.parquet` plus `state.json`.
- `scripts/token_similarity_stage2.py`: stage 2 token embedding similarity processing into `token_similarities/*.parquet` plus `similarity_state.json`.
- `monitoring/scripts/export_processing_states.py`: aggregates processing state files for Grafana.
- `monitoring/grafana/`: Grafana dashboard, datasource provisioning, and monitoring documentation.
- `tests/`: unittest-based regression tests.

## Knowledge Base

Read these docs before non-trivial changes:

- `docs/knowledge/project-overview.md`
- `docs/knowledge/data-layout.md`
- `docs/knowledge/processing-pipeline.md`
- `docs/knowledge/datasets-and-models.md`
- `docs/knowledge/monitoring.md`
- `docs/knowledge/sync-and-remote.md`
- `docs/knowledge/troubleshooting.md`
