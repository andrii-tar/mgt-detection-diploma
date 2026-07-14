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
- Do not overuse TDD for small, low-risk tasks. Make simple documentation, wording, rename-reference, and config-comment changes directly without adding tests. Keep tests for data imports, processing behavior, public interfaces, monitoring/export contracts, and changes that could silently affect experiment outputs.
- Prefer small smoke runs before large runs: use low `--count` values and `--inference-batch-size 1` when memory is uncertain.
- Do not run Mutagen upload and download data sync sessions together unless the user explicitly wants bidirectional activity.
- Before changing dataset or model aliases, inspect `scripts/experiment_config.py`.
- After adding datasets or changing dataset names, model names, stages, or processing state fields, check and update monitoring docs, dashboard variables, and state export logic as needed.

## Knowledge Base

Read these docs before non-trivial changes:

- `docs/knowledge/project-overview.md`
- `docs/knowledge/data-layout.md`
- `docs/knowledge/processing-pipeline.md`
- `docs/knowledge/monitoring.md`
- `docs/knowledge/troubleshooting.md`
