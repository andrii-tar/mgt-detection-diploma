# Processing Pipeline

Pipeline commands use dataset and model aliases from `scripts/experiment_config.py`. See `docs/knowledge/data-layout.md` for data layout patterns and state file fields.

## Stage 0: Dataset Preparation

Outfox ChatGPT pickle files are converted to CSV files by:

```powershell
uv run python scripts\extract_data\prepare_outfox_chatgpt.py
```

The script reads from `raw/outfox_raw` and writes to `raw/outfox` under `TEXT_DETECTION_DATA` by default.

## Stage 1: Token Logprob Processing

Stage 1 is implemented by `scripts/process_pipeline.py`. It loads a local Hugging Face causal LM, scores text rows, and writes Parquet batches under `processed/{dataset}__{model}/{split}/data`.

Small local example with Facebook OPT:

```powershell
uv run python scripts\process_pipeline.py --count 10 --model opt_350m --dataset <dataset-alias> --top-n 20 --inference-batch-size 1
```

Key arguments:

- `--count`: number of new rows to process per split.
- `--dataset`: dataset alias from `scripts/experiment_config.py`.
- `--model`: model alias from `scripts/experiment_config.py`.
- `--model-id`: raw Hugging Face model id, used instead of `--model` when provided.
- `--top-n`: number of top alternatives per token.
- `--inference-batch-size`: number of texts per forward pass.

Resume behavior for stage 1 and stage 2:

- Reads `state.json` or `similarity_state.json` when present.
- Continues after `last_id`.
- Falls back to existing Parquet max `id` if the state file is missing.

## Stage 2: Token Similarity Processing

Stage 2 is implemented by `scripts/token_similarity_stage2.py`. It reads stage 1 Parquet rows and computes cosine similarity between the ground-truth input token embedding and each top-logprob candidate token embedding.

Example:

```powershell
uv run python scripts\token_similarity_stage2.py --count 10 --model opt_350m --dataset <dataset-alias>
```

Stage 2 requires stage 1 output at:

```text
processed/{dataset}__{model}/{split}/data
```

Resume behavior:

- Reads `similarity_state.json` when present.
- Continues after stage 2 `last_id`.
- Falls back to existing token similarity Parquet max `id` if the state file is missing.

## Verification

Run tests after behavior changes:

```powershell
uv run python -m unittest discover -s tests -v
```
