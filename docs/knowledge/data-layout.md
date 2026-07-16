# Data Layout

This file describes the data layout patterns. The authoritative source for dataset aliases, model aliases, and concrete input split paths is `scripts/experiment_config.py`.

## Data Root

`TEXT_DETECTION_DATA` points to the external data root.

```text
Local Windows: D:/Projects/master_diploma/text-detection-data
Linux/Vast:    /workspace/text-detection/data
```

Large data stays outside the repository. Do not commit raw files, processed Parquet files, model weights, or generated monitoring output.

## `scripts/experiment_config.py`

This script defines:

- `DATA_ROOT`: `Path(os.environ["TEXT_DETECTION_DATA"])`.
- `MODEL_IDS`: model alias to Hugging Face model id mapping.
- `DATASETS`: dataset alias to dataset name and split CSV paths.
- `resolve_model_id(model)`: returns a registered model id or the raw value.
- `resolve_dataset_name(dataset)`: returns configured `dataset_name`.
- `resolve_dataset_splits(dataset)`: returns split labels and CSV paths.

Inspect this file before changing dataset aliases, model aliases, split names, or input paths.

Use `--model <alias>` for models registered in `MODEL_IDS`. Use `--model-id <hugging-face-id>` only for one-off tests of models not yet registered there.

## Raw Inputs

Raw split CSV paths are configured in `DATASETS` inside `scripts/experiment_config.py`. Each configured input CSV must include stable `id` and `text` columns.

Current datasets use `HG` and `MG` split labels. Older notes may use `HW` for human-written data; current code and generated files use `HG`.

Outfox raw pickle inputs follow this pattern before conversion:

```text
raw/outfox_raw/common/<split>/<split>_humans.pkl
raw/outfox_raw/chatgpt/<split>/<split>_lms.pkl
```

Prepare Outfox CSV files with `scripts/extract_data/prepare_outfox_chatgpt.py`.

## Processed Outputs

Stage 1 logprob outputs:

```text
processed/{dataset}__{model}/{split}/data/*.parquet
processed/{dataset}__{model}/{split}/state.json
```

Stage 2 token similarity outputs:

```text
processed/{dataset}__{model}/{split}/token_similarities/*.parquet
processed/{dataset}__{model}/{split}/similarity_state.json
```

## State Files

`state.json` tracks stage 1 progress. `similarity_state.json` tracks stage 2 progress.

Both state files include:

```text
dataset
dataset_name
model
model_id
experiment_name
experiment_params
split
last_id
processed_count
updated_at
```

Stage 2 states also include:

```text
stage = token_similarity
similarity_method = input_embedding_cosine
```

## Change Checklist

When adding or renaming datasets, models, stages, or state fields:

- Update `scripts/experiment_config.py` first.
- Keep input CSVs with stable `id` and `text` columns.
- Update command examples if aliases changed.
- Update monitoring dashboard variables and export logic if Grafana should display the new values.
- Add tests for data imports, processing behavior, public interfaces, monitoring/export contracts, or changes that could silently affect experiment outputs.
