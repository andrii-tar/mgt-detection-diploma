# Data Layout

The data root is controlled by `TEXT_DETECTION_DATA`.

Common values:

```text
D:/Projects/master_diploma/text-detection-data
/workspace/text-detection/data
```

## Raw Inputs

Kaggle train inputs:

```text
raw/kaggle_train/kaggle_train_hg.csv
raw/kaggle_train/kaggle_train_mg.csv
```

Outfox prepared inputs:

```text
raw/outfox/outfox_chatgpt_train_HG.csv
raw/outfox/outfox_chatgpt_train_MG.csv
raw/outfox/outfox_chatgpt_valid_HG.csv
raw/outfox/outfox_chatgpt_valid_MG.csv
raw/outfox/outfox_chatgpt_test_HG.csv
raw/outfox/outfox_chatgpt_test_MG.csv
raw/outfox/outfox_chatgpt_all_HG.csv
raw/outfox/outfox_chatgpt_all_MG.csv
```

Outfox raw pickle inputs are expected under:

```text
raw/outfox_raw/common/<split>/<split>_humans.pkl
raw/outfox_raw/chatgpt/<split>/<split>_lms.pkl
```

## Processed Outputs

Stage 1 writes logprob batches:

```text
processed/{dataset}__{model}/{split}/data/*.parquet
processed/{dataset}__{model}/{split}/state.json
```

Stage 2 writes token similarity batches:

```text
processed/{dataset}__{model}/{split}/token_similarities/*.parquet
processed/{dataset}__{model}/{split}/similarity_state.json
```

Example:

```text
processed/outfox_chatgpt_train__opt_350m/HG/data/batch_00000000_00000009.parquet
processed/outfox_chatgpt_train__opt_350m/HG/state.json
processed/outfox_chatgpt_train__opt_350m/HG/token_similarities/batch_00000000_00000009.parquet
processed/outfox_chatgpt_train__opt_350m/HG/similarity_state.json
```

## State Files

`state.json` tracks stage 1 progress. `similarity_state.json` tracks stage 2 progress. Both include dataset, model, split, `last_id`, `processed_count`, and `updated_at` fields.

## Legacy Note

Older notes may use `HW` for human-written data. Current code and generated Outfox files use `HG`.
