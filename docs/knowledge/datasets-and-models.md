# Datasets And Models

The source of truth for dataset and model aliases is `scripts/experiment_config.py`.

## Model Aliases

Current aliases:

- `gemma_2_2b_it`: `google/gemma-2-2b-it`
- `gemma_3_1b_it`: `google/gemma-3-1b-it`
- `gpt2`: `openai-community/gpt2`
- `opt_350m`: `facebook/opt-350m`
- `smollm2_360m`: `HuggingFaceTB/SmolLM2-360M-Instruct`

Use `--model <alias>` for normal runs. Use `--model-id <hugging-face-id>` only when testing a model not yet in `MODEL_IDS`.

## Dataset Aliases

Current aliases:

- `kaggle_train`
- `outfox_chatgpt_train`
- `outfox_chatgpt_valid`
- `outfox_chatgpt_test`
- `outfox_chatgpt_all`

Each dataset entry maps split labels such as `HG` and `MG` to CSV input files under `TEXT_DETECTION_DATA/raw`.

## Adding A Dataset

1. Add raw CSV files under `TEXT_DETECTION_DATA/raw`.
2. Ensure each CSV has at least `id` and `text` columns.
3. Add an entry to `DATASETS` in `scripts/experiment_config.py`.
4. Add or update tests if the change modifies import or processing behavior.
5. Run a small smoke command with `--count 10` before a larger run.

## Adding A Model

1. Confirm the model works with `AutoTokenizer` and `AutoModelForCausalLM`.
2. Add an alias to `MODEL_IDS` in `scripts/experiment_config.py`.
3. Run a small local or remote smoke test.
4. Reduce `--inference-batch-size` if GPU memory is low.
