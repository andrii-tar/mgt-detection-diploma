# Name That LLM Dataset Preparation Design

## Objective

Add the Name That LLM training data as a reproducible, MG-only dataset input for
the text-detection processing pipeline. Each generated text must retain the
source model name in the standard `source_llm` metadata column.

## Input and Output

The preparation script will use these defaults under `TEXT_DETECTION_DATA`:

```text
Input:  raw/name-that-llm/train.csv
Output: raw/name-that-llm/name_that_llm_train_MG.csv
```

The input is expected to contain `id`, `prompt_id`, `text`, and `model` columns.
The output will contain, in this order:

```text
id, source_id, text, prompt_id, source_llm, origin, source_dataset
```

The pipeline `id` will be a zero-based integer sequence because all processing
stages use integer checkpoint comparisons. The original string IDs, including
leading zeroes, will be preserved in `source_id`. The `model` value will be
copied to `source_llm`; `origin` will be `MG` for every row; and
`source_dataset` will be `name_that_llm` for every row.

## Implementation

Add `scripts/extract_data/prepare_name_that_llm.py` following the existing
dataset preparation script conventions. It will:

1. Accept optional `--input-path` and `--output-path` arguments.
2. Resolve omitted paths from the shared data-root helper.
3. Validate that the input exists and has all required columns.
4. Reject duplicate source IDs, blank text values, and blank model values.
5. Create the output parent directory and write the transformed CSV.

Register the dataset alias `name_that_llm_train` in
`scripts/experiment_config.py` with one `MG` split pointing to the generated
CSV. No HG split will be added.

## Testing

Add focused tests using temporary CSV files. Tests will cover:

- Successful conversion and exact output column order.
- Preservation of string IDs and `prompt_id` values.
- Copying each input model value to `source_llm`.
- Constant `MG` and `name_that_llm` metadata.
- Rejection of missing columns, duplicate IDs, blank text, and blank models.

The implementation will be verified with the focused tests and the relevant
configuration/import checks. Raw data and generated CSV output remain outside
the repository.
