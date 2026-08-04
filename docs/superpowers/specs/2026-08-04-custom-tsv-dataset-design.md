# Custom TSV Dataset Import Design

## Goal

Convert the custom TSV corpus under the external `TEXT_DETECTION_DATA` root into the CSV split format consumed by the processing pipeline.

## Input

The importer accepts one TSV file through `--input-path`. The file must contain:

- `LLM_origin`: source identifier; the literal value `human` identifies human-written text.
- `text`: source document text.

The default input is `raw/custom/samples_2025/custom_examples.tsv` relative to the external data root. This is the only source file for the dataset.

## Classification and Output

- Rows with `LLM_origin == "human"` are written to the `HG` split.
- All other rows are written to the `MG` split.
- Each split receives sequential zero-based `id` values independently.
- The output preserves `id`, `text`, and `LLM_origin` columns.
- Outputs are written below the external data root by default:
  - `raw/custom/samples_2025/custom_2025_HG.csv`
  - `raw/custom/samples_2025/custom_2025_MG.csv`

The output root is configurable with `--output-root`.

## Integration

Add a `custom_2025` dataset alias to `scripts/experiment_config.py` with `HG` and `MG` paths pointing to the generated CSV files. The existing runtime data-root resolution must be used so the importer works with `TEXT_DETECTION_DATA` without writing repository-local data.

## Interface

Add `scripts/extract_data/prepare_custom.py` with a reusable preparation function and a CLI:

```powershell
uv run python scripts\extract_data\prepare_custom.py
uv run python scripts\extract_data\prepare_custom.py --input-path <path> --output-root <path>
```

## Validation

The importer must fail clearly when required columns are missing. Tests will verify split classification, metadata preservation, independent IDs, and invalid-input handling using temporary files. The full test suite must pass after implementation.
