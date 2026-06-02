# AGENTS_DATA.md

## Data usage overview

This project uses a staged data layout to keep raw sources, extracted text slices, and processed model outputs separate. The flow is:

raw datasets -> extracted texts -> processed outputs

The key split is by origin:
- HW = human written
- MG = machine generated

Datasets are versioned by name and modification:
{dataset_name}__{modification}

Example: dataset_1__full, dataset_1__first_10_sent

## Directory tree (expected layout)

data_base_dir/
  raw/
    dataset_1_downloaded
    dataset_2_downloaded
    dataset_3_downloaded
    dataset_4_generated
    ...
  extracted/
    dataset_1__full/
      MG/
        texts.csv
      HW/
        texts.csv
    dataset_1__first_10_sent/
      MG/
      HW/
    dataset_1__first_paragraph/
      MG/
      HW/
    dataset_2__full/
      MG/
      HW/
    ...
  processed/
    {dataset_name}__{modification}/
      HW/
        data.csv
        metrics.csv
      MG/
        data.csv
        metrics.csv
    ...
  dataset_mapping.csv

## Data flow and file roles

1) raw/
   - Original downloaded or generated datasets.

2) extracted/
   - Text segments prepared for processing.
   - Each dataset version has MG and HW splits.
   - texts.csv schema:
     - id
     - document_id
     - raw_text_id
     - start_pos
     - end_pos
     - text

3) processed/
   - Model-derived outputs for each dataset version.
   - Split by MG and HW.
   - data.csv schema:
     - id
     - dataset_id
     - text_id
     - logprobs_100
     - logprobs_20
   - metrics.csv schema:
     - id
     - dataset_id
     - text_id
     - perplexity_100
     - d_score_100
     - perplexity_20
     - d_score_20
     - ...

4) dataset_mapping.csv
   - Maps dataset_full_name to a numeric dataset_id and a dataset_label for plots.
   - Fields:
     - dataset_full_name: {dataset_name}__{modification}
     - dataset_id: int
     - dataset_label: short string for visualization
