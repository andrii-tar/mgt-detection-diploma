# SemEval Subtask C Preparation Design

## Goal

Prepare SemEval-2024 Task 8 Subtask C train and development JSONL files for the existing logprob pipeline without changing that pipeline.

## Input

The official files are `subtaskC_train.jsonl` and `subtaskC_dev.jsonl`. Each record has a UUID `id`, complete mixed `text`, and integer `label`. `label` is the machine-generated boundary in `text.split(" ")`: words before the boundary are human-written and words from it onward are machine-generated. A boundary of zero is valid and means the text has no human-written prefix.

## Output

The converter writes `SemEval2024_task8_C_train_MG.csv` and `SemEval2024_task8_C_dev_MG.csv` under the raw SemEval directory. Both contain all original texts and these columns: numeric `id`, `text`, `boundary_word_index`, `source_id`, `split`, `origin`, and `source_dataset`.

`id` is a stable sequential integer because the existing pipeline requires integer IDs. `source_id` retains the source UUID. `origin` is `MG` because all full sequences are deliberately processed through the existing `MG` split. The boundary metadata remains in the prepared CSV and validation joins it to processed Parquet rows by numeric `id`.

## Integration and Validation

Two dataset aliases expose the train and development CSVs with an `MG`-only split. The converter rejects malformed JSONL records, blank text, non-string UUIDs, non-integer boundaries, duplicate UUIDs, and boundaries outside the inclusive range from zero to `len(text.split(" ")) - 1`.

Tests cover output metadata, exact boundary preservation, valid zero boundaries, invalid input rejection, and dataset-alias paths. No processing or raw data files are committed.
