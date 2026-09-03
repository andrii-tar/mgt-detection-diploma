# Cross-Dataset Classification PoC

## Goal

Create a small, reproducible notebook experiment comparing cross-dataset text classification with the processed metrics available for `smollm2_360m` and `opt_350m`.

## Scope

The experiment uses only existing metric Parquet files under `TEXT_DETECTION_DATA/processed`. It does not run Stage 1 or Stage 2 processing and does not modify the production pipeline. The implementation is contained in a new notebook; reusable production utilities are out of scope for this proof of concept.

The datasets are Kaggle (`kaggle_train`), Outfox (`outfox_chatgpt_train`), and MAGE (`mage_train`). The two language models are `smollm2_360m` and `opt_350m`. The notebook reports the available HG/MG row counts before training. A missing model/dataset combination is represented by `NaN` in the result matrix rather than causing processing or silently substituting another model.

## Experiment

For each language model, filter the loaded metrics to the three datasets and HG/MG labels. Use the common scalar feature columns available in every included dataset. Convert HG to label 0 and MG to label 1, discard rows with non-finite feature values, and create deterministic stratified train/validation/test partitions independently within each dataset.

Construct four training/evaluation regimes: each individual dataset and `Mixed`, where the mixed partition is the concatenation of the corresponding partitions from datasets with available data. Fit a scaled, class-balanced logistic-regression classifier on each training regime. Evaluate every classifier on every compatible evaluation regime using test-set macro-F1. A matrix has training regime on rows and evaluation regime on columns. The notebook produces one annotated heatmap per language model, with blank annotations for missing combinations.

The PoC uses a fixed seed and does not tune thresholds on the test set. The validation partitions are retained for consistency with the existing classification workflow, although the simple classifier uses the default 0.5 probability threshold.

## Outputs

The notebook displays:

- Selected datasets/models and HG/MG coverage counts.
- A tidy table containing model, training regime, evaluation regime, row counts, and macro-F1.
- One macro-F1 heatmap for each language model.

Generated tables and figures remain notebook/runtime artifacts and are not committed with stored outputs.

## Verification

The notebook source must remain output-free. A source-level test should verify the new notebook includes both models, all four regimes, macro-F1 calculation, and heatmap plotting. Existing tests must continue to pass.
