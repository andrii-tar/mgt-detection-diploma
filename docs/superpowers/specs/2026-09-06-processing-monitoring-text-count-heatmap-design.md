# Processed Text Count Heatmap

## Goal

Add a monitoring-notebook heatmap showing the number of processed texts in each split for every dataset/model pair.

## Design

The chart will derive counts from the existing `readiness` DataFrame and its `stage1_rows` column. It will reshape the data into dataset/model rows and split columns, then render a Seaborn heatmap with exact counts annotated in each cell. Missing dataset/model/split combinations remain blank rather than being treated as zero.

The chart will be added to the existing `Readiness Overview` code cell. Existing readiness and artifact-count charts remain unchanged. No utility-module or generated-output changes are required.

## Validation

Update the notebook source-contract tests to require the processed-text count pivot and heatmap annotations. The notebook must remain output-free and all existing tests must pass.
