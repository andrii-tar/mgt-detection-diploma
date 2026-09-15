# Classifier Analysis Dimension Plot Matrix

## Goal

Extend classifier analysis beyond scorer-LLM coloring so the same important
plots can compare each experiment dimension independently.

## Design

Generate dimension-aware versions of the SemEval tradeoff and stability/transfer
plots for `scorer_llm`, `model_family`, `feature_set`, `training_regime`,
`training_window`, and `evaluation_window`. Each figure uses the active
dimension as its categorical color and legend, includes that dimension in the
title, and avoids per-point text labels that overlap in dense result sets.

The six dimensions are declared once in the notebook and passed to reusable
plotting helpers. Dimensions absent from a run are skipped with an availability
note. Existing plotting functions remain compatible for callers that need a
single fixed view.

## Validation

Tests cover all six dimensions, deterministic categorical colors, legend labels,
and empty or missing dimensions. The analysis notebook must execute without
errors or layout warnings, and all generated image payloads must remain valid.
