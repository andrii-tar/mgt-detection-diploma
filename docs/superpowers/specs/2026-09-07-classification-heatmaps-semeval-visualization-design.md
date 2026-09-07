# Classification Heatmaps and SemEval Boundary Visualization

## Goal

Extend the existing cross-dataset classification notebook with visual analysis while preserving its result tables and output-free committed state.

## Scope

- Evaluate and expose `train`, `validation`, and `test` partitions for scalar whole-text and local-evidence classification results.
- Append one notebook visualization cell that renders macro-F1 heatmaps for every scorer LLM and each partition.
- Use training dataset/regime on heatmap rows and evaluation dataset/regime on columns.
- Add SemEval boundary visualizations: aggregate boundary-accuracy bars and reproducible sample plots showing predicted and gold boundaries.
- Skip empty result frames without failing the notebook.

## Design

Existing metric and prediction frames remain the source of truth. The notebook adds train partition evaluation to the existing loops, then the appended cell pivots `macro_f1` for scalar and local-evidence frames. The heatmap cell uses a deterministic classifier/feature selection when multiple rows map to one matrix cell.

SemEval aggregate plots use `exact_accuracy`, `within_1_accuracy`, `within_3_accuracy`, and `within_5_accuracy`. Sample plots use `boundary_predictions`, select a fixed number of examples per dataset with a fixed random seed, and draw predicted and gold token boundaries over token-position axes. Missing or empty SemEval data produces no plot rather than an exception.

Contract tests inspect notebook cell ordering, partition coverage, heatmap construction, and boundary visualization references. Tests do not execute model processing or the notebook against real artifacts.

## Verification

- Notebook remains output-free.
- Focused contract and utility tests pass.
- Full `uv run pytest -q` passes.
- No stage 1/2 processing jobs or large model runs are started.
