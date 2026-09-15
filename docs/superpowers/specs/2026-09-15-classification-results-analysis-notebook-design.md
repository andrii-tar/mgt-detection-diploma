# Classification Results Analysis Notebook Design

## Goal

Add a read-only notebook that analyzes one feature-vector classifier evaluation
run. It will load the run's document-classification and SemEval exports,
prepare reproducible tables and plots, and provide Markdown conclusions based on
the computed results.

The notebook is an analysis surface. It must not train models, run inference,
change result files, or regenerate evaluation outputs.

## Scope

The notebook accepts exactly one `RUN_ID`. It filters every loaded table to that
run ID and fails clearly when the run directory or matching rows do not exist.
It supports the current headered export format and the known legacy format whose
header-only placeholder has fewer columns than its appended rows.

The notebook covers:

- experiment inventory, availability, and result-data provenance;
- document-classification comparisons by feature set, classifier family,
  scorer LLM, training regime, training window, evaluation window, and
  evaluation dataset;
- aggregate comparisons by scorer LLM and by dataset;
- a matched comparison of `Mixed` and `Mixed+BBC` training configurations;
- SemEval MG-region and boundary-localization results; and
- descriptive research conclusions in Markdown cells.

The design does not add model families, run evaluations, compare multiple run
IDs, or make statistical-significance claims.

## Architecture

Create two files:

- `notebooks/analyze_classifier_results.ipynb`: a thin, output-free notebook
  that defines `RUN_ID`, configures display options, calls utility functions,
  displays returned tables and figures, and contains interpretation Markdown.
- `notebooks/utils/classification_results_analysis_utils.py`: the reusable,
  tested implementation for result loading, schema normalization, filtering,
  aggregation, plotting, and conclusion summaries.

The utility module receives paths and DataFrames explicitly. It does not depend
on notebook state and does not write files. The notebook resolves
`TEXT_DETECTION_DATA` through the existing project helper and builds the
evaluation directory at
`experiment-results/classification/evaluations/<RUN_ID>`.

## Data Loading And Validation

The loader reads document result exports when present:

- `evaluation_metrics.csv`
- `evaluation_coverage.csv`
- `evaluation_predictions.csv` when needed for diagnostics
- `run_summary.csv`

It reads SemEval result exports when present:

- `semeval_summary.csv`
- `semeval_evaluations.csv`
- `semeval_boundary_summary.csv`
- `semeval_boundary_exclusions.csv`

For a normal CSV, the loader verifies required columns, normalizes numeric
columns, filters `artifact_run_id` or `run_id` to the requested run, and records
the source as `headered`.

For a legacy CSV, the loader identifies the known header-to-row-width mismatch,
assigns the versioned positional schema used by the exporter, normalizes the
same fields, and records the source as `legacy positional recovery`. The
notebook displays a warning and provenance table whenever this path is used.
Unknown malformed layouts fail with a targeted error instead of silently
shifting columns.

All summaries exclude unavailable or malformed configurations. Missing rows are
represented as missing data, never as zero scores.

## Tables And Plots

### Inventory And Data Quality

Display tables for:

- row counts and schema/provenance by export;
- scorer LLMs, feature sets, model families, training regimes, window lengths,
  datasets, and partitions available for the run;
- coverage and unavailable configurations from `run_summary` and coverage
  exports.

### Document Classification

Use test-partition macro-F1 on binary-source datasets as the primary selection
metric. Report ROC-AUC, PR-AUC, balanced accuracy, and false-positive rate as
supporting diagnostics. Do not rank BBC using binary metrics because BBC is
HG-only.

Prepare:

- ranked configuration tables within each `(scorer_llm, evaluation_dataset)`;
- feature-set heatmaps, faceted by scorer LLM and dataset;
- training-window by evaluation-window heatmaps for each feature set;
- model-family comparisons when more than one family is present, otherwise an
  explicit scope note;
- scorer-LLM aggregate tables and plots across binary datasets;
- dataset aggregate tables and plots across scorer LLMs; and
- robust ranking tables with mean, minimum, maximum, standard deviation, and
  count of binary-source macro-F1 values for each configuration.

Plots use Matplotlib and Seaborn, matching existing notebook conventions. Each
plot has an explicit metric, aggregation level, filtering rule, and title that
names the run and partition.

### Mixed Versus Mixed+BBC Training

Match configurations on all non-regime parameters: scorer LLM, classifier
family, feature set, training window, evaluation window, evaluation dataset,
and partition. For each matched configuration, calculate the macro-F1 delta
(`Mixed+BBC - Mixed`) on binary sources.

Display a paired comparison table and a delta plot. Separately display BBC
false-positive rate and predicted-MG rate for the same configurations. Markdown
interpretation identifies whether adding BBC changes binary-source performance
and whether it reduces false positives on HG-only text. It does not combine BBC
into a pooled macro-F1 ranking.

### SemEval

Prepare tables and plots grouped by scorer LLM, feature set, training regime,
classifier family, and window length. Rank configurations separately by:

- MG-region F1, with precision, recall, and accuracy; and
- mean boundary error, with exact and tolerance boundary accuracy when present.

Show an exclusions table so unavailable configurations remain visible. State the
trade-off whenever the best region detector differs from the best boundary
localizer.

## Notebook Conclusions

Markdown cells state descriptive conclusions using values computed in preceding
cells. They cover:

- the highest and most robust document-classification configurations;
- best configuration by binary evaluation dataset;
- feature-set, window, and scorer-LLM patterns;
- `Mixed` versus `Mixed+BBC` effects, including BBC HG false-positive safety;
- leading SemEval region and boundary configurations; and
- limitations from legacy recovery, missing coverage, one seed, unavailable
  configurations, and the absence of alternative classifier families.

The notebook must use conditional language when a necessary comparison is
absent, for example: “No matched `Mixed` and `Mixed+BBC` configurations were
available in this run.”

## Testing

Add unit tests for the utility module covering:

- normal headered loading and filtering to one run;
- legacy positional schema recovery and its provenance warning;
- rejection of an unknown malformed CSV;
- binary-only aggregation that excludes HG-only BBC from macro-F1 ranking;
- matched `Mixed`/`Mixed+BBC` deltas; and
- SemEval ranking and exclusions handling.

Add a notebook contract test that checks the notebook is output-free, imports
its utility module, exposes one `RUN_ID`, and does not contain function or class
definitions. Run the focused tests and the full pytest suite after
implementation.

## Acceptance Criteria

- A researcher changes one `RUN_ID` and can analyze that run without editing
  reusable code.
- Tables and figures answer parameter, feature-set, LLM, dataset, and training
  regime questions without manually reshaping CSVs.
- Results explicitly distinguish binary evaluation from BBC HG-only safety.
- Results explicitly distinguish SemEval region quality from boundary quality.
- Legacy export recovery is conspicuous and safe; unknown layouts fail.
- The notebook is output-free and all reusable behavior is tested.
