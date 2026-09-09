# BBC HG-Only Evaluation Design

## Goal

Evaluate the configured classifiers on BBC News content using its available
`HG` split, with an explicitly empty `MG` class, so the experiment reports how
each model classifies human news text.

## Scope

The change is limited to the cross-dataset classification notebook and its
tests. BBC remains an evaluation-only dataset and is not added to any training
regime. The existing configured dataset alias and `HG` input path remain the
source of truth.

## Behavior

- When building BBC shared parent cohorts, request only the `HG` split from
  local-evidence metadata and processing caches.
- Do not create synthetic rows for the absent `MG` split.
- Evaluate both classifier families:
  - whole-text scalar logistic regression;
  - local-evidence logistic regression with each configured window length and
    aggregation strategy.
- Preserve the existing train/validation/test partitions of the BBC HG cohort.
- Report per scorer LLM, training regime, classifier family, partition, and
  local configuration:
  - evaluated HG rows;
  - false HG predicted as MG;
  - false-positive/error rate;
  - predicted-MG rate.
- Keep ROC-AUC, PR-AUC, and balanced accuracy undefined for BBC because the
  target has only one class.

## Implementation

The notebook's source-data loading loop will select configured source splits,
using `("HG",)` for BBC and `("HG", "MG")` for binary datasets. The existing
evaluation loops will then consume the resulting BBC cohorts through the same
whole-text and local-evidence interfaces as other evaluation regimes. The
existing `summarize_bbc_false_positives` helper will provide the compact BBC
summary used by notebook displays and plots.

## Validation

- Add or update regression coverage proving an HG-only source can be loaded
  without an MG cache and produces one-class evaluation metrics.
- Prove that BBC summary rows contain zero MG support and correctly count
  false-HG-as-MG predictions.
- Run the focused tests and full `uv run pytest -q` suite.
- Do not run large inference or processing jobs as part of validation.
