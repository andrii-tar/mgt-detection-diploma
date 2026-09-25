# Pooled Mixed Evaluation Design

## Goal

Remove duplicate computation from mixed-dataset classifier evaluation by sampling
one pooled cohort and producing one combined result per split, rather than
evaluating each constituent dataset independently.

## Scope

Single-dataset evaluations remain unchanged. This change applies only to the
`Mixed` (`Kaggle+MAGE+Outfox`) and `Mixed+BBC`
(`Kaggle+MAGE+Outfox+BBC`) evaluation compositions.

## Sampling

For each mixed evaluation composition, source parent rows are pooled across all
constituent datasets before cohort selection. The sampler selects at most 1,000
parent texts per partition split from the pooled rows. Parent identity remains
the tuple `(dataset, split, text_id)` so equal text IDs in different datasets do
not collide.

Sampling remains deterministic using the artifact's existing cohort seed. The
selection must preserve enough identity information to load the selected
feature rows from each source dataset and to report combined coverage.

## Evaluation Outputs

For each model artifact, mixed composition, evaluation window, and partition
(`validation` or `test`):

- Score all selected source rows in one combined evaluation pass.
- Write one prediction artifact keyed by the composition ID, not by an
  individual source dataset.
- Write one evaluation-metrics row for the combined composition.
- Write one coverage row with combined parent/row counts and HG/MG support.

The combined metric therefore represents the model's performance over the
pooled sample. Per-dataset metric rows are not emitted for mixed compositions.

Single-dataset evaluation artifacts and metrics retain their current behavior.

## Implementation Boundaries

The reusable cohort utility should support pooled sampling without changing the
default per-dataset behavior used by training and single-dataset evaluation.
The evaluator should keep source dataset identities while combining selected
rows for scoring and aggregation. Existing resume keys and artifact-completion
checks must distinguish combined composition artifacts from single-dataset
artifacts.

## Verification

- Test pooled cohort sampling is capped at 1,000 total parent texts per split.
- Test pooled identities retain dataset, split, and text ID.
- Test single-dataset cohort behavior remains unchanged.
- Test the evaluation notebook emits combined mixed artifacts/metrics rather
  than per-dataset mixed outputs.
- Run the focused classifier utility and notebook contract tests.
