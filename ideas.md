# Local Evidence Classification

The project currently visualizes token-level candidate log-probability,
embedding-similarity, and top-candidate-coverage signals. The next research
direction is to turn these signals into local evidence detectors: find words,
phrases, or regions whose feature patterns provide strong evidence for
human-generated (`HG`) or machine-generated (`MG`) text. A document prediction
should therefore depend on detected evidence regions rather than only on
whole-text means and standard deviations.

## Shared Experiment Design

- Generate overlapping rolling token windows across every text. Window lengths
  will be varied around paragraph scale to measure how local context affects
  precision; natural paragraph boundaries may group adjacent evidence windows
  for presentation, but do not constrain detection.
- Derive local features from the existing aligned token sequences: observed
  log-probability, candidate log-probability delta, candidate embedding
  similarity, and top-candidate coverage. Candidate summary features include
  local means, standard deviations, quantiles, extrema, trends, and missing
  value rates.
- Split the data by original text before generating windows. All windows of a
  text must remain in the same train, validation, or test partition.
- Score each window for HG and MG evidence. Compare two text-level aggregation
  rules: the strongest evidence window, and the amount of text whose evidence
  exceeds a class-specific threshold.
- Select the smallest effective window length and aggregation thresholds on the
  validation set only, then report a single final result on the held-out test
  set. Report eligibility, class support, and precision/recall at every tested
  length so that changed text populations do not masquerade as a length effect.

## Methods To Compare

| Method | Local score | Explanation | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| Standardized logistic regression | Window-level MG probability from pooled local features | Feature contribution is standardized value multiplied by its coefficient | Simple, calibrated, and directly interpretable | Linear decision boundary; windows inherit weak labels from their parent text |
| Class-conditional Gaussian mixtures | HG/MG log-likelihood ratio for a window | Most likely class component and feature deviations from that component | Represents several local writing-style modes | Requires covariance regularization and careful probability calibration |
| Multiple-instance learning | Learns which windows in a labeled text bag support the text label | Attention or selected-window scores | Matches the assumption that only some regions determine the text label | More complex and less transparent; evaluate after the simpler baselines |

Start with logistic regression and both aggregation rules. The feature-table
and aggregation interfaces should remain model-agnostic so Gaussian-mixture or
Bayesian models can be added without changing the stored window data or the
evaluation protocol.

## Interpretation Outputs

For every text-level decision, retain the window scores and the regions that
contributed to the result. A result should expose the source `text_id`, token
start and end positions, reconstructed text span or paragraph, predicted
label, class score, aggregation result, raw local feature values, and the
largest positive and negative contributions. The existing token heatmaps can
then render those ranges for inspection.

Because labels are available only per full text, highlighted regions are model
evidence for HG or MG, not proof that a region was independently authored by
that class. Align stage-1 and stage-2 records by text ID, token position, and
token ID before extracting features; missing or mismatched records must fail
explicitly rather than be silently truncated.

## Recommended Sequence

1. Correct the rolling observed-logprob summary so it uses the saved
   ground-truth token log-probabilities rather than the heatmap baseline row.
2. Add validated rolling-window extraction and a versioned window-level feature
   table with parent text IDs and token boundaries.
3. Establish leakage-safe logistic-regression baselines for strongest-region
   and evidence-count aggregation across a predefined range of window lengths.
4. Add per-window and per-text contribution reports and connect selected ranges
   to the heatmap notebook.
5. Compare class-conditional Gaussian mixtures, then consider
   multiple-instance learning only if simpler models do not capture the local
   patterns.
