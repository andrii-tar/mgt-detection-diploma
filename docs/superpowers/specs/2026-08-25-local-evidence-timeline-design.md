# Local Evidence Timeline Design

## Scope

Add a fast, notebook-facing visualization for local-evidence classification.

## Visualization

A reusable utility will plot each window's MG score against its token midpoint.
Windows at or above the fitted evidence threshold will be shaded as suspicious
regions. The plot title will state the text ID, true label, prediction,
aggregation, document score, and document threshold.

The notebook will render this plot for three selections:

- The test prediction with the highest MG-oriented score.
- The most confident incorrect test prediction, when one exists.
- A manual `INSPECT_TEXT` selection, when configured.

For each selected text, the notebook will also invoke the existing token
similarity heatmap for the highest-scoring highlighted window.

## Related Corrections

- Represent a single selected window length as `(32,)`, not `(32)`.
- Add default-enabled, opt-out `tqdm` progress reporting to
  `run_local_evidence_experiments(...)`, with one bar per window length.

## Testing

Test timeline selection and evidence-region shading with synthetic scored
windows. Test the three notebook inspection controls structurally and retain
the output-free notebook contract. Test enabled and disabled experiment
progress reporting.
