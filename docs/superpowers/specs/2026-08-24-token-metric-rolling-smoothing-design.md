# Token Metric Rolling Smoothing Design

## Goal

Add configurable rolling smoothing to token-position diagnostics without
altering the existing raw heatmaps or coverage scatter plots.

The default smoothing window is 10 tokens. Smoothed values are rendered as
additional plots with raw values retained for comparison.

## Scope

Modify the reusable heatmap helper module:

- add a NaN-aware rolling-mean utility;
- derive aligned one-dimensional token summaries from existing heatmap data;
- add an additional raw-versus-smoothed token-summary figure;
- expose `smoothing_window=10` and `show_smoothed=True` through the existing
  visualization entry point.

Update the heatmap notebook examples to configure and pass the smoothing
options. Add focused tests for numerical behavior, validation, and notebook
wiring.

Do not change the existing heatmap matrices, their scales, the raw coverage
scatter, stored processed data, or stage-1/stage-2 processing behavior.

## Rolling Semantics

Add:

```python
rolling_mean(values, *, window=10, min_periods=1) -> np.ndarray
```

The function accepts a one-dimensional numeric array-like input and returns an
array with the same length. It uses a centered rolling window, ignores NaNs,
and returns NaN when fewer than `min_periods` finite values are available in a
window. `window` must be positive and `min_periods` must be between 1 and
`window`, inclusive.

The helper applies rolling smoothing independently along the token-position
axis. It does not smooth across candidate-rank rows.

## Token Summary Series

Derive these aligned one-dimensional series from `TextHeatmapMatrices`:

- `observed_logprob`: the ground-truth row from the signed logprob matrix;
- `mean_candidate_logprob_delta`: mean finite candidate rows from the signed
  logprob matrix;
- `mean_candidate_similarity`: mean finite candidate rows from the similarity
  matrix, excluding the ground-truth row;
- `top20_coverage`: cumulative coverage vector from `compute_token_coverage`.

The summary helper returns the raw series in a named mapping. A plotting helper
uses the same mapping to draw raw and rolling-smoothed lines, preserving token
positions and NaN gaps.

## Plotting

Keep the current three heatmaps and raw coverage scatter unchanged. When
`show_smoothed=True`, `visualize_text_similarities` renders one additional
multi-panel figure:

- one panel per token summary series;
- raw values as a light line;
- rolling values as a stronger line;
- token position on the x-axis;
- metric name and smoothing window in titles/labels.

When `show_smoothed=False`, no additional summary figure is rendered. The
existing visualization call remains valid without new arguments.

Update `token_candidate_similarity_heatmaps.ipynb` with:

```python
SMOOTHING_WINDOW = 10
SHOW_SMOOTHED = True
```

and pass these values to every example invocation. Notebook cells remain
output-free.

## Errors And Compatibility

Invalid smoothing arguments raise `ValueError` with clear messages. Empty or
all-NaN input returns an all-NaN output of the same length. Existing heatmap
loading, matrix construction, scales, and function call behavior remain
compatible when the new options are omitted.

## Tests

Add tests for:

- centered rolling means and edge behavior;
- NaN omission and `min_periods` behavior;
- invalid window and `min_periods` rejection;
- token-summary series names, lengths, and candidate-row aggregation;
- smoothed plot dispatch and raw plot preservation;
- notebook smoothing parameters and argument wiring;
- output-free notebook cells.

Run the focused heatmap tests and the complete suite with the headless
Matplotlib backend.
