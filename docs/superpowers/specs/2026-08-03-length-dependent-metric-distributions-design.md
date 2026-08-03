# Length-Dependent Metric Distributions

## Goal

Create a new notebook that shows how logprob metric distributions change as more
tokens from each text are included. Human-generated (`HG`) and machine-generated
(`MG`) values must appear together in each plot.

## Scope

- Add reusable loading, truncation, recomputation, and plotting helpers to
  `notebooks/logprob_metrics_distributions_utils.py`.
- Add `notebooks/logprob_metric_length_distributions.ipynb` as a focused notebook
  based on the existing metric-distribution notebook.
- Do not modify the existing distribution notebook's behavior.
- Exclude text-derived metrics: `text_len_symbols`, `text_len_words`, and
  `text_len_sentences`.

## Data Flow

1. Load stage-1 token rows from
   `processed/{dataset}__{model}/{split}/data/*.parquet`.
2. For each row, evaluate configurable token lengths `N = step, 2 * step, ...`
   up to the row's available token count.
3. For each valid `N`, make a shallow row copy with `tokens[:N]` and pass it to
   the production `compute_logprob_metrics` function.
4. Add dataset, model, split, and truncation length metadata to the resulting
   metric record.
5. Skip any length greater than the available token count; do not pad or reuse a
   full-text metric.

The default step is 50 and is set as a notebook parameter. The recomputation
therefore uses the already stored top-20 logprobs and performs no model inference.

## Metrics

The notebook plots the available token-derived metrics:

- `prompt_mu_score`
- `prompt_sigma_score`
- `perplexity`
- `num_tokens`
- `d_score_top20`
- `d_score_top20_coverage`

Metrics missing from the loaded data or producing only missing values are skipped
with an informative message.

## Visualization

For every selected metric and dataset/model pair, create one 3D figure:

- X axis: metric-value histogram bins
- Y axis: density
- Z axis: truncation length in tokens
- HG and MG: overlaid in the same figure with distinct colors

Each truncation length has its own histogram. The density normalization is done
independently for HG and MG so their distributions remain comparable even when
the split counts differ. The notebook exposes dataset/model filters and the
length-step parameter.

## Validation

- Unit-test that token rows are truncated at each requested length and that
  lengths larger than a row are omitted.
- Unit-test that recomputed metrics match `compute_logprob_metrics` applied to the
  corresponding token prefix.
- Unit-test that text-derived fields are not included in the selected metric list.
- Verify the new notebook imports reusable helpers before calling them.
- Run the repository test suite with `uv run pytest -q`.
