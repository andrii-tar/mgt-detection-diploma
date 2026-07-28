# D-Score Notebook Exploration Design

## Goal

Extend the two tracked notebooks so D-score and D-score coverage are visible in
both aggregate experiment analysis and text-level exploration of HG/MG splits.
The analysis remains descriptive; no classifier or AUC evaluation is added.

## Scope

The metrics notebook will treat `d_score_top20` and
`d_score_top20_coverage` as first-class scalar metrics. They will appear in
the default metric selection, distribution plots, per-experiment plots, and
numeric summaries. A dedicated HG/MG comparison will expose distributions and
grouped descriptive statistics by dataset and model.

The heatmap notebook will continue to show signed candidate-vs-ground-truth
logprob and embedding similarity. It will add a token-level candidate
probability-mass coverage heatmap, plus passage-level D-score, coverage, and
valid-token-count annotations. The published D-score is passage-normalized,
so no unnormalized token quantity will be labeled as D-score.

## Architecture

Reusable loading, calculation, and plotting logic belongs in the existing
`*_utils.py` modules. Notebook cells only configure experiments, invoke those
helpers, and display returned tables or plots. The metrics utility will use
the existing metric Parquet columns. The heatmap utility will derive token
coverage from stage-1 candidate logprobs and derive passage D-score values from
the same candidate data using the production implementation where practical.

## Validation

- Unit-test coverage and heatmap matrix behavior with small synthetic rows.
- Verify both notebooks remain output-free and contain only importable,
  executable orchestration cells.
- Run the full test suite with `uv run pytest -q`.
- Run a small notebook utility smoke test against representative local data,
  without starting a large model-processing job.
