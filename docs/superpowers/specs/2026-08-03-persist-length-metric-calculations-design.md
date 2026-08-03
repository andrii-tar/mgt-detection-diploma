# Persist Length Metric Calculations

## Goal

Make the length-dependent logprob metric notebook show progress during prefix
calculation and persist the calculated rows so later notebook runs do not need
to repeat the computation.

## Scope

- Add progress reporting with the existing `tqdm` dependency.
- Persist the combined length metric table as one generated Parquet file.
- Update the notebook to load the persisted result by default and recompute it
  on request.
- Keep the existing calculation helper usable without persistence.
- Do not commit generated Parquet output.

## Output

The default output path is:

```text
<TEXT_DETECTION_DATA>/processed/length_metric_distributions/metrics.parquet
```

The parent directory is created when writing. The output is overwritten on a
recomputation and written with compressed Parquet. Generated output remains
under the external data root and is not tracked by Git.

## Data Flow

1. The notebook checks the configured output path.
2. When `RECOMPUTE` is false and the output exists, it loads the Parquet file.
3. Otherwise it loads stage-1 token rows, computes all complete prefixes, and
   displays a `tqdm` progress bar over source rows.
4. It writes the resulting metric table to the output path.
5. The notebook plots the loaded or newly calculated table using the existing
   3D visualization.

The calculation remains based on stored top-20 token logprobs and performs no
model inference. The existing `step`, dataset filter, and model filter remain
the inputs controlling a recomputation. The output includes the selected metric
columns and token-length metadata; cache reuse is explicitly controlled by the
notebook's `RECOMPUTE` setting rather than inferred from file contents.

## API Design

- Add a default output-path helper for the processed data root.
- Add a persistence helper that accepts a DataFrame and output path, creates the
  parent directory, writes compressed Parquet, and returns the path.
- Add a load helper that reads a persisted table.
- Extend the table calculation with an optional progress display while
  preserving its DataFrame return value.

## Validation

- Test that prefix calculation accepts the progress-enabled path without
  changing metric results.
- Test that a calculated DataFrame survives a Parquet write/read round trip.
- Test that the output directory is created and the expected path is returned.
- Test that the notebook has a recompute switch and loads or writes the cache
  through reusable helpers rather than defining calculation logic in cells.
- Run `uv run pytest -q` and `git diff --check`.
