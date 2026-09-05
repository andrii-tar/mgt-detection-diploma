# Postprocessing Pair Progress Design

## Goal

Show visible progress while `scripts/postprocess_pipeline.py` processes each
dataset/model pair.

## Behavior

- Use the existing `tqdm` dependency and conventions.
- Create one progress bar for each selected dataset/model pair.
- Set the bar total to the number of discovered split directories for that
  pair.
- Label the bar with the pair's dataset and model names.
- Advance the bar once after every split, whether the split is complete or
  invalid.
- Preserve the existing per-split status output and final summary.
- Close the pair bar before rebuilding local evidence so terminal output stays
  readable.
- Keep filtering by `--dataset`, `--model`, and `--split` unchanged.

## Implementation

Wrap the existing split loop in `run_postprocessing()` with `tqdm`. The bar
will be created only after filters are applied and the split list is known.
No new dependency or public API is needed. Existing split processing remains
responsible for reporting detailed status.

## Testing

Add a unit test that replaces the progress iterator with a recording wrapper
and verifies the pair description, split total, and one update per split.
Run the focused postprocessing tests followed by the complete test suite.
