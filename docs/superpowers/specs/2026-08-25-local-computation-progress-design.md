# Local Computation Progress Design

## Scope

Add terminal progress reporting to the two token-level computations that can
take longer than 20 seconds:

- `build_local_evidence_windows(...)`
- `calculate_rolling_statistics(...)`

## Interface

Each function will accept `show_progress: bool = True`. Callers can pass
`False` to suppress output in tests and non-interactive use.

## Behavior

For each requested split, the function will show one `tqdm` progress bar while
processing stage-1 text rows. The bar will use a split-specific description and
`text` as its unit. It will advance once for every stage-1 text processed.

Similarity rows are indexed before the text loop and will not receive a
separate progress bar, keeping notebook output concise.

## Testing

Tests will verify that each function creates split-specific bars with the
expected description and unit when enabled, and produces no bars when
`show_progress=False`.
