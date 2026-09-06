# Postprocessing Phase Progress Design

## Goal

Make each dataset/model progress bar advance during meaningful postprocessing
phases instead of only moving when a complete split finishes.

## Behavior

- Keep one progress bar per selected dataset/model pair.
- Report meaningful phases within each split:
  - read and compare IDs
  - Stage 1 canonicalization
  - metrics rebuild
  - Stage 2 canonicalization
  - similarity-statistics rebuild
  - local-evidence rebuild when applicable
- Advance the pair bar after each completed phase.
- Show the current phase in the bar's postfix text.
- Preserve existing split status lines, summary accounting, filters, and return
  codes.
- Do not add dependencies or row-level progress overhead.

## Implementation

Pass a small phase-reporting callback into split processing and local-evidence
rebuilding. The callback updates the existing pair-level `tqdm` bar and its
current phase label. The split processor remains responsible for deciding
which phases apply based on available inputs and ID validity.

## Testing

Extend the postprocessing progress tests with a recording bar and callback
assertions. Verify that phase updates are emitted in order, that unavailable
optional phases are not reported, and that existing split accounting remains
unchanged. Run focused postprocessing tests and the complete test suite.
