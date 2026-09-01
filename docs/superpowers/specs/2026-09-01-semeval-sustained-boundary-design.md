# SemEval Sustained Boundary Detection

## Goal

Reduce false early boundary predictions caused by a single noisy token score crossing the MG threshold.

## Design

`predict_boundary_positions` will expand each model's window scores to token positions, smooth the token score series with a centered rolling median, and select the first run of three consecutive smoothed scores at or above the configured threshold. If no run exists, the predicted boundary remains the text end, preserving the current fallback behavior.

The smoothing width and required run length will be explicit keyword parameters with defaults of 5 and 3. Existing callers can retain the current threshold and output columns while the notebook opts into the new defaults explicitly.

## Diagnostics

Boundary summaries will retain mean/median absolute token error and tolerance accuracies. The notebook will additionally report early-trigger rate, late-trigger rate, and no-sustained-crossing rate so threshold behavior is measurable rather than inferred from aggregate error alone.

## Testing

Tests will cover isolated crossings being rejected, sustained crossings being selected at the first run, smoothing suppressing a one-token spike, fallback when no run exists, and validation of invalid smoothing/run parameters. Existing boundary and notebook tests must continue to pass.
