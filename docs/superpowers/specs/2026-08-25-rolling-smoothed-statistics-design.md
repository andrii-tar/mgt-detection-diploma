# Rolling Smoothed Statistics

## Goal

Show the mean, standard deviation, and variance of each rolling-smoothed token
metric in the existing smoothed summary plots.

## Design

`plot_smoothed_token_summaries` will calculate statistics from each metric's
rolling-smoothed values after applying the existing `window` argument. NaN
values will be excluded. The plotted series and function return behavior will
remain unchanged.

Each subplot title will include the smoothing window and formatted values for
mean, standard deviation, and variance. Calling the function with
`window=5` therefore reports statistics for the five-position rolling result.

## Verification

Focused tests will verify the numerical values and the presence of the
statistics in the generated plot titles. The existing notebook utility test
suite and full test suite will then be run.
