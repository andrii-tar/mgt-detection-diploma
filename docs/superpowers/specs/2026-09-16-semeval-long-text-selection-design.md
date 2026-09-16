# SemEval Long-Text Selection Design

## Goal

Restrict SemEval best, worst, and random visualization selections to texts containing at least 200 tokens.

## Data Source

Use `token_count` from the persisted `semeval_boundary_predictions.csv` export because the evaluation ranking table does not reliably contain token lengths. Build the eligible text-ID set after loading and filtering boundary predictions to the selected run and SemEval configuration.

## Selection Behavior

Extend `select_semeval_comparison_texts` with an optional `eligible_text_ids` collection. When provided, filter the evaluation rows by those IDs before calculating aggregate MG-region F1, best/worst ranking, or random sampling. Preserve existing behavior when the argument is omitted.

The notebook will derive eligible IDs using `token_count >= 200` and pass them to the selector. The existing five best, five worst, and fixed-seed random groups remain unchanged otherwise.

## Testing

Add a utility test proving short IDs are excluded from all groups and a notebook contract test proving the 200-token eligibility filter is derived from boundary predictions and passed into selection.
