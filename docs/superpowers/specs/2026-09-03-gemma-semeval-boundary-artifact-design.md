# Gemma SemEval Boundary Artifact Design

## Goal

Allow SemEval boundary evaluation to align Gemma token labels when a non-breaking
space is serialized as consecutive replacement characters, without accepting
unrelated text mismatches.

## Design

The boundary utility will use a separate raw-text normalization for
replacement-artifact matching. It will collapse ASCII whitespace only and retain
non-breaking spaces, allowing the existing replacement-artifact matcher to map a
consecutive replacement-character run to that original non-ASCII character.

The existing canonical comparison remains responsible for ordinary whitespace
normalization. Meaningful content mismatches remain errors. `add_gold_boundaries`
will add the SemEval text ID to reconstruction errors so future invalid processed
artifacts can be diagnosed directly.

## Verification

Add a unit test covering a non-breaking space reconstructed as two replacement
characters. Run the focused SemEval boundary utility tests and verify all 270
available Gemma SemEval token-label records map to their annotations.
