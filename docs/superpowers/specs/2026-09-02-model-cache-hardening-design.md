# Persisted Model Cache Hardening

## Goal

Harden persisted local-evidence model reuse against stale training data, unsafe implicit pickle deserialization, and false SemEval token reconstruction matches.

## Design

The comparison workflow computes a stable SHA-256 fingerprint from the actual `windows` and `partitions` objects used for training. When persisted-cache reuse is enabled, it derives that fingerprint unconditionally, even if the caller manifest omits the field, and uses the derived manifest for both cache loading and saving. Canonical serialization sorts rows and columns while preserving values and partition names, so equivalent input ordering produces the same digest and any training-input or partition change invalidates the manifest. With cache reuse disabled, caller manifests retain their legacy save behavior.

Pickle cache loading is opt-in. `TRUST_FITTED_MODEL_CACHE` defaults to `False` in the notebook, and the comparison API passes an explicit trust decision to the loader. The loader returns a cache miss before opening or unpickling a file unless trust is enabled. Saving and the existing reuse option remain separate, allowing users to save models without trusting a disk cache.

SemEval reconstruction preserves Unicode. It removes only observed U+FFFD replacement artifacts when they correspond to non-ASCII source characters and normalizes whitespace; compatibility remains bounded to one or two omitted characters. An inserted U+FFFD in an otherwise ASCII annotation, and meaningful Unicode substitutions, fail reconstruction validation.

## Verification

Regression tests cover deterministic fingerprints, stale-input invalidation, default-deny pickle loading, explicit trusted reuse, notebook wiring and output-free state, accepted replacement artifacts, bounded character loss, and rejection of meaningful Unicode mismatches. The affected tests and the full `uv run pytest -q` suite are required before completion.
