# Persisted Local-Evidence Models

## Goal

Allow the comparison notebook to save fitted classifiers and reload them on later runs, avoiding repeated training when the training configuration is unchanged.

## Storage

Persist models under `TEXT_DETECTION_DATA/processed/{dataset}__{model}/local_evidence/models/`. Each cache contains one serialized fitted model per `(classifier, window_length)` and a manifest describing the dataset/model identity, feature columns, seed, window lengths, and classifier configuration.

## Reuse Contract

The notebook exposes `SAVE_FITTED_MODELS` and `REUSE_FITTED_MODELS` options. Reuse is enabled only when a cache entry and manifest exactly match the current training identity and configuration. Missing, malformed, or incompatible entries are ignored and retrained. Newly fitted models are saved atomically after successful fitting when saving is enabled.

## Compatibility and Safety

The existing in-memory `fitted_models` registry remains supported for same-run SemEval reuse. Disk persistence is an optional notebook workflow and does not alter classifier output schemas. Cache files are generated data and remain outside git. Pickle loading is a trusted-operation boundary: callers must explicitly set the trust option, which defaults to false, before any cache bytes are deserialized. Loading is restricted to the expected cache directory and failures fall back to retraining with a visible progress message. The manifest also contains a stable fingerprint of the actual windows and partitions used for training, not merely notebook constants.

## Testing

Tests will cover manifest matching, incompatible-cache invalidation, round-trip model persistence, atomic writes, and notebook configuration wiring. Existing model comparison and SemEval tests must continue to pass.
