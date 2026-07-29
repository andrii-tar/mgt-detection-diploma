# Centralize `TEXT_DETECTION_DATA` Resolution

## Goal

Provide one authoritative implementation for dotenv loading, `TEXT_DETECTION_DATA`
lookup, and the missing-configuration error. Commands run from the main checkout or
from a worktree will use the same absolute data root configured in that checkout's
`.env` file. Reusable functions will accept explicit paths and will not require the
environment merely to be imported.

## Architecture

- Add a shared data-root helper in `scripts` that loads dotenv and resolves
  `TEXT_DETECTION_DATA` to a `Path`.
- Keep the canonical error text, `TEXT_DETECTION_DATA is not set`, in that helper.
- Refactor `experiment_config` so dataset paths are derived from an explicit data
  root rather than from an import-time environment lookup.
- Preserve model and dataset alias behavior.
- Update processing and extraction command boundaries to resolve the root when
  defaults are needed, then pass explicit paths/configuration to reusable logic.

## CLI Behavior

- Commands with omitted data-dependent paths resolve the root through the shared
  helper and fail consistently when the variable is unavailable.
- Explicit input, output, processed-root, or equivalent path arguments remain usable
  without `TEXT_DETECTION_DATA`.
- Each checkout may contain a small `.env` with the same absolute
  `TEXT_DETECTION_DATA` value. No raw data or generated files are added to the repo.

## Testing

- Preserve dotenv-loading coverage.
- Test the shared resolver's success and canonical missing-variable failure paths.
- Verify reusable extraction modules can be imported without the environment.
- Verify explicit temporary roots continue to work for configuration and export/
  preparation functions.
- Run the existing test suite with `uv run pytest -q` during implementation.

## Scope

This is a configuration-resolution refactor only. Dataset aliases, model aliases,
input layouts, processing outputs, and command semantics remain unchanged. Documentation
will be updated only where `.env` setup or command behavior needs clarification.
