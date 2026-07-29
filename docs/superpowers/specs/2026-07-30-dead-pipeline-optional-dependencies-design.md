# Dead Pipeline Helpers and Optional Dependencies

## Goal

Make the supported installation and command surface explicit while removing
confirmed-dead pipeline helpers. A default installation must support dataset
preparation and local processing without notebook-only packages. Notebook and
visual-analysis workflows must remain installable through an explicit optional
dependency group.

## Scope

- Remove `_write_parquet_batches` from `scripts/process_pipeline.py` and
  `scripts/token_similarity_stage2.py` after confirming there are no callers.
- Keep the existing preparation, stage 1, stage 2, metrics, export, and other
  utility modules in the repository.
- Remove `j-lab` from the package's declared console-script surface. The
  `scripts/jupyter_lab.py` module is not advertised as a supported package
  entrypoint.
- Keep the supported command surface documented through direct script or
  package-module invocations.

## Dependency Layout

`pyproject.toml` will define three explicit layers:

- Default core dependencies: `accelerate`, `pandas`, `pyarrow`,
  `python-dotenv`, `torch`, `transformers`, and `tqdm`. These cover dataset
  preparation, stage 1, stage 2, and core data utilities.
- Project optional extra `notebook`: Jupyter, Jupyter collaboration, and the
  plotting/analysis dependencies used by notebooks, including Matplotlib,
  Seaborn, and NumPy where it is a direct notebook import.
- Development dependency group `dev`: pytest.

The default `uv sync` must not install notebook-only packages. Notebook users
will use `uv sync --extra notebook` in addition to the core dependencies.

## Documentation and Supported Commands

README and knowledge documentation will:

- State that preparation and processing commands are the primary supported
  entrypoints.
- Preserve examples for dataset preparation, stage 1, and stage 2.
- Document `uv sync --extra notebook` for notebook workflows.
- Remove any claim that `j-lab` is a declared project entrypoint.

Documentation changes will be limited to behavior required by the new package
metadata and command surface. Existing data-layout, processing, monitoring,
and resume semantics remain unchanged.

## Verification

- Repository-wide static search finds no definition or caller for either
  removed helper.
- Metadata tests verify the core dependency set, the `notebook` optional extra,
  the `dev` group, and absence of the `j-lab` console script.
- Existing package-module help tests continue to pass for the stage 1 and
  stage 2 commands.
- The full test suite passes with `uv run pytest -q`.
- Where the environment permits, verify that core command imports/help do not
  require notebook-only packages.

No model inference, data processing, Docker, or generated-data operation is
part of this change.
