from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"


def test_notebook_utilities_are_package_modules() -> None:
    assert list(NOTEBOOKS_ROOT.glob("*.py")) == []

    utility_paths = sorted((NOTEBOOKS_ROOT / "utils").glob("*_utils.py"))
    assert utility_paths

    for utility_path in utility_paths:
        importlib.import_module(f"notebooks.utils.{utility_path.stem}")
