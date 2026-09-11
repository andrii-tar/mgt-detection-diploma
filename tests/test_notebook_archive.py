from __future__ import annotations

from pathlib import Path


NOTEBOOKS_DIRECTORY = Path(__file__).parents[1] / "notebooks"
ARCHIVE_DIRECTORY = NOTEBOOKS_DIRECTORY / "archive"
ARCHIVED_NOTEBOOKS = {
    "text_classification.ipynb",
    "cross_dataset_classification_poc.ipynb",
    "local_evidence_classification.ipynb",
    "local_evidence_segmentation.ipynb",
    "gemma_cross_dataset_boundary_validation.ipynb",
    "bayesian_model_comparison.ipynb",
}


def test_archive_contains_exactly_the_superseded_training_notebooks() -> None:
    assert {path.name for path in ARCHIVE_DIRECTORY.glob("*.ipynb")} == ARCHIVED_NOTEBOOKS
    assert not ARCHIVED_NOTEBOOKS & {path.name for path in NOTEBOOKS_DIRECTORY.glob("*.ipynb")}


def test_archive_readme_documents_archived_notebooks() -> None:
    readme = ARCHIVE_DIRECTORY / "README.md"
    assert readme.is_file()
    contents = readme.read_text(encoding="utf-8")

    assert "superseded" in contents.lower()
    for notebook in sorted(ARCHIVED_NOTEBOOKS):
        assert notebook in contents


def test_archive_readme_has_one_purpose_status_replacement_row_per_notebook() -> None:
    contents = (ARCHIVE_DIRECTORY / "README.md").read_text(encoding="utf-8")
    rows = [line for line in contents.splitlines() if line.startswith("|") and not line.startswith("|---")]

    assert len(rows) == len(ARCHIVED_NOTEBOOKS) + 1
    assert rows[0].lower().count("notebook") == 1
    assert rows[0].lower().count("purpose") == 1
    assert rows[0].lower().count("inactive") == 1
    assert rows[0].lower().count("replacement") == 1
    for notebook in ARCHIVED_NOTEBOOKS:
        row = next(row for row in rows[1:] if notebook in row)
        assert row.count("|") == 5
