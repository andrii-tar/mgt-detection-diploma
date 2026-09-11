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
