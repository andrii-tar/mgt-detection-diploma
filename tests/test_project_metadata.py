from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)


def test_core_dependencies_exclude_notebook_only_packages():
    dependencies = set(_project_metadata()["project"]["dependencies"])

    assert {
        "accelerate>=1.14.0",
        "pandas>=3.0.3",
        "pyarrow>=18.0.0",
        "python-dotenv>=1.0.1",
        "torch>=2.6.0",
        "transformers>=5.13.1",
        "tqdm>=4.66.5",
    } <= dependencies
    assert not any(name.startswith("jupyter") for name in dependencies)
    assert not any(name.startswith("matplotlib") for name in dependencies)
    assert not any(name.startswith("seaborn") for name in dependencies)


def test_notebook_extra_contains_notebook_workflow_dependencies():
    extras = _project_metadata()["project"]["optional-dependencies"]
    notebook = set(extras["notebook"])

    assert {
        "jupyter>=1.1.1",
        "jupyter-collaboration>=3.1.2",
        "matplotlib>=3.10.0,<3.11",
        "numpy",
        "seaborn>=0.13.2",
    } <= notebook


def test_dev_group_contains_pytest():
    assert "pytest>=8.0.0" in _project_metadata()["dependency-groups"]["dev"]


def test_no_project_console_scripts_are_declared():
    assert "scripts" not in _project_metadata()["project"]
