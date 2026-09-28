import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "try_gpt_who_baseline.ipynb"


def _code_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
        sources.append(source)
    return "\n".join(sources)


def test_gpt_who_notebook_is_output_free_and_declares_pipeline_contract():
    source = _code_source()
    assert 'row.get("dataset", DATASET)' not in source

    for required in (
        "from notebooks.utils.gpt_who_experiment_config import",
        "EXPERIMENTS =",
        "validate_experiments(",
        "experiment_dataset_names(",
        "experiment_run_id(",
        "scorer_llm",
        "training_datasets",
        "evaluation_datasets",
        "MIXED_MAX_PER_SPLIT = 500",
        "TRAIN_COUNT_PER_CLASS = 900",
        "EVAL_COUNT_PER_CLASS = 300",
        "iter_logprob_row_batches(",
        "from tqdm.auto import tqdm",
        'desc="Loading Stage 1 rows"',
        'desc="Extracting GPT-who features"',
        "build_gpt_who_feature_frame(",
        "make_experiment_cohorts(",
        "train_keys = set(zip(cohorts[\"train\"][\"dataset\"]",
        "evaluation_keys = set(zip(cohorts[\"evaluation\"][\"dataset\"]",
        "fit_and_score_logistic(",
        "summarize_scores(",
        "metrics.parquet",
        "scores.parquet",
        "coefficients.parquet",
        "exclusions.parquet",
        "coverage.parquet",
        "configuration.json",
        "confusion_matrix.png",
        "score_distributions.png",
    ):
        assert required in source


def test_gpt_who_notebook_iterates_configured_experiments():
    source = _code_source()
    assert "for experiment in EXPERIMENTS:" in source
    assert "for scorer_llm in experiment[\"scorer_llms\"]:" in source
    assert "training_dataset_names = experiment_dataset_names(experiment, \"training_datasets\")" in source
    assert "evaluation_dataset_names = experiment_dataset_names(experiment, \"evaluation_datasets\")" in source
