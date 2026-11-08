import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "report" / "08_mixed_similarity_ablation.ipynb"


def test_mixed_similarity_ablation_notebook_has_isolated_run_and_sampling_config():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert code_cells
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "mixed-similarity-ablation-2026-10-29-v6" in source
    assert "TEXTS_PER_DATASET_SPLIT_TRAIN = 50" in source
    assert "TEXTS_PER_DATASET_SPLIT_EVALUATION = 50" in source
    assert "paired_macro_f1_bootstrap" in source
    assert "intersect_eligible_texts_by_stratum" in source
    assert "name_that_llm" not in source
    assert "bbc_news" not in source


def test_expanded_scorer_run_preserves_original_cohort_and_adds_two_scorers():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "mixed-similarity-ablation-2026-10-29-v6" in source
    assert "COHORT_SOURCE_RUN_ID = 'mixed-similarity-ablation-2026-10-29-v2'" in source
    assert "'gemma_3_1b_it'" in source
    assert "'smollm2_360m'" in source
    assert "selected = reference_texts.assign(scorer_llm=scorer_llm)" in source
    assert "for scorer_llm in SCORER_LLMS[2:]" in source
