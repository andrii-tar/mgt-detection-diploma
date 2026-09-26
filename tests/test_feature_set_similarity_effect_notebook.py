from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "feature_set_similarity_effect_analysis.ipynb"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_feature_set_similarity_effect_notebook_is_output_free() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_feature_set_similarity_effect_notebook_declares_both_hypothesis_outputs() -> None:
    source = _source()

    for required in (
        "CLASSIFIER_ANALYSIS_SOURCE_RUN_ID",
        "CLASSIFIER_ANALYSIS_RUN_IDS",
        "BASELINE_FEATURE_SET",
        "COMPARISON_FEATURE_SETS",
        "matched_feature_set_deltas(",
        "plot_feature_set_comparison(",
        "plot_feature_set_delta_heatmap(",
        "plot_delta_by_model_size(",
        "matched_feature_set_deltas.csv",
        "feature_set_effect_summary.csv",
        "feature_set_comparison.png",
        "delta_by_model_size.png",
    ):
        assert required in source


def test_notebook_iterates_training_regimes_and_matches_gpt_who_contexts() -> None:
    source = _source()

    assert "training_regimes = sorted(" in source
    assert "for training_regime in training_regimes" in source
    assert "training_regime_output_root" in source
    assert "gpt_who_experiment_for_training_regime" in source
