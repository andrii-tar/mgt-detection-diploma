import pandas as pd

from notebooks.utils.gpt_who_results_analysis_utils import (
    load_gpt_who_metrics,
    summarize_gpt_who_metrics,
)


def test_load_and_summarize_gpt_who_metrics(tmp_path):
    for scorer_llm, macro_f1 in (("model_a", 0.7), ("model_b", 0.8)):
        output_root = tmp_path / f"run__{scorer_llm}"
        output_root.mkdir()
        pd.DataFrame([{
            "experiment_name": "mixed",
            "run_id": "run__mixed",
            "scorer_llm": scorer_llm,
            "partition": "evaluation",
            "sample_count": 10,
            "macro_f1": macro_f1,
        }]).to_parquet(output_root / "metrics.parquet", index=False)

    metrics = load_gpt_who_metrics(tmp_path)
    summary = summarize_gpt_who_metrics(metrics)

    assert len(metrics) == 2
    assert summary.set_index("scorer_llm")["mean_macro_f1"].to_dict() == {
        "model_a": 0.7,
        "model_b": 0.8,
    }
