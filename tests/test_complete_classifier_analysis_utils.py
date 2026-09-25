from pathlib import Path

import pandas as pd

from notebooks.utils.complete_classifier_analysis_utils import (
    AnalysisPaths,
    configuration_fingerprint,
    deduplicate_parent_counts,
    evaluation_metric_table,
    run_step,
    select_smoke_artifacts,
    step_is_complete,
    unavailable_table,
)


def test_analysis_paths_create_separate_intermediate_and_output_roots(tmp_path: Path):
    paths = AnalysisPaths.from_root(tmp_path, "smoke-run")

    assert paths.intermediate == tmp_path / "smoke-run" / "intermediate"
    assert paths.output == tmp_path / "smoke-run" / "output"
    assert paths.manifest == tmp_path / "smoke-run" / "manifest.json"


def test_configuration_fingerprint_is_order_independent():
    left = configuration_fingerprint({"run_id": "r1", "window": 32})
    right = configuration_fingerprint({"window": 32, "run_id": "r1"})

    assert left == right


def test_step_is_complete_requires_matching_fingerprint_and_outputs(tmp_path: Path):
    output = tmp_path / "table.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    manifest = {"steps": {"metrics": {"fingerprint": "abc", "outputs": [str(output)]}}}

    assert step_is_complete(
        manifest, step_id="metrics", fingerprint="abc", expected_outputs=[output]
    )
    assert not step_is_complete(
        manifest, step_id="metrics", fingerprint="stale", expected_outputs=[output]
    )

    output.unlink()
    assert not step_is_complete(
        manifest, step_id="metrics", fingerprint="abc", expected_outputs=[output]
    )


def test_run_step_skips_completed_callback_and_reruns_stale_step(tmp_path: Path):
    paths = AnalysisPaths.from_root(tmp_path, "run")
    output = paths.output / "value.txt"
    calls: list[str] = []

    def write_output() -> None:
        calls.append("called")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("ok", encoding="utf-8")

    assert run_step(
        paths,
        step_id="value",
        fingerprint="one",
        expected_outputs=[output],
        callback=write_output,
    )
    assert not run_step(
        paths,
        step_id="value",
        fingerprint="one",
        expected_outputs=[output],
        callback=write_output,
    )
    assert run_step(
        paths,
        step_id="value",
        fingerprint="two",
        expected_outputs=[output],
        callback=write_output,
    )
    assert calls == ["called", "called"]


def test_select_smoke_artifacts_returns_one_artifact_per_configuration():
    artifacts = pd.DataFrame(
        [
            {
                "run_id": "r",
                "scorer_llm": "s",
                "model_family": "logreg",
                "training_window": 32,
                "path": "b",
            },
            {
                "run_id": "r",
                "scorer_llm": "s",
                "model_family": "logreg",
                "training_window": 32,
                "path": "a",
            },
        ]
    )

    result = select_smoke_artifacts(artifacts, limit=1)

    assert result["path"].tolist() == ["a"]


def test_deduplicate_parent_counts_does_not_count_windows_as_parents():
    frame = pd.DataFrame(
        {"text_id": [1, 1, 2], "label": [0, 0, 1], "window_id": [0, 1, 0]}
    )

    result = deduplicate_parent_counts(
        frame, parent_columns=["text_id"], label_column="label"
    )

    assert result.set_index("label")["parent_count"].to_dict() == {0: 1, 1: 1}


def test_evaluation_metric_table_marks_aggregate_metrics_as_summary_data():
    metrics = pd.DataFrame({"dataset": ["kaggle_train"], "macro_f1": [0.75]})

    result = evaluation_metric_table(metrics)

    assert result["metric_source"].tolist() == ["evaluation_metrics"]
    assert result["macro_f1"].tolist() == [0.75]


def test_unavailable_table_explains_missing_prediction_dependent_output():
    result = unavailable_table("detailed predictions are not available")

    assert result.to_dict("records") == [{
        "status": "unavailable",
        "reason": "detailed predictions are not available",
    }]
