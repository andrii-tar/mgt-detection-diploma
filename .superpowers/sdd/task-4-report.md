# Task 4 Verification Report

## Status

Verified. The completed Task 3 overlay implementation passes the focused tests when run headlessly, the full suite, notebook output-free validation, and whitespace checks.

## Verification

Focused tests, exact Task 4 command:

```text
uv run pytest tests/test_notebook_utils.py tests/test_classification_utils.py -q
1 failed, 111 passed in 41.67s
```

The single failure was `test_plot_length_metric_distributions_uses_token_length_for_y_axis`; Matplotlib selected Tk and failed because the local Python installation lacks `C:/Python312/tcl/tcl8.6/auto.tcl`.

Focused tests, headless rerun:

```text
$env:MPLBACKEND = 'Agg'; uv run pytest tests/test_notebook_utils.py tests/test_classification_utils.py -q
112 passed in 21.73s
```

Full suite, required headless command:

```text
$env:MPLBACKEND = 'Agg'; uv run pytest -q
181 passed in 128.71s (0:02:08)
```

Notebook output-free JSON validation:

```text
uv run python -c "import json; p=json.load(open('notebooks/local_evidence_classification.ipynb')); assert all(c.get('execution_count') is None and c.get('outputs', []) == [] for c in p['cells'] if c['cell_type'] == 'code')"
```

Passed with no output.

Diff integrity:

```text
git diff --check
git diff --check c73301c e5faada
```

Both passed with no output.

## Spec Review

Manual review of `c73301c..e5faada` found no blocking or important issues. The implementation preserves mean-overlap aggregation, independent normalization, and same-axis plotting; integrates the overlay into all four custom cells; uses custom context and `top_k`; scores before plotting; avoids held-out `scored_windows`; and preserves output-free notebook structure.

The delegated reviewer request could not be started because the available subagent depth was exhausted.

## Final-Review Fix Verification

The notebook import contract was strengthened to require `plot_normalized_token_metrics_with_local_evidence` from `token_candidate_similarity_heatmaps_utils` and forbid it from the `local_evidence_utils` import.

Pytest command:

```powershell
$env:MPLBACKEND='Agg'; uv run pytest tests/test_classification_utils.py tests/test_notebook_utils.py -q
```

Exact output:

```text
........................................................................ [ 64%]
........................................                                 [100%]
112 passed in 14.89s
```

Fresh-kernel notebook command with `TEXT_DETECTION_DATA` set:

```powershell
$env:TEXT_DETECTION_DATA='D:/Projects/master_diploma/text-detection-data'; uv run jupyter nbconvert --to notebook --execute notebooks/local_evidence_classification.ipynb --output 'C:\Users\Andrii\AppData\Local\Temp\opencode\local_evidence_classification.final.ipynb' --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3
```

Exact output:

```text
[NbConvertApp] Converting notebook notebooks/local_evidence_classification.ipynb to notebook
D:\Projects\master_diploma\text-detection\.venv\Lib\site-packages\zmq\_future.py:718: RuntimeWarning: Proactor event loop does not implement add_reader family of methods required for zmq. Registering an additional selector thread for add_reader support via tornado. Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid this warning.
  self._get_loop()
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 5559046 bytes to C:\Users\Andrii\AppData\Local\Temp\opencode\local_evidence_classification.final.ipynb
```

The source notebook remains output-free; execution was written to the temporary output path.

## Concerns

- The exact focused command is headless and passes; notebook execution emits existing Windows/ZeroMQ warnings but exits successfully.
- The fresh execution output is intentionally outside the repository at the requested temporary path.
