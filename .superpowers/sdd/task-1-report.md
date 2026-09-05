# Task 1 Report

## Status

DONE_WITH_CONCERNS

## Files Changed

- `tests/test_postprocess_pipeline.py`
  - Imported `run_postprocessing`.
  - Added `test_run_postprocessing_reports_progress_per_dataset_model_pair`.
  - The test creates HG and MG split directories, records the progress wrapper configuration and updates, patches `_process_split`, and asserts the required summary.
- `.superpowers/sdd/task-1-report.md`
  - Added this report.

No production files were modified. No generated data was created or committed. No commit was made.

## Verification

Exact command:

```text
uv run pytest tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair -q
```

Exact output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
_______ test_run_postprocessing_reports_progress_per_dataset_model_pair _______

obj = <module 'scripts.postprocess_pipeline' from 'D:\\Projects\\master_diploma\\text-detection\\scripts\\postprocess_pipeline.py'>
name = 'tqdm', ann = 'scripts.postprocess_pipeline'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
>           obj = getattr(obj, name)
                  ^^^^^^^^^^^^^^^^^^
E           AttributeError: module 'scripts.postprocess_pipeline' has no attribute 'tqdm'

.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:95: AttributeError
tmp_path = WindowsPath('C:/Users/Andrii/AppData/Local/Temp/pytest-of-Andrii/pytest-158/test_run_postprocessing_report0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x0000014C760D4740>
capsys = <_pytest.capture.CaptureFixture object at 0x0000014C760D7320>

    def test_run_postprocessing_reports_progress_per_dataset_model_pair(tmp_path, monkeypatch, capsys):
        experiment = tmp_path / "demo__model_a"
        for split in ("HG", "MG"):
            (experiment / split / "data").mkdir(parents=True)

        progress = {}

        class RecordingBar:
            def __init__(self, iterable, **kwargs):
                progress.update(kwargs)
                self.iterable = iterable
                self.updates = 0
                progress["bar"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def update(self, amount):
            self.updates += amount

>       monkeypatch.setattr("scripts.postprocess_pipeline.tqdm", RecordingBar)

tests\\test_postprocess_pipeline.py:85: 
.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:109: in derive_importpath
    annotated_getattr(target, attr, ann=module)
.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:97: in annotated_getattr
    raise AttributeError from e
E   AttributeError: 'module' object at scripts.postprocess_pipeline has no attribute 'tqdm'

1 failed in 0.77s
```

The failure is expected and is caused by the missing production progress integration, not by a test typo: `scripts/postprocess_pipeline.py` defines `run_postprocessing` but has no module-level `tqdm` symbol.

## Self-Review

- The test is limited to the requested progress-bar contract.
- It asserts `desc="demo/model_a"`, `total=2`, `unit="split"`, two updates, return code `0`, and the expected summary.
- The test uses a recording iterator and does not depend on terminal rendering.
- Existing tests and production code were not changed.

## Concerns

- The focused test is intentionally failing until the next task adds the production `tqdm` integration.
- The full test suite was not run because this task requires confirming the focused red test only.

## Review Fix

- Changed `RecordingBar` to collect individual `update(amount)` arguments in a list.
- The progress assertion now requires exactly two calls, each with amount `1`: `progress["bar"].updates == [1, 1]`.
- No production code or generated data was modified.

## Verification After Review Fix

Exact command:

```text
uv run pytest tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair -q
```

Exit status: `1`

Exact output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
_______ test_run_postprocessing_reports_progress_per_dataset_model_pair _______

obj = <module 'scripts.postprocess_pipeline' from 'D:\\Projects\\master_diploma\\text-detection\\scripts\\postprocess_pipeline.py'>
name = 'tqdm', ann = 'scripts.postprocess_pipeline'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
>           obj = getattr(obj, name)
              E           AttributeError: module 'scripts.postprocess_pipeline' has no attribute 'tqdm'

.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:95: AttributeError

The above exception was the direct cause of the following exception:

tmp_path = WindowsPath('C:/Users/Andrii/AppData/Local/Temp/pytest-of-Andrii/pytest-159/test_run_postprocessing_report0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x0000020DD44539E0>
capsys = <_pytest.capture.CaptureFixture object at 0x0000020DD4453AD0>

    def test_run_postprocessing_reports_progress_per_dataset_model_pair(tmp_path, monkeypatch, capsys):
        experiment = tmp_path / "demo__model_a"
        for split in ("HG", "MG"):
            (experiment / split / "data").mkdir(parents=True)
    
        progress = {}
    
        class RecordingBar:
            def __init__(self, iterable, **kwargs):
                progress.update(kwargs)
                self.iterable = iterable
                self.updates = []
                progress["bar"] = self
    
            def __enter__(self):
                return self
    
            def __exit__(self, exc_type, exc, traceback):
                return False
    
            def __iter__(self):
                return iter(self.iterable)
    
            def update(self, amount):
                self.updates.append(amount)
    
>       monkeypatch.setattr("scripts.postprocess_pipeline.tqdm", RecordingBar)

tests\\test_postprocess_pipeline.py:85: 
.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:109: in derive_importpath
    annotated_getattr(target, attr, ann=module)
.venv\\Lib\\site-packages\\_pytest\\monkeypatch.py:97: in annotated_getattr
    raise AttributeError(
E   AttributeError: 'module' object at scripts.postprocess_pipeline has no attribute 'tqdm'
=========================== short test summary info ===========================
FAILED tests/test_postprocess_pipeline.py::test_run_postprocessing_reports_progress_per_dataset_model_pair
1 failed in 0.64s
```
