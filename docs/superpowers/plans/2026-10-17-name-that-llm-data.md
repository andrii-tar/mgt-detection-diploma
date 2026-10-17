# Name That LLM Dataset Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible MG-only Name That LLM dataset importer with per-text `source_llm` metadata and a registered pipeline alias.

**Architecture:** A dedicated pandas-based preparation script reads `train.csv`, validates its required fields, and writes the normalized MG CSV under the external data root. The experiment configuration exposes that generated CSV as the `name_that_llm_train` dataset with only an `MG` split.

**Tech Stack:** Python 3.12+, pandas, argparse, pathlib, pytest, uv.

## Global Constraints

- Input default: `$TEXT_DETECTION_DATA/raw/name-that-llm/train.csv`.
- Output default: `$TEXT_DETECTION_DATA/raw/name-that-llm/name_that_llm_train_MG.csv`.
- Output columns: `id`, `source_id`, `text`, `prompt_id`, `source_llm`, `origin`, `source_dataset`.
- Generate zero-based integer pipeline IDs and preserve original string IDs, including leading zeroes, in `source_id`.
- Copy `model` to `source_llm`; set `origin` to `MG`; set `source_dataset` to `name_that_llm`.
- Do not commit raw data or generated CSV output.

---

### Task 1: Add importer regression tests

**Files:**
- Create: `tests/test_prepare_name_that_llm.py`

**Interfaces:**
- Consumes: the planned `prepare_name_that_llm(input_path: Path, output_path: Path) -> None` function.
- Produces: executable expectations for output shape, metadata, ID preservation, and validation failures.

- [ ] **Step 1: Write the successful conversion test**

Create a temporary input CSV with two rows such as:

```python
pd.DataFrame(
    [
        {"id": "0018b3765972", "prompt_id": "p081", "text": "first text", "model": "qwen3.8-max"},
        {"id": "0041ac86ee93", "prompt_id": "p073", "text": "second text", "model": "claude-sonnet-5"},
    ]
).to_csv(input_path, index=False)
```

Call `prepare_name_that_llm(input_path, output_path)`, read the output with
`pd.read_csv(output_path, dtype={"id": str})`, and assert:

```python
assert list(result.columns) == [
    "id", "source_id", "text", "prompt_id", "source_llm", "origin", "source_dataset"
]
assert result["id"].tolist() == ["0", "1"]
assert result["source_id"].tolist() == ["0018b3765972", "0041ac86ee93"]
assert result["prompt_id"].tolist() == ["p081", "p073"]
assert result["source_llm"].tolist() == ["qwen3.8-max", "claude-sonnet-5"]
assert set(result["origin"]) == {"MG"}
assert set(result["source_dataset"]) == {"name_that_llm"}
```

- [ ] **Step 2: Add validation tests**

Parameterize malformed inputs and assert `ValueError`:

```python
[
    ({"id": ["a"], "prompt_id": ["p"], "text": ["x"]}, "model"),
    ({"id": ["a", "a"], "prompt_id": ["p", "q"], "text": ["x", "y"], "model": ["m", "n"]}, "duplicate"),
    ({"id": ["a"], "prompt_id": ["p"], "text": ["  "], "model": ["m"]}, "text"),
    ({"id": ["a"], "prompt_id": ["p"], "text": ["x"], "model": ["  "]}, "model"),
]
```

Also assert a missing input path raises `FileNotFoundError`.

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```powershell
uv run pytest -q tests/test_prepare_name_that_llm.py
```

Expected: FAIL because `scripts.extract_data.prepare_name_that_llm` does not yet exist.

### Task 2: Implement the importer and dataset registration

**Files:**
- Create: `scripts/extract_data/prepare_name_that_llm.py`
- Modify: `scripts/experiment_config.py` after the existing dataset entries

**Interfaces:**
- Consumes: `Path` input and output arguments; input columns `id`, `prompt_id`, `text`, `model`.
- Produces: `prepare_name_that_llm(input_path: Path, output_path: Path) -> None`; CLI options `--input-path` and `--output-path`; `name_that_llm_train` configuration alias.

- [ ] **Step 1: Implement input validation and transformation**

Implement the script with the existing project pattern:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_COLUMNS = ["id", "prompt_id", "text", "model"]
OUTPUT_COLUMNS = ["id", "source_id", "text", "prompt_id", "source_llm", "origin", "source_dataset"]


def prepare_name_that_llm(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    frame = pd.read_csv(input_path, dtype={"id": str, "prompt_id": str, "model": str})
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError("source id values must be unique")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("text contains blank values")
    if frame["model"].isna().any() or frame["model"].astype(str).str.strip().eq("").any():
        raise ValueError("model contains blank values")

    result = pd.DataFrame(
        {
            "id": range(len(frame)),
            "source_id": frame["id"],
            "text": frame["text"],
            "prompt_id": frame["prompt_id"],
            "source_llm": frame["model"],
            "origin": "MG",
            "source_dataset": "name_that_llm",
        },
        columns=OUTPUT_COLUMNS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
```

Add `_parse_args()` and `main()` so omitted arguments resolve to:

```python
data_root / "raw" / "name-that-llm" / "train.csv"
data_root / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
```

- [ ] **Step 2: Register the MG-only dataset**

Add this entry inside `build_datasets(data_root)`:

```python
"name_that_llm_train": {
    "dataset_name": "name_that_llm_train",
    "splits": {
        "MG": data_root / "raw/name-that-llm/name_that_llm_train_MG.csv",
    },
},
```

- [ ] **Step 3: Run focused tests and verify they pass**

Run:

```powershell
uv run pytest -q tests/test_prepare_name_that_llm.py
```

Expected: all importer tests pass.

### Task 3: Verify configuration and repository hygiene

**Files:**
- Modify: none

**Interfaces:**
- Consumes: the importer, dataset configuration, and focused tests from Tasks 1-2.
- Produces: verified implementation ready for review; no raw or generated data added to Git.

- [ ] **Step 1: Verify the registered split**

Run:

```powershell
uv run python -c "from pathlib import Path; from scripts.experiment_config import resolve_dataset_splits; print(resolve_dataset_splits('name_that_llm_train', Path('D:/Projects/master_diploma/text-detection-data')))"
```

Expected: a dictionary containing exactly one `MG` entry pointing to
`D:/Projects/master_diploma/text-detection-data/raw/name-that-llm/name_that_llm_train_MG.csv`.

- [ ] **Step 2: Run the relevant test suite and formatting checks**

Run:

```powershell
uv run pytest -q tests/test_prepare_name_that_llm.py tests/test_prepare_semeval2024_task8_subtask_c.py
git diff --check
```

Expected: tests pass and `git diff --check` produces no output.

- [ ] **Step 3: Review and commit only source and tests**

Run:

```powershell
git add scripts/extract_data/prepare_name_that_llm.py scripts/experiment_config.py tests/test_prepare_name_that_llm.py
```

Confirm the raw `train.csv` and generated `name_that_llm_train_MG.csv` are not staged.
