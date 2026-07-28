from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["TEXT_DETECTION_DATA"] = str(REPOSITORY_ROOT)
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_process_pipeline_supports_package_execution():
    result = _run_help("scripts.process_pipeline")

    assert result.returncode == 0, result.stderr
    assert "Process next HG and MG texts" in result.stdout


def test_token_similarity_stage2_supports_package_execution():
    result = _run_help("scripts.token_similarity_stage2")

    assert result.returncode == 0, result.stderr
    assert "Stage 2: compute token embedding similarities" in result.stdout
