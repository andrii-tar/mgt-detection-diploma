from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from notebooks.utils.local_evidence_models_utils import load_fitted_models, save_fitted_models
from notebooks.utils.local_evidence_training_utils import LocalEvidenceModelBundle, LocalEvidenceTrainingResult


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_DIRECTORY = ("experiment-results", "local-evidence-classification")
_RESULT_NAMES = ("metrics", "predictions", "coverage")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class LocalEvidenceArtifact:
    path: Path
    metadata: dict[str, object]
    model: LocalEvidenceModelBundle
    training_metrics: pd.DataFrame
    training_coverage: pd.DataFrame
    evaluation_results: dict[str, pd.DataFrame]


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, periods, underscores, or hyphens")
    return run_id


def local_evidence_artifact_path(data_root: Path | str, run_id: str) -> Path:
    return Path(data_root).joinpath(*ARTIFACT_DIRECTORY, _validate_run_id(run_id))


def _bundle_configuration(model: LocalEvidenceModelBundle) -> dict[str, object]:
    return {
        "llm_name": model.llm_name,
        "classifier_names": sorted({name for name, _ in model.classifiers}),
        "feature_columns": list(model.feature_columns),
        "window_lengths": list(model.window_lengths),
        "stride": model.stride,
        "top_k": model.top_k,
        "training_datasets": list(model.training_datasets),
    }


def _cache_manifest(bundle: Mapping[str, object], training_input_fingerprint: str) -> dict[str, object]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle": dict(bundle),
        "features": bundle["feature_columns"],
        "model_names": bundle["classifier_names"],
        "training_input_fingerprint": training_input_fingerprint,
    }


def _validate_evaluation_results(evaluation_results: Mapping[str, pd.DataFrame]) -> None:
    missing = set(_RESULT_NAMES).difference(evaluation_results)
    if missing:
        raise ValueError(f"evaluation_results is missing frames: {sorted(missing)}")
    if any(not isinstance(evaluation_results[name], pd.DataFrame) for name in _RESULT_NAMES):
        raise ValueError("evaluation result frames must be pandas DataFrames")


def save_local_evidence_artifact(
    data_root: Path | str,
    run_id: str,
    result: LocalEvidenceTrainingResult,
    *,
    metadata: Mapping[str, object],
    evaluation_results: Mapping[str, pd.DataFrame],
) -> Path:
    """Persist a fitted local-evidence bundle and its classification results."""
    _validate_evaluation_results(evaluation_results)
    path = local_evidence_artifact_path(data_root, run_id)
    if path.exists():
        raise FileExistsError(f"local-evidence artifact already exists: {path}")

    bundle = _bundle_configuration(result.model)
    cache_manifest = _cache_manifest(bundle, result.training_input_fingerprint)
    artifact_metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "metadata": dict(metadata),
        "bundle": bundle,
        "training_input_fingerprint": result.training_input_fingerprint,
        "model_cache_manifest": cache_manifest,
    }
    path.mkdir(parents=True)
    save_fitted_models(path / "model-cache", result.model.classifiers, cache_manifest)
    result.model.decision_rules.to_parquet(path / "decision_rules.parquet", index=False)
    result.metrics.to_parquet(path / "training_metrics.parquet", index=False)
    result.coverage.to_parquet(path / "training_coverage.parquet", index=False)
    for name in _RESULT_NAMES:
        evaluation_results[name].to_parquet(path / f"evaluation_{name}.parquet", index=False)
    (path / "metadata.json").write_text(
        json.dumps(artifact_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _load_metadata(path: Path) -> dict[str, object]:
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local-evidence artifact metadata: {path}") from error
    if not isinstance(metadata, dict) or metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported local-evidence artifact schema")
    return metadata


def _model_from_metadata(
    metadata: Mapping[str, object],
    classifiers: dict[tuple[str, int], object],
    decision_rules: pd.DataFrame,
) -> LocalEvidenceModelBundle:
    bundle = metadata.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError("local-evidence artifact is missing bundle metadata")
    try:
        return LocalEvidenceModelBundle(
            llm_name=str(bundle["llm_name"]),
            classifiers=classifiers,
            decision_rules=decision_rules,
            feature_columns=tuple(bundle["feature_columns"]),
            window_lengths=tuple(int(length) for length in bundle["window_lengths"]),
            stride=int(bundle["stride"]),
            top_k=int(bundle["top_k"]),
            training_datasets=tuple(bundle["training_datasets"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid local-evidence artifact bundle metadata") from error


def load_local_evidence_artifact(
    data_root: Path | str,
    run_id: str,
    *,
    expected_metadata: Mapping[str, object],
    trust_artifact: bool = False,
) -> LocalEvidenceArtifact:
    """Load a compatible artifact only after explicit trusted-pickle consent."""
    path = local_evidence_artifact_path(data_root, run_id)
    metadata = _load_metadata(path)
    if metadata.get("run_id") != run_id:
        raise ValueError("local-evidence artifact run ID does not match its metadata")
    if metadata.get("metadata") != dict(expected_metadata):
        raise ValueError("local-evidence artifact metadata mismatch")
    if not trust_artifact:
        raise PermissionError("loading a local-evidence artifact requires trust_artifact=True")

    cache_manifest = metadata.get("model_cache_manifest")
    if not isinstance(cache_manifest, dict):
        raise ValueError("local-evidence artifact is missing model-cache metadata")
    if metadata.get("training_input_fingerprint") != cache_manifest.get("training_input_fingerprint"):
        raise ValueError("local-evidence artifact training fingerprint mismatch")
    classifiers = load_fitted_models(path / "model-cache", cache_manifest, trust_cache=True)
    if not classifiers:
        raise ValueError("local-evidence artifact model cache is missing, corrupt, or incompatible")
    try:
        decision_rules = pd.read_parquet(path / "decision_rules.parquet")
        training_metrics = pd.read_parquet(path / "training_metrics.parquet")
        training_coverage = pd.read_parquet(path / "training_coverage.parquet")
        evaluation_results = {
            name: pd.read_parquet(path / f"evaluation_{name}.parquet")
            for name in _RESULT_NAMES
        }
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("local-evidence artifact result files are missing or invalid") from error
    return LocalEvidenceArtifact(
        path=path,
        metadata=metadata,
        model=_model_from_metadata(metadata, classifiers, decision_rules),
        training_metrics=training_metrics,
        training_coverage=training_coverage,
        evaluation_results=evaluation_results,
    )
