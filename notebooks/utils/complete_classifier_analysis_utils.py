"""Reusable helpers for the resumable classifier analysis notebook."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    intermediate: Path
    output: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: Path | str, run_id: str) -> "AnalysisPaths":
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty directory name")
        run_root = Path(root) / run_id
        return cls(run_root, run_root / "intermediate", run_root / "output", run_root / "manifest.json")


def configuration_fingerprint(configuration: Mapping[str, object]) -> str:
    payload = json.dumps(configuration, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path | str) -> dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"version": 1, "steps": {}}
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid analysis manifest: {manifest_path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("steps", {}), dict):
        raise ValueError(f"analysis manifest must contain a steps object: {manifest_path}")
    return value


def step_is_complete(
    manifest: Mapping[str, object],
    *,
    step_id: str,
    fingerprint: str,
    expected_outputs: Sequence[Path | str],
) -> bool:
    steps = manifest.get("steps", {})
    if not isinstance(steps, Mapping):
        return False
    record = steps.get(step_id)
    if not isinstance(record, Mapping) or record.get("fingerprint") != fingerprint:
        return False
    recorded_outputs = {str(path) for path in record.get("outputs", [])}
    expected = {str(Path(path)) for path in expected_outputs}
    return recorded_outputs == expected and all(Path(path).is_file() for path in expected)


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def mark_step_complete(
    path: Path | str,
    *,
    step_id: str,
    fingerprint: str,
    expected_outputs: Sequence[Path | str],
    row_count: int | None = None,
) -> None:
    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    record: dict[str, object] = {
        "fingerprint": fingerprint,
        "outputs": [str(Path(output)) for output in expected_outputs],
        "status": "complete",
    }
    if row_count is not None:
        record["row_count"] = row_count
    steps = dict(manifest.get("steps", {}))
    steps[step_id] = record
    updated = dict(manifest)
    updated["version"] = 1
    updated["steps"] = steps
    _write_manifest(manifest_path, updated)


def run_step(
    paths: AnalysisPaths,
    *,
    step_id: str,
    fingerprint: str,
    expected_outputs: Sequence[Path | str],
    callback: Callable[[], None],
    row_count: int | None = None,
) -> bool:
    manifest = load_manifest(paths.manifest)
    if step_is_complete(
        manifest,
        step_id=step_id,
        fingerprint=fingerprint,
        expected_outputs=expected_outputs,
    ):
        return False
    paths.intermediate.mkdir(parents=True, exist_ok=True)
    paths.output.mkdir(parents=True, exist_ok=True)
    callback()
    missing = [str(path) for path in expected_outputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"analysis step {step_id} did not create outputs: {missing}")
    mark_step_complete(
        paths.manifest,
        step_id=step_id,
        fingerprint=fingerprint,
        expected_outputs=expected_outputs,
        row_count=row_count,
    )
    return True


def select_smoke_artifacts(artifacts: pd.DataFrame, *, limit: int = 1) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be positive")
    if artifacts.empty:
        return artifacts.copy()
    configuration_columns = [
        column
        for column in ("run_id", "scorer_llm", "model_family", "feature_set", "training_regime", "training_window")
        if column in artifacts.columns
    ]
    sort_columns = configuration_columns + [column for column in ("path", "artifact_path") if column in artifacts.columns]
    result = artifacts.sort_values(sort_columns, kind="stable") if sort_columns else artifacts.copy()
    if configuration_columns:
        result = result.groupby(configuration_columns, sort=False, dropna=False).head(limit)
    else:
        result = result.head(limit)
    return result.reset_index(drop=True)


def limit_smoke_rows(
    frame: pd.DataFrame, *, rows_per_group: int = 1, group_columns: Sequence[str]
) -> pd.DataFrame:
    if rows_per_group < 1:
        raise ValueError("rows_per_group must be positive")
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing smoke grouping columns: {missing}")
    return frame.groupby(list(group_columns), sort=True, dropna=False).head(rows_per_group).reset_index(drop=True)


def evaluation_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate evaluation metrics with an explicit provenance marker."""
    result = metrics.copy()
    result.insert(0, "metric_source", "evaluation_metrics")
    return result


def unavailable_table(reason: str) -> pd.DataFrame:
    """Represent an intentionally unavailable analysis output without an empty file."""
    if not reason.strip():
        raise ValueError("reason must not be empty")
    return pd.DataFrame([{"status": "unavailable", "reason": reason}])


def deduplicate_parent_counts(
    frame: pd.DataFrame, *, parent_columns: Sequence[str], label_column: str
) -> pd.DataFrame:
    required = [*parent_columns, label_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing parent-count columns: {missing}")
    parents = frame.loc[:, required].drop_duplicates()
    return (
        parents.groupby(label_column, dropna=False)
        .size()
        .rename("parent_count")
        .reset_index()
    )


def _safe_metric(function: Callable[..., float], *args: object, **kwargs: object) -> float | None:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return None


def binary_metric_table(
    predictions: pd.DataFrame,
    *,
    label_column: str,
    score_column: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    required = {label_column, score_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    clean = predictions.loc[:, [label_column, score_column]].dropna()
    y_true = clean[label_column].astype(int)
    scores = clean[score_column].astype(float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame([{
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "roc_auc": _safe_metric(roc_auc_score, y_true, scores),
        "pr_auc": _safe_metric(average_precision_score, y_true, scores),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if tn + fp else None,
        "tpr": float(tp / (tp + fn)) if tp + fn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_mg_rate": float(y_pred.mean()) if len(y_pred) else None,
        "row_count": int(len(clean)),
    }])


def calibration_table(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    if bins < 1:
        raise ValueError("bins must be positive")
    clean = predictions.loc[:, [label_column, score_column]].dropna().copy()
    clean["bin"] = pd.cut(clean[score_column], bins=bins, labels=False, include_lowest=True)
    result = (
        clean.groupby("bin", dropna=False)
        .agg(sample_count=(label_column, "size"), mean_score=(score_column, "mean"), empirical_positive_rate=(label_column, "mean"))
        .reset_index()
    )
    return result
