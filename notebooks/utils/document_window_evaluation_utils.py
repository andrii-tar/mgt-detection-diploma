"""Utilities for evaluating document predictions from window-level scores."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


KEY_COLUMNS = ("source_dataset", "split", "text_id")
WINDOW_COLUMNS = (*KEY_COLUMNS, "window_id")


@dataclass(frozen=True)
class DocumentWindowEvaluationConfig:
    run_id: str
    scorer_llms: tuple[str, ...]
    model_families: tuple[str, ...]
    training_regime: str
    evaluation_regimes: tuple[str, ...]
    feature_set: str
    training_window: int
    evaluation_windows: tuple[int, ...]
    partition: str
    document_threshold: float = 0.5
    sample_mode: str = "common"
    batch_size: int = 100_000


def _with_compatibility_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    aliases = {
        "dataset": "source_dataset",
        "text_index": "text_id",
        "window_index": "window_id",
        "start_token": "window_id",
        "score": "mg_score",
        "y_true": "label",
    }
    for source, target in aliases.items():
        if target not in rows and source in rows:
            rows[target] = rows[source]
    missing = [column for column in WINDOW_COLUMNS + ("label", "mg_score") if column not in rows]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    return rows


def _document_key(rows: pd.DataFrame) -> pd.Series:
    return rows[list(KEY_COLUMNS)].astype(str).agg("::".join, axis=1)


def _validate_window_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = _with_compatibility_columns(rows)
    if rows.empty:
        raise ValueError("Prediction rows cannot be empty")
    if rows[list(WINDOW_COLUMNS)].isna().any().any():
        raise ValueError("Document and window keys cannot be null")
    if not np.isfinite(pd.to_numeric(rows["mg_score"], errors="coerce")).all():
        raise ValueError("Window scores must be finite")
    if not rows["mg_score"].astype(float).between(0, 1).all():
        raise ValueError("Window scores must be in [0, 1]")
    if rows.duplicated(list(WINDOW_COLUMNS)).any():
        raise ValueError("Prediction rows contain duplicate windows")
    labels = rows.groupby(list(KEY_COLUMNS), dropna=False)["label"].nunique(dropna=False)
    if (labels > 1).any():
        raise ValueError("A document has conflicting labels")
    return rows


def aggregate_window_predictions(rows: pd.DataFrame, document_threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate each document's continuous window MG-scores by arithmetic mean."""
    if not 0 <= document_threshold <= 1:
        raise ValueError("document_threshold must be in [0, 1]")
    rows = _validate_window_rows(rows)
    rows["document_key"] = _document_key(rows)
    result = (
        rows.groupby([*KEY_COLUMNS, "document_key"], as_index=False, sort=False)
        .agg(
            label=("label", "first"),
            document_score=("mg_score", "mean"),
            window_count=("window_id", "nunique"),
        )
    )
    result["prediction"] = (result["document_score"] >= document_threshold).astype(int)
    return result


def aggregate_window_chunks(chunks: Iterable[pd.DataFrame], document_threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate rows from multiple artifacts without changing their document identity."""
    frames = [chunk for chunk in chunks if chunk is not None and not chunk.empty]
    if not frames:
        raise ValueError("At least one non-empty prediction chunk is required")
    return aggregate_window_predictions(pd.concat(frames, ignore_index=True), document_threshold)


def common_document_keys(frames: Sequence[pd.DataFrame]) -> set[str]:
    """Return the intersection of fully qualified document keys across frames."""
    if not frames:
        return set()
    key_sets = []
    for frame in frames:
        if "document_key" in frame:
            key_sets.append(set(frame["document_key"].dropna().astype(str)))
            continue
        normalized = _with_compatibility_columns(frame)
        key_sets.append(set(_document_key(normalized)))
    return set.intersection(*key_sets)


def evaluate_document_predictions(documents: pd.DataFrame, document_threshold: float = 0.5) -> dict[str, float | int | str]:
    """Compute metrics on one row per document, never on window rows."""
    required = {"label", "document_score"}
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(f"Missing required document columns: {sorted(missing)}")
    if documents.empty:
        raise ValueError("Document predictions cannot be empty")
    predictions = (documents["document_score"] >= document_threshold).astype(int)
    labels = documents["label"].astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    result = {
        "document_count": int(len(documents)),
        "hg_document_count": int((labels == 0).sum()),
        "mg_document_count": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, documents["document_score"])) if labels.nunique() == 2 else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": fp / (tn + fp) if tn + fp else float("nan"),
        "evaluation_unit": "document",
        "aggregation": "mean_window_score",
        "threshold": float(document_threshold),
    }
    return result


def document_coverage(documents: pd.DataFrame, *, available_count: int | None = None) -> dict[str, int | float]:
    """Summarize usable and excluded document counts for one configuration."""
    usable = int(documents["document_key"].nunique()) if "document_key" in documents else int(len(documents))
    available = usable if available_count is None else int(available_count)
    return {
        "available_document_count": available,
        "evaluated_document_count": usable,
        "excluded_document_count": max(available - usable, 0),
        "coverage": usable / available if available else 0.0,
    }


def inspect_prediction_artifacts(root: Path, *, expected_metadata: Mapping[str, object] | None = None) -> pd.DataFrame:
    """Report metadata availability and mismatches without loading prediction tables."""
    expected_metadata = expected_metadata or {}
    rows = []
    metadata_paths = sorted(root.rglob("metadata.json"))
    if not metadata_paths:
        return pd.DataFrame([{"artifact_path": str(root), "reason": "missing_metadata"}])
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected_metadata.items() if metadata.get(key) != value]
        rows.append({
            "artifact_path": str(metadata_path.parent),
            "reason": "metadata_mismatch" if mismatches else "accepted",
            "mismatched_fields": ",".join(mismatches),
            **metadata,
        })
    return pd.DataFrame(rows)


def evaluate_configurations(
    documents_by_configuration: Mapping[str, pd.DataFrame],
    *,
    document_threshold: float = 0.5,
    sample_mode: str = "common",
    common_group_columns: tuple[str, ...] = ("evaluation_regime",),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate configurations on either their own or their common document sample."""
    if sample_mode not in {"common", "available"}:
        raise ValueError("sample_mode must be 'common' or 'available'")
    if not documents_by_configuration:
        raise ValueError("No document configurations loaded; check the prediction artifact root and selection")
    filtered = filter_common_documents(documents_by_configuration, common_group_columns=common_group_columns) if sample_mode == "common" else dict(documents_by_configuration)
    common_scope = "+".join(common_group_columns) if sample_mode == "common" else "available"
    metric_rows = []
    coverage_rows = []
    prediction_rows = []
    for configuration, frame in filtered.items():
        metrics = evaluate_document_predictions(frame, document_threshold)
        dimensions = {
            column: frame[column].iloc[0]
            for column in ("scorer_llm", "model_family", "evaluation_regime", "evaluation_window", "training_window", "training_regime", "feature_set", "partition")
            if column in frame and not frame.empty
        }
        metrics.update({"configuration": configuration, "sample_mode": sample_mode, "common_scope": common_scope, **dimensions})
        metric_rows.append(metrics)
        coverage = document_coverage(frame, available_count=len(documents_by_configuration[configuration]))
        coverage.update({"configuration": configuration, "sample_mode": sample_mode, "common_scope": common_scope, **dimensions})
        coverage_rows.append(coverage)
        prediction_rows.append(frame.assign(configuration=configuration, sample_mode=sample_mode, common_scope=common_scope))
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows)


def write_document_window_outputs(
    output_dir: Path,
    *,
    document_predictions: pd.DataFrame,
    document_metrics: pd.DataFrame,
    document_coverage: pd.DataFrame,
    manifest: Mapping[str, object],
) -> None:
    """Write deterministic tabular outputs and a manifest, replacing prior report files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sort_columns = [column for column in (*KEY_COLUMNS, "configuration") if column in document_predictions]
    document_predictions.sort_values(sort_columns).to_parquet(
        output_dir / "document_predictions.parquet", index=False
    )
    document_metrics.to_csv(output_dir / "document_metrics.csv", index=False)
    document_coverage.to_csv(output_dir / "document_coverage.csv", index=False)
    (output_dir / "run_manifest.json").write_text(json.dumps(dict(manifest), indent=2, default=str) + "\n", encoding="utf-8")


def plot_macro_f1_comparison(metrics: pd.DataFrame):
    """Plot macro-F1 by configuration using the caller's plotting backend."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4))
    if not metrics.empty:
        metrics.plot.bar(x="configuration", y="macro_f1", ax=axis, legend=False)
    axis.set_ylabel("Document macro-F1")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    return figure


def plot_fpr_mg_recall(metrics: pd.DataFrame):
    """Plot false-positive rate against MG recall when those columns are available."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    if not metrics.empty and {"fpr", "recall"}.issubset(metrics.columns):
        axis.scatter(metrics["fpr"], metrics["recall"])
    axis.set_xlabel("FPR")
    axis.set_ylabel("MG recall")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    return figure


def plot_confusion_matrices(metrics: pd.DataFrame):
    """Return a compact confusion-matrix figure when matrix columns are exported."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.axis("off")
    axis.set_title("Document confusion matrices")
    return figure


def iter_prediction_artifacts(root: Path, *, partition: str = "test") -> Iterator[tuple[Path, dict]]:
    """Yield prediction artifact directories and metadata without loading their tables."""
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("partition") == partition:
            yield metadata_path.parent, metadata


def load_prediction_artifact(artifact_dir: Path) -> pd.DataFrame:
    """Load a prediction table from an artifact directory."""
    candidates = sorted(artifact_dir.glob("*.parquet")) + sorted(artifact_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No prediction table found in {artifact_dir}")
    path = candidates[0]
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)


def filter_common_documents(
    documents_by_configuration: Mapping[str, pd.DataFrame], *,
    common_group_columns: tuple[str, ...] = ("evaluation_regime",),
) -> dict[str, pd.DataFrame]:
    """Intersect document populations within an explicit comparison scope."""
    regimes: dict[tuple, dict[str, pd.DataFrame]] = {}
    for name, frame in documents_by_configuration.items():
        values = []
        for column in common_group_columns:
            if column == "evaluation_regime" and column not in frame:
                values.append("default")
                continue
            if column not in frame or frame.empty or frame[column].isna().any() or frame[column].nunique() != 1:
                raise ValueError(f"Configuration {name} must have one non-null {column} value")
            values.append(str(frame[column].iloc[0]))
        regime = tuple(values)
        regimes.setdefault(regime, {})[name] = frame
    result = {}
    for regime, frames in regimes.items():
        keys = common_document_keys(list(frames.values()))
        if not keys:
            raise ValueError(f"No common documents for evaluation regime {regime}")
        result.update({name: frame[frame["document_key"].isin(keys)].copy() for name, frame in frames.items()})
    return result


def load_document_configurations(
    root: Path, *, scorer_llms: Sequence[str], model_families: Sequence[str],
    evaluation_regimes: Sequence[str], evaluation_windows: Sequence[int],
    training_window: int, feature_set: str, partition: str,
    training_regime: str | None = None, document_threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Read selected prediction artifacts, retaining only document summaries in memory."""
    import pyarrow.parquet as pq
    from itertools import product

    selections = {
        "scorer_llm": tuple(scorer_llms), "model_family": tuple(model_families),
        "evaluation_regime": tuple(evaluation_regimes),
        "evaluation_window": tuple(str(value) for value in evaluation_windows),
        "training_window": (str(training_window),), "feature_set": (feature_set,), "partition": (partition,),
    }
    if training_regime is not None:
        selections["training_regime"] = (training_regime,)
    summaries: dict[str, list[pd.DataFrame]] = {}
    dimensions = ("model_family", "scorer_llm", "evaluation_regime", "evaluation_window")
    for path in sorted(root.rglob("data.parquet")):
        # Atomic artifact writes leave unfinished Parquet files in hidden .tmp
        # directories. Only finalized artifacts with a metadata sidecar are inputs.
        if any(part.startswith(".") or part.endswith(".tmp") for part in path.relative_to(root).parts[:-1]):
            continue
        if not (path.parent / "metadata.json").is_file():
            continue
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if not set(selections).issubset(names):
            continue
        identity = parquet.read_row_group(0, columns=list(selections)).slice(0, 1).to_pandas()
        if identity.empty or any(str(identity[column].iloc[0]) not in values for column, values in selections.items()):
            continue
        columns = [column for column in (*KEY_COLUMNS, "label", "mg_score", "start_token", "window_id", *selections) if column in names]
        rows = parquet.read(columns=list(dict.fromkeys(columns))).to_pandas()
        for column, values in selections.items():
            if not rows[column].astype(str).isin(values).all():
                raise ValueError(f"Inconsistent {column} in prediction artifact {path}")
        documents = aggregate_window_predictions(rows, document_threshold)
        for column in selections:
            documents[column] = rows[column].iloc[0]
        name = "__".join(str(rows[column].iloc[0]) for column in dimensions)
        summaries.setdefault(name, []).append(documents)
    expected = {"__".join((family, scorer, regime, str(window))) for family, scorer, regime, window in product(model_families, scorer_llms, evaluation_regimes, evaluation_windows)}
    missing = expected - summaries.keys()
    if missing:
        raise ValueError(f"Missing prediction configurations under {root}: {sorted(missing)}")
    result = {}
    for name, frames in summaries.items():
        documents = pd.concat(frames, ignore_index=True)
        if documents["document_key"].duplicated().any():
            raise ValueError(f"Duplicate document artifacts for {name}")
        result[name] = documents
    return result
