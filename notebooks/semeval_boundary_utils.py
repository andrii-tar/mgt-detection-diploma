from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from notebooks.local_evidence_comparison_utils import expand_model_scores_to_tokens


def predict_boundary_positions(
    scored_windows: pd.DataFrame,
    token_counts: Mapping[int, int],
    *,
    threshold: float = 0.5,
    show_progress: bool = False,
) -> pd.DataFrame:
    required = {"model", "window_length", "text_id", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    rows: list[dict[str, object]] = []
    grouped = scored_windows.groupby(["model", "window_length", "text_id"], sort=True)
    groups = tqdm(grouped, total=grouped.ngroups, desc="Finding SemEval boundaries", unit="text") if show_progress else grouped
    for (model, window_length, text_id), text_windows in groups:
        text_id = int(text_id)
        try:
            token_count = int(token_counts[text_id])
        except KeyError as error:
            raise KeyError(f"missing token count for text id {text_id}") from error
        expanded = expand_model_scores_to_tokens(text_windows, token_count)
        scores = expanded.sort_values("token_position")["mg_score"].to_numpy(dtype="float64")
        matches = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
        rows.append(
            {
                "model": model,
                "window_length": int(window_length),
                "text_id": text_id,
                "predicted_token_boundary": int(matches[0]) if len(matches) else token_count,
                "token_count": token_count,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["model", "window_length", "text_id", "predicted_token_boundary", "token_count"],
    )


def _gold_token_boundary(text: str, word_boundary: int, token_labels: Sequence[str]) -> int:
    words = text.split()
    if not 0 <= word_boundary < len(words):
        raise ValueError("boundary_word_index is outside the text word range")
    normalized_text = re.sub(r"\s+", " ", text).strip()
    decoded_text = re.sub(r"\s+", " ", "".join(map(str, token_labels))).strip()
    if decoded_text != normalized_text:
        raise ValueError("token labels do not reconstruct the annotated text")
    character_boundary = len(" ".join(words[:word_boundary]))
    character_end = 0
    saw_token_text = False
    for token_index, token in enumerate(token_labels):
        normalized_token = re.sub(r"\s+", " ", str(token)).strip()
        if normalized_token:
            saw_token_text = True
            character_end += len(normalized_token) + int(character_end > 0)
        if character_end >= character_boundary:
            return 0 if word_boundary == 0 else token_index + 1
    if saw_token_text and character_end >= character_boundary:
        return len(token_labels)
    raise ValueError("word boundary does not map to a decoded token")


def add_gold_boundaries(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    token_labels: Mapping[int, Sequence[str]],
) -> pd.DataFrame:
    required_predictions = {"text_id", "token_count", "predicted_token_boundary"}
    required_annotations = {"id", "text", "boundary_word_index"}
    missing_predictions = required_predictions.difference(predictions.columns)
    missing_annotations = required_annotations.difference(annotations.columns)
    if missing_predictions:
        raise ValueError(f"missing prediction columns: {', '.join(sorted(missing_predictions))}")
    if missing_annotations:
        raise ValueError(f"missing annotation columns: {', '.join(sorted(missing_annotations))}")
    if annotations["id"].duplicated().any():
        raise ValueError("annotation ids must be unique")

    annotation_by_id = annotations.set_index("id")
    result = predictions.copy()
    gold_boundaries: list[int] = []
    for row in result.itertuples(index=False):
        text_id = int(row.text_id)
        try:
            annotation = annotation_by_id.loc[text_id]
            labels = token_labels[text_id]
        except KeyError as error:
            raise KeyError(f"missing SemEval annotation or tokens for text id {text_id}") from error
        if len(labels) != int(row.token_count):
            raise ValueError(f"token count mismatch for text id {text_id}")
        gold_boundaries.append(_gold_token_boundary(str(annotation.text), int(annotation.boundary_word_index), labels))

    result["gold_token_boundary"] = gold_boundaries
    result["absolute_token_error"] = (
        result["predicted_token_boundary"].sub(result["gold_token_boundary"]).abs().astype("int64")
    )
    return result


def summarize_boundary_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "window_length", "absolute_token_error"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for (model, window_length), group in predictions.groupby(["model", "window_length"], sort=True):
        errors = group["absolute_token_error"].to_numpy(dtype="float64")
        rows.append(
            {
                "model": model,
                "window_length": int(window_length),
                "texts": len(group),
                "mean_absolute_token_error": float(errors.mean()),
                "median_absolute_token_error": float(np.median(errors)),
                "exact_accuracy": float(np.mean(errors == 0)),
                "within_1_accuracy": float(np.mean(errors <= 1)),
                "within_3_accuracy": float(np.mean(errors <= 3)),
                "within_5_accuracy": float(np.mean(errors <= 5)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "window_length",
            "texts",
            "mean_absolute_token_error",
            "median_absolute_token_error",
            "exact_accuracy",
            "within_1_accuracy",
            "within_3_accuracy",
            "within_5_accuracy",
        ],
    )
