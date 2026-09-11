from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from notebooks.utils.local_evidence_comparison_utils import expand_model_scores_to_tokens


def predict_boundary_positions(
    scored_windows: pd.DataFrame,
    token_counts: Mapping[int, int],
    *,
    threshold: float = 0.5,
    smoothing_window: int = 5,
    sustained_length: int = 3,
    show_progress: bool = False,
) -> pd.DataFrame:
    required = {"model", "window_length", "text_id", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if sustained_length <= 0:
        raise ValueError("sustained_length must be positive")

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
        smoothed_scores = (
            pd.Series(scores)
            .rolling(smoothing_window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        above_threshold = np.isfinite(smoothed_scores) & (smoothed_scores >= threshold)
        run_length = 0
        boundary = token_count
        crossed = False
        for token_position, is_above_threshold in enumerate(above_threshold):
            run_length = run_length + 1 if is_above_threshold else 0
            if run_length >= sustained_length:
                boundary = token_position - sustained_length + 1
                crossed = True
                break
        rows.append(
            {
                "model": model,
                "window_length": int(window_length),
                "text_id": text_id,
                "predicted_token_boundary": boundary,
                "token_count": token_count,
                "no_crossing": not crossed,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["model", "window_length", "text_id", "predicted_token_boundary", "token_count", "no_crossing"],
    )


def _canonicalize_reconstructed_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufffd", "")).strip()


def _is_small_character_loss(expected: str, actual: str) -> bool:
    loss = len(expected) - len(actual)
    if not 1 <= loss <= 2:
        return False

    expected_index = 0
    actual_index = 0
    skipped = 0
    skipped_characters: list[str] = []
    while expected_index < len(expected) and actual_index < len(actual):
        if expected[expected_index] == actual[actual_index]:
            expected_index += 1
            actual_index += 1
        else:
            expected_index += 1
            skipped += 1
            skipped_characters.append(expected[expected_index - 1])
            if skipped > 2:
                return False
    skipped_characters.extend(expected[expected_index:])
    skipped += len(expected) - expected_index
    return skipped == loss and all(ord(character) < 128 for character in skipped_characters)


def _matches_replacement_artifacts(expected: str, actual: str) -> tuple[bool, bool]:
    expected_index = 0
    actual_index = 0
    masked_whitespace = False
    while actual_index < len(actual):
        if actual[actual_index] == "\ufffd":
            while actual_index < len(actual) and actual[actual_index] == "\ufffd":
                actual_index += 1
            if expected_index >= len(expected) or ord(expected[expected_index]) < 128:
                return False, False
            masked_whitespace = masked_whitespace or expected[expected_index].isspace()
            expected_index += 1
            continue
        if expected_index >= len(expected) or actual[actual_index] != expected[expected_index]:
            return False, False
        actual_index += 1
        expected_index += 1
    return expected_index == len(expected), masked_whitespace


def _gold_token_boundary(text: str, word_boundary: int, token_labels: Sequence[str]) -> int:
    normalized_text = _canonicalize_reconstructed_text(text)
    expected_raw = re.sub(r"[ \t\r\n\f\v]+", " ", text).strip()
    decoded_raw = re.sub(r"[ \t\r\n\f\v]+", " ", "".join(map(str, token_labels))).strip()
    decoded_text = _canonicalize_reconstructed_text(decoded_raw)
    replacement_match, masked_whitespace = _matches_replacement_artifacts(expected_raw, decoded_raw)
    if not replacement_match and (
        decoded_text == normalized_text or not _is_small_character_loss(normalized_text, decoded_text)
    ):
        raise ValueError("token labels do not reconstruct the annotated text")
    boundary_text = normalized_text if masked_whitespace else decoded_text if replacement_match else normalized_text
    words = boundary_text.split()
    if not 0 <= word_boundary < len(words):
        raise ValueError("boundary_word_index is outside the text word range")
    character_boundary = len(" ".join(words[:word_boundary]))
    if replacement_match:
        expected_index = 0
        replacement_run_continues = False
        for token_index, token in enumerate(token_labels):
            raw_token = re.sub(r"[ \t\r\n\f\v]+", " ", str(token))
            token_index_in_raw = 0
            while token_index_in_raw < len(raw_token):
                if raw_token[token_index_in_raw] != "\ufffd":
                    replacement_run_continues = False
                if (
                    raw_token[token_index_in_raw] == " "
                    and expected_index < len(expected_raw)
                    and expected_raw[expected_index] != " "
                ):
                    token_index_in_raw += 1
                    continue
                if raw_token[token_index_in_raw] == "\ufffd":
                    while (
                        token_index_in_raw < len(raw_token)
                        and raw_token[token_index_in_raw] == "\ufffd"
                    ):
                        token_index_in_raw += 1
                    if replacement_run_continues:
                        continue
                    if (
                        expected_index >= len(expected_raw)
                        or ord(expected_raw[expected_index]) < 128
                    ):
                        raise ValueError("token labels do not reconstruct the annotated text")
                    replacement_run_continues = True
                else:
                    if (
                        expected_index >= len(expected_raw)
                        or raw_token[token_index_in_raw] != expected_raw[expected_index]
                    ):
                        raise ValueError("token labels do not reconstruct the annotated text")
                    token_index_in_raw += 1
                expected_index += 1
            if len(_canonicalize_reconstructed_text(expected_raw[:expected_index])) >= character_boundary:
                return 0 if word_boundary == 0 else token_index + 1
        raise ValueError("word boundary does not map to a decoded token")
    character_end = 0
    saw_token_text = False
    for token_index, token in enumerate(token_labels):
        normalized_token = _canonicalize_reconstructed_text(str(token))
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
        try:
            gold_boundary = _gold_token_boundary(
                str(annotation.text), int(annotation.boundary_word_index), labels
            )
        except ValueError as error:
            raise ValueError(f"{error} for text id {text_id}") from error
        gold_boundaries.append(gold_boundary)

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
        if {"predicted_token_boundary", "gold_token_boundary"}.issubset(group.columns):
            predicted = group["predicted_token_boundary"].to_numpy(dtype="float64")
            gold = group["gold_token_boundary"].to_numpy(dtype="float64")
            valid_boundaries = np.isfinite(predicted) & np.isfinite(gold)
            if valid_boundaries.any():
                early_rate = float(np.mean(predicted[valid_boundaries] < gold[valid_boundaries]))
                late_rate = float(np.mean(predicted[valid_boundaries] > gold[valid_boundaries]))
            else:
                early_rate = float("nan")
                late_rate = float("nan")
        else:
            early_rate = float("nan")
            late_rate = float("nan")
        if "no_crossing" in group:
            no_crossing_rate = float(
                group["no_crossing"].astype("boolean").mean()
            )
        elif {"predicted_token_boundary", "token_count"}.issubset(group.columns):
            no_crossing_rate = float(
                np.mean(
                    group["predicted_token_boundary"].to_numpy(dtype="float64")
                    == group["token_count"].to_numpy(dtype="float64")
                )
            )
        else:
            no_crossing_rate = float("nan")
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
                "early_rate": early_rate,
                "late_rate": late_rate,
                "no_crossing_rate": no_crossing_rate,
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
            "early_rate",
            "late_rate",
            "no_crossing_rate",
        ],
    )


def _merge_short_region_labels(labels: np.ndarray, minimum_region_length: int) -> np.ndarray:
    merged = labels.copy()
    while True:
        changes = False
        starts = np.flatnonzero(np.r_[True, merged[1:] != merged[:-1]])
        stops = np.r_[starts[1:], len(merged)]
        if len(starts) == 1:
            return merged
        for index, (start, stop) in enumerate(zip(starts, stops)):
            if stop - start >= minimum_region_length:
                continue
            if index == 0:
                merged[start:stop] = merged[stops[index]]
            elif index == len(starts) - 1:
                merged[start:stop] = merged[starts[index - 1]]
            else:
                merged[start:stop] = merged[starts[index - 1]]
            changes = True
            break
        if not changes:
            return merged


def _regions_from_labels(labels: np.ndarray) -> list[tuple[int, int, str]]:
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    stops = np.r_[starts[1:], len(labels)]
    return [(int(start), int(stop), str(labels[start])) for start, stop in zip(starts, stops)]


def predict_mg_regions(
    scored_windows: pd.DataFrame,
    token_counts: Mapping[int, int],
    *,
    threshold: float = 0.5,
    smoothing_window: int = 5,
    minimum_region_length: int = 3,
    show_progress: bool = False,
) -> pd.DataFrame:
    required = {"model", "window_length", "text_id", "start_token", "stop_token", "mg_score"}
    missing = required.difference(scored_windows.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if minimum_region_length <= 0:
        raise ValueError("minimum_region_length must be positive")

    rows: list[dict[str, object]] = []
    grouped = scored_windows.groupby(["model", "window_length", "text_id"], sort=True)
    groups = tqdm(grouped, total=grouped.ngroups, desc="Finding MG regions", unit="text") if show_progress else grouped
    for (model, window_length, text_id), text_windows in groups:
        text_id = int(text_id)
        try:
            token_count = int(token_counts[text_id])
        except KeyError as error:
            raise KeyError(f"missing token count for text id {text_id}") from error
        expanded = expand_model_scores_to_tokens(text_windows, token_count)
        scores = expanded.sort_values("token_position")["mg_score"].to_numpy(dtype="float64")
        if not np.isfinite(scores).all():
            raise ValueError(f"token-level MG scores do not cover text id {text_id}")
        smoothed_scores = (
            pd.Series(scores)
            .rolling(smoothing_window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        labels = np.where(smoothed_scores >= threshold, "MG", "HG")
        labels = _merge_short_region_labels(labels, minimum_region_length)
        for start_token, stop_token, label in _regions_from_labels(labels):
            rows.append(
                {
                    "model": model,
                    "window_length": int(window_length),
                    "text_id": text_id,
                    "start_token": start_token,
                    "stop_token": stop_token,
                    "label": label,
                    "token_count": token_count,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["model", "window_length", "text_id", "start_token", "stop_token", "label", "token_count"],
    )


def build_gold_mg_regions(
    annotations: pd.DataFrame,
    token_labels: Mapping[int, Sequence[str]],
) -> pd.DataFrame:
    required = {"id", "text", "boundary_word_index"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"missing annotation columns: {', '.join(sorted(missing))}")
    if annotations["id"].duplicated().any():
        raise ValueError("annotation ids must be unique")

    rows: list[dict[str, object]] = []
    for annotation in annotations.itertuples(index=False):
        text_id = int(annotation.id)
        try:
            labels = token_labels[text_id]
        except KeyError as error:
            raise KeyError(f"missing SemEval tokens for text id {text_id}") from error
        boundary = _gold_token_boundary(
            str(annotation.text), int(annotation.boundary_word_index), labels
        )
        token_count = len(labels)
        for start_token, stop_token, label in ((0, boundary, "HG"), (boundary, token_count, "MG")):
            if stop_token == start_token:
                continue
            rows.append(
                {
                    "text_id": text_id,
                    "start_token": start_token,
                    "stop_token": stop_token,
                    "label": label,
                    "token_count": token_count,
                    "gold_token_boundary": boundary,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["text_id", "start_token", "stop_token", "label", "token_count", "gold_token_boundary"],
    )


def evaluate_mg_regions(predicted_regions: pd.DataFrame, gold_regions: pd.DataFrame) -> pd.DataFrame:
    required_predicted = {"model", "window_length", "text_id", "start_token", "stop_token", "label", "token_count"}
    required_gold = {"text_id", "start_token", "stop_token", "label", "token_count", "gold_token_boundary"}
    missing_predicted = required_predicted.difference(predicted_regions.columns)
    missing_gold = required_gold.difference(gold_regions.columns)
    if missing_predicted:
        raise ValueError(f"missing predicted region columns: {', '.join(sorted(missing_predicted))}")
    if missing_gold:
        raise ValueError(f"missing gold region columns: {', '.join(sorted(missing_gold))}")

    rows: list[dict[str, object]] = []
    gold_by_text = {int(text_id): group for text_id, group in gold_regions.groupby("text_id")}
    for (model, window_length, text_id), predicted in predicted_regions.groupby(
        ["model", "window_length", "text_id"], sort=True
    ):
        text_id = int(text_id)
        if text_id not in gold_by_text:
            raise KeyError(f"missing gold regions for text id {text_id}")
        gold = gold_by_text[text_id]
        token_count = int(gold["token_count"].iloc[0])
        predicted_labels = np.full(token_count, "HG", dtype=object)
        for region in predicted.itertuples(index=False):
            predicted_labels[int(region.start_token):int(region.stop_token)] = region.label
        gold_labels = np.full(token_count, "HG", dtype=object)
        for region in gold.itertuples(index=False):
            gold_labels[int(region.start_token):int(region.stop_token)] = region.label
        predicted_mg = predicted_labels == "MG"
        gold_mg = gold_labels == "MG"
        true_positive = int(np.sum(predicted_mg & gold_mg))
        false_positive = int(np.sum(predicted_mg & ~gold_mg))
        false_negative = int(np.sum(~predicted_mg & gold_mg))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        mg_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        predicted_mg_regions = [region for region in _regions_from_labels(predicted_labels) if region[2] == "MG"]
        gold_mg_regions = [region for region in _regions_from_labels(gold_labels) if region[2] == "MG"]
        spurious = sum(not np.any(gold_mg[int(start):int(stop)]) for start, stop, _ in predicted_mg_regions)
        missed = sum(not any(np.any(predicted_mg[max(start, other_start):min(stop, other_stop)]) for other_start, other_stop, _ in predicted_mg_regions) for start, stop, _ in gold_mg_regions)
        predicted_boundaries = [start for index, (start, _, label) in enumerate(_regions_from_labels(predicted_labels)) if label == "MG" and index > 0]
        gold_boundary = int(gold["gold_token_boundary"].iloc[0])
        nearest_boundary = min(predicted_boundaries, key=lambda boundary: abs(boundary - gold_boundary), default=None)
        rows.append(
            {
                "model": model,
                "window_length": int(window_length),
                "text_id": text_id,
                "token_count": token_count,
                "accuracy": float(np.mean(predicted_labels == gold_labels)),
                "mg_precision": precision,
                "mg_recall": recall,
                "mg_f1": mg_f1,
                "predicted_mg_region_count": len(predicted_mg_regions),
                "spurious_mg_region_count": int(spurious),
                "missed_mg_region_count": int(missed),
                "exact_region_match": predicted_mg.tolist() == gold_mg.tolist(),
                "gold_token_boundary": gold_boundary,
                "nearest_predicted_hg_to_mg_boundary": nearest_boundary,
                "boundary_error": abs(nearest_boundary - gold_boundary) if nearest_boundary is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_mg_region_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "window_length", "text_id", "accuracy", "mg_precision", "mg_recall", "mg_f1"}
    missing = required.difference(evaluations.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {', '.join(sorted(missing))}")
    aggregations = evaluations.groupby(["model", "window_length"], sort=True)
    result = aggregations[["accuracy", "mg_precision", "mg_recall", "mg_f1", "boundary_error"]].mean(numeric_only=True).reset_index()
    result.insert(2, "texts", aggregations["text_id"].nunique().to_numpy())
    for column in ("exact_region_match", "spurious_mg_region_count", "missed_mg_region_count"):
        if column in evaluations:
            values = aggregations[column].mean(numeric_only=True).rename(column)
            result = result.merge(values.reset_index(), on=["model", "window_length"])
    return result
