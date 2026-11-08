from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from notebooks.utils.document_window_evaluation_utils import aggregate_window_predictions
from notebooks.utils.paper_model_comparison_utils import (
    label_semeval_windows,
    load_processed_token_counts,
    load_semeval_token_labels,
    score_prepared_window_partition,
)
from notebooks.utils.local_evidence_comparison_utils import expand_model_scores_to_tokens


REPORT_TABLE_FILES = {
    "T06_window_metrics_long": "T06_window_metrics_long.csv",
    "T07_document_mean_window": "T07_document_mean_window.csv",
    "T08_document_length_metrics": "T08_document_length_metrics.csv",
    "T08_window_vs_whole_text": "T08_window_vs_whole_text.csv",
    "T09_external_document_results": "T09_external_document_results.csv",
}


def load_report_tables(report_root: Path) -> dict[str, pd.DataFrame]:
    """Load the report CSVs required by the Chapter 3 figure suite."""
    table_root = Path(report_root) / "tables"
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REPORT_TABLE_FILES.items():
        path = table_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"required report table is missing: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def save_figure_pair(figure: Figure, figure_root: Path, stem: str) -> tuple[Path, Path]:
    """Save one figure as 300-dpi PNG and SVG, then close its Matplotlib handle."""
    if not stem or Path(stem).name != stem:
        raise ValueError("figure stem must be a non-empty filename component")
    output_root = Path(figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / f"{stem}.png"
    svg_path = output_root / f"{stem}.svg"
    try:
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(svg_path, bbox_inches="tight")
    finally:
        plt.close(figure)
    return png_path, svg_path


def window_transfer_matrix(
    metrics: pd.DataFrame,
    *,
    scorer_llm: str,
    model_family: str,
    evaluation_regime: str,
) -> pd.DataFrame:
    """Pivot one W-block macro-F1 matrix while retaining absent cells as NA."""
    required = {"block", "scorer_llm", "model_family", "evaluation_regime", "training_window", "evaluation_window", "macro_f1"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"window metrics are missing columns: {missing}")
    rows = metrics.loc[
        metrics.block.eq("W")
        & metrics.scorer_llm.eq(scorer_llm)
        & metrics.model_family.eq(model_family)
        & metrics.evaluation_regime.eq(evaluation_regime)
    ]
    if rows.duplicated(["training_window", "evaluation_window"]).any():
        raise ValueError("W-block configuration has duplicate window metrics")
    matrix = rows.pivot(index="training_window", columns="evaluation_window", values="macro_f1")
    return matrix.reindex(index=[32, 64], columns=[32, 64])


def normalize_confusion_counts(*, tn: int, fp: int, fn: int, tp: int) -> np.ndarray:
    """Return a 2x2 confusion matrix normalized by true-class row."""
    counts = np.asarray([[tn, fp], [fn, tp]], dtype="float64")
    if (counts < 0).any():
        raise ValueError("confusion counts must be non-negative")
    supports = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, supports, out=np.full_like(counts, np.nan), where=supports != 0)


def eligible_length_metrics(metrics: pd.DataFrame, *, minimum_class_support: int = 20) -> pd.DataFrame:
    """Keep length bins with the required support in each binary class."""
    required = {"hg_support", "mg_support", "macro_f1"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"length metrics are missing columns: {missing}")
    if minimum_class_support < 1:
        raise ValueError("minimum_class_support must be positive")
    return metrics.loc[
        metrics.hg_support.ge(minimum_class_support)
        & metrics.mg_support.ge(minimum_class_support)
        & metrics.macro_f1.notna()
    ].copy()


def source_llm_recall(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute MG detection rate and support per source generator/configuration."""
    required = {"source_llm", "scorer_llm", "prediction", "model_family", "training_window", "evaluation_window"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Name That LLM predictions are missing columns: {missing}")
    if rows.empty:
        return pd.DataFrame(columns=["source_llm", "model_family", "training_window", "evaluation_window", "document_count", "mg_recall"])
    return (
        rows.groupby(["scorer_llm", "source_llm", "model_family", "training_window", "evaluation_window"], as_index=False, dropna=False)
        .agg(document_count=("prediction", "size"), mg_recall=("prediction", "mean"))
    )


def attach_source_llm_metadata(documents: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach one source-LLM label per document ID without changing row count."""
    if "text_id" not in documents or not {"id", "source_llm"}.issubset(metadata.columns):
        raise ValueError("documents need text_id and metadata needs id/source_llm")
    source = metadata[["id", "source_llm"]].copy()
    source["text_id"] = source.id.astype(str).str.strip()
    source = source.drop(columns="id")
    if source.text_id.duplicated().any():
        raise ValueError("source-LLM metadata must have unique document IDs")
    result = documents.copy()
    result["text_id"] = result.text_id.astype(str).str.strip()
    joined = result.merge(source, on="text_id", how="left", validate="many_to_one")
    if joined.source_llm.isna().any():
        raise ValueError("source-LLM metadata is missing for one or more documents")
    if len(joined) != len(documents):
        raise ValueError("source-LLM metadata join changed the number of document rows")
    return joined


def validate_external_score_counts(scores: pd.DataFrame, t09: pd.DataFrame) -> None:
    """Ensure cached document score cohorts match T09's mean-window support."""
    if scores.empty:
        return
    required_score = {"dataset_label", "scorer_llm", "model_family", "training_window", "evaluation_window", "text_id"}
    required_t09 = {"dataset", "evaluation_unit", "model", "scorer_llm", "model_family", "training_window", "evaluation_window", "document_count"}
    if required_score.difference(scores.columns) or required_t09.difference(t09.columns):
        raise ValueError("external scores and T09 must contain their dataset/configuration/support columns")
    dataset_names = {"BBC-only": "BBC-only", "Name That LLM": "Name That LLM >=250 tokens"}
    summary = t09.loc[t09.evaluation_unit.eq("document") & t09.model.eq("mean_score_baseline")]
    for keys, group in scores.groupby(["dataset_label", "scorer_llm", "model_family", "training_window", "evaluation_window"], dropna=False):
        dataset_label, scorer, family, train_window, eval_window = keys
        expected = summary.loc[
            summary.dataset.eq(dataset_names.get(dataset_label, dataset_label))
            & summary.scorer_llm.astype(str).eq(str(scorer))
            & summary.model_family.astype(str).eq(str(family))
            & summary.training_window.astype(str).eq(str(train_window))
            & summary.evaluation_window.astype(str).eq(str(eval_window))
        ]
        if len(expected) != 1 or pd.isna(expected.document_count.iloc[0]):
            raise ValueError(f"no unique T09 baseline support for {keys}")
        observed_count = group.text_id.astype(str).nunique()
        expected_count = int(expected.document_count.iloc[0])
        if observed_count != expected_count:
            raise ValueError(f"external score support {observed_count} does not match T09 support {expected_count} for {keys}")


def build_external_document_scores(
    data_root: Path,
    *,
    run_id: str,
    scorer_llms: tuple[str, ...] = ("gemma_3_270m_it", "littlelamb"),
    model_families: tuple[str, ...] = ("logistic_regression", "random_forest"),
    training_windows: tuple[int, ...] = (32, 64),
    evaluation_windows: tuple[int, ...] = (32, 64),
    n2llm_limit: int = 1800,
    min_token_count: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score cached external prepared windows with stored classifiers and aggregate per document."""
    if n2llm_limit < 1 or min_token_count < 0:
        raise ValueError("cohort limit must be positive and token minimum non-negative")
    data_root = Path(data_root)
    metadata_path = data_root / "raw" / "name-that-llm" / "name_that_llm_train_MG.csv"
    metadata = pd.read_csv(metadata_path, dtype={"id": str})[["id", "source_llm"]]
    scored_documents: list[pd.DataFrame] = []
    availability_rows = []
    datasets = (("bbc_news_content", "HG", "BBC-only"), ("name_that_llm_train", "MG", "Name That LLM"))
    for scorer in scorer_llms:
        token_counts = None
        try:
            token_counts = load_processed_token_counts(
                data_root, scorer_llm=scorer, dataset="name_that_llm_train", split="MG",
            )
        except FileNotFoundError:
            pass
        for dataset, split, dataset_label in datasets:
            configurations = product(model_families, training_windows, evaluation_windows)
            for family, train_window, eval_window in tqdm(configurations, desc=f"Scoring {dataset_label} · {scorer}", leave=False):
                config = {
                    "dataset": dataset, "dataset_label": dataset_label, "scorer_llm": scorer,
                    "model_family": family, "training_window": train_window,
                    "evaluation_window": eval_window,
                }
                try:
                    windows = score_prepared_window_partition(
                        data_root, run_id=run_id, scorer_llm=scorer, dataset=dataset, split=split,
                        model_family=family, training_regime="Kaggle+MAGE+Outfox",
                        training_window=train_window, evaluation_window=eval_window,
                    )
                except FileNotFoundError as error:
                    availability_rows.append({**config, "status": "unavailable", "reason": str(error)})
                    continue
                except ValueError as error:
                    if str(error).startswith("expected one stored classifier, found []"):
                        availability_rows.append({**config, "status": "unavailable", "reason": str(error)})
                        continue
                    raise
                documents = aggregate_window_predictions(windows, document_threshold=0.5)
                documents = documents.assign(**config)
                if dataset == "name_that_llm_train":
                    if token_counts is None:
                        availability_rows.append({**config, "status": "unavailable", "reason": "scorer token counts missing"})
                        continue
                    documents["text_id"] = documents.text_id.astype(str).str.strip()
                    counts = token_counts.rename(columns={"id": "text_id"}).copy()
                    counts["text_id"] = counts.text_id.astype(str).str.strip()
                    documents = documents.merge(counts, on="text_id", how="left", validate="many_to_one")
                    if documents.token_count.isna().any():
                        raise ValueError(f"missing Name That LLM token counts for {scorer}/{family}/{train_window}/{eval_window}")
                    documents = attach_source_llm_metadata(documents, metadata)
                scored_documents.append(documents)
                availability_rows.append({**config, "status": "available", "reason": ""})
    if not scored_documents:
        empty = pd.DataFrame()
        return empty, pd.DataFrame(availability_rows)
    available = pd.concat(scored_documents, ignore_index=True, sort=False)
    filtered_frames = []
    group_keys = ["dataset", "scorer_llm"]
    config_keys = ["model_family", "training_window", "evaluation_window"]
    common_ids_by_group: dict[tuple[str, str], set[str]] = {}
    shared_n2llm_ids: set[str] = set()
    for (dataset, scorer), group in available.groupby(group_keys, sort=True):
        config_frames = [frame for _, frame in group.groupby(config_keys, sort=True)]
        common_ids = (
            set.intersection(*(set(frame.text_id.astype(str)) for frame in config_frames))
            if config_frames else set()
        )
        common_ids_by_group[(dataset, scorer)] = common_ids
        if dataset == "name_that_llm_train":
            shared_n2llm_ids.update(common_ids)

    selected_n2llm_ids = set(sorted(shared_n2llm_ids, key=int)[-n2llm_limit:])
    for (dataset, scorer), group in available.groupby(group_keys, sort=True):
        selected_ids = (
            selected_n2llm_ids
            if dataset == "name_that_llm_train"
            else common_ids_by_group[(dataset, scorer)]
        )
        current = group.loc[group.text_id.astype(str).isin(selected_ids)].copy()
        if dataset == "name_that_llm_train":
            current = current.loc[current.token_count.ge(min_token_count)]
        filtered_frames.append(current)
    return pd.concat(filtered_frames, ignore_index=True, sort=False), pd.DataFrame(availability_rows)


def plot_f05_external_score_distributions(scores: pd.DataFrame, figure_root: Path) -> list[Path]:
    if scores.empty:
        return []
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharey=True)
    for row, dataset_label in enumerate(("BBC-only", "Name That LLM")):
        for column, family in enumerate(("logistic_regression", "random_forest")):
            axis = axes[row, column]
            current = scores.loc[scores.dataset_label.eq(dataset_label) & scores.model_family.eq(family)]
            if current.empty:
                axis.text(.5, .5, "Немає кешованих результатів", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            current = current.copy()
            current["train/eval"] = current.training_window.astype(str) + "/" + current.evaluation_window.astype(str)
            sns.boxplot(data=current, x="train/eval", y="document_score", hue="scorer_llm", ax=axis, showfliers=False)
            axis.axhline(.5, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"{dataset_label} · {family.replace('_', ' ')} · N={current.text_id.nunique()}")
            axis.set_xlabel("Train/eval window")
            axis.set_ylabel("Document MG score")
            axis.set_ylim(0, 1)
            if row == 0 and column == 1:
                axis.legend(fontsize=7, title="Scorer")
            else:
                legend = axis.get_legend()
                if legend:
                    legend.remove()
    figure.suptitle("F05 · External per-document MG-score distributions · threshold=0.5")
    return save_figure_pair(figure, figure_root, "F05_external_document_scores")


def plot_f06_name_that_llm_source_recall(scores: pd.DataFrame, figure_root: Path) -> list[Path]:
    if scores.empty or "source_llm" not in scores:
        return []
    rows = scores.loc[scores.dataset.eq("name_that_llm_train")]
    summary = source_llm_recall(rows)
    if summary.empty:
        return []
    summary["configuration"] = (
        summary.model_family.str.replace("_", " ") + " " + summary.training_window.astype(str)
        + "/" + summary.evaluation_window.astype(str)
    )
    scorers = [scorer for scorer in ("gemma_3_270m_it", "littlelamb") if scorer in rows.scorer_llm.unique()]
    if not scorers:
        return []
    figure, axes = plt.subplots(1, len(scorers), figsize=(max(8, 4.5 * len(scorers)), max(4, .38 * summary.source_llm.nunique() + 2)),
                                constrained_layout=True, squeeze=False, sharey=True)
    image = None
    for axis, scorer in zip(axes[0], scorers):
        current = summary.loc[summary.scorer_llm.eq(scorer)]
        value_matrix = current.pivot(index="source_llm", columns="configuration", values="mg_recall")
        count_matrix = current.pivot(index="source_llm", columns="configuration", values="document_count")
        value_matrix = value_matrix.sort_index()
        annotations = value_matrix.copy().astype(object)
        for source in value_matrix.index:
            for config in value_matrix.columns:
                value = value_matrix.loc[source, config]
                count = count_matrix.loc[source, config]
                annotations.loc[source, config] = "NA" if pd.isna(value) else f"{value:.0%}\nN={int(count)}"
        image = sns.heatmap(value_matrix, annot=annotations, fmt="", cmap="viridis", vmin=0, vmax=1,
                            ax=axis, cbar=False, linewidths=.3)
        axis.set_title(scorer)
        axis.set_xlabel("Класифікатор · train/eval")
        axis.set_ylabel("Генератор")
        axis.tick_params(axis="x", labelrotation=35)
    if image is not None:
        figure.colorbar(image.collections[0], ax=axes.ravel().tolist(), label="MG recall", shrink=.8)
    figure.suptitle("F06 · Name That LLM · MG recall by source generator")
    return save_figure_pair(figure, figure_root, "F06_name_that_llm_source_recall")


def select_semeval_profile_examples(
    evaluations: pd.DataFrame,
    *,
    correct: bool,
    required_window_lengths: tuple[int, ...] | None = None,
) -> tuple[int, str]:
    """Choose a deterministic low-error (F07) or high-error (F08) SemEval example."""
    required = {"text_id", "absolute_token_error"}
    missing = sorted(required.difference(evaluations.columns))
    if missing:
        raise ValueError(f"SemEval boundary rows are missing columns: {missing}")
    rows = evaluations.copy()
    rows["absolute_token_error"] = pd.to_numeric(rows.absolute_token_error, errors="coerce")
    rows = rows.loc[rows.absolute_token_error.notna() & np.isfinite(rows.absolute_token_error)]
    if required_window_lengths is not None:
        if "window_length" not in rows:
            raise ValueError("SemEval rows need window_length when required windows are specified")
        required = set(required_window_lengths)
        present = rows.groupby("text_id").window_length.agg(lambda values: set(pd.to_numeric(values, errors="raise")))
        eligible_ids = present.index[present.map(lambda values: required.issubset(values))]
        rows = rows.loc[rows.text_id.isin(eligible_ids)]
    if rows.empty:
        raise ValueError("no finite SemEval boundary errors are available for example selection")
    grouped = rows.groupby("text_id", as_index=False).absolute_token_error.mean()
    grouped["text_id_sort"] = pd.to_numeric(grouped.text_id, errors="raise")
    grouped = grouped.sort_values(
        ["absolute_token_error", "text_id_sort"], ascending=[correct, True], kind="stable",
    )
    text_id = int(grouped.iloc[0].text_id)
    rule = (
        "lowest finite boundary error; tie-break by lowest text_id"
        if correct else "highest finite boundary error; tie-break by lowest text_id"
    )
    if required_window_lengths is not None:
        rule += f"; requires window lengths {','.join(map(str, required_window_lengths))}"
    return text_id, rule


def attach_semeval_gold_boundaries(labeled_windows: pd.DataFrame, exclusion_summary: pd.DataFrame) -> pd.DataFrame:
    """Attach per-document gold boundaries returned in the labeling summary."""
    if "text_id" not in labeled_windows or not {"text_id", "gold_token_boundary"}.issubset(exclusion_summary.columns):
        raise ValueError("SemEval labeled windows and exclusion summary need text_id/gold_token_boundary")
    boundaries = exclusion_summary[["text_id", "gold_token_boundary"]]
    if boundaries.text_id.duplicated().any():
        raise ValueError("SemEval exclusion summary must have one gold boundary per document")
    result = labeled_windows.merge(boundaries, on="text_id", how="left", validate="many_to_one")
    if not result.empty and result.gold_token_boundary.isna().any():
        raise ValueError("gold boundaries are missing for labeled SemEval windows")
    return result


def build_semeval_profile_rows(
    data_root: Path,
    *,
    run_id: str,
    scorer_llm: str,
    model_family: str,
    training_window: int,
    evaluation_window: int,
    text_id: int,
) -> pd.DataFrame:
    """Score one selected SemEval document with a saved model and attach gold/predicted boundaries."""
    data_root = Path(data_root)
    dataset = "semeval2024_task8_subtask_c_train"
    scored = score_prepared_window_partition(
        data_root, run_id=run_id, scorer_llm=scorer_llm, dataset=dataset, split="MG",
        model_family=model_family, training_regime="Kaggle+MAGE+Outfox",
        training_window=training_window, evaluation_window=evaluation_window,
    )
    scored = scored.loc[pd.to_numeric(scored.text_id, errors="raise").eq(int(text_id))].copy()
    if scored.empty:
        raise KeyError(f"selected SemEval text {text_id} has no cached scored windows")
    annotation_path = data_root / "raw" / "semevals_2024-task8" / "SemEval2024-task8" / "subtaskC" / "SemEval2024_task8_C_train_MG.csv"
    annotations = pd.read_csv(annotation_path)
    token_labels = load_semeval_token_labels(data_root, scorer_llm=scorer_llm)
    labeled, excluded = label_semeval_windows(scored, annotations, token_labels)
    if labeled.empty:
        reason = excluded.exclusion_reason.iloc[0] if not excluded.empty else "no pure windows"
        raise ValueError(f"selected SemEval text {text_id} has no labelable windows: {reason}")
    labeled = attach_semeval_gold_boundaries(labeled, excluded)
    labeled["token_count"] = labeled.text_id.map(lambda value: len(token_labels[int(value)]))
    boundary_path = data_root / "experiment-results" / "classification" / "evaluations" / run_id / "semeval_boundary_predictions.csv"
    boundary_columns = [
        "model", "window_length", "text_id", "predicted_token_boundary", "scorer_llm",
        "feature_set", "training_regime", "training_window",
    ]
    boundaries = pd.read_csv(boundary_path, usecols=boundary_columns)
    boundaries = boundaries.loc[
        boundaries.model.eq(model_family)
        & boundaries.window_length.astype(str).eq(str(evaluation_window))
        & boundaries.text_id.astype(str).eq(str(text_id))
        & boundaries.scorer_llm.eq(scorer_llm)
        & boundaries.feature_set.eq("simple_similarity")
        & boundaries.training_regime.eq("Kaggle+MAGE+Outfox")
        & boundaries.training_window.astype(str).eq(str(training_window))
    ].drop_duplicates("text_id")
    if len(boundaries) != 1:
        raise ValueError(f"expected one saved SemEval boundary for {scorer_llm}/{model_family}/{training_window}/{evaluation_window}/{text_id}")
    labeled["predicted_token_boundary"] = boundaries.predicted_token_boundary.iloc[0]
    labeled["window_length"] = int(evaluation_window)
    labeled["training_window"] = int(training_window)
    labeled["model_family"] = model_family
    labeled["scorer_llm"] = scorer_llm
    return labeled.sort_values("start_token").reset_index(drop=True)


def _plot_semeval_profile(
    rows: pd.DataFrame,
    figure_root: Path,
    *,
    text_id: int,
    selection_rule: str,
    figure_id: str,
    title: str,
) -> list[Path]:
    required = {"text_id", "window_length", "training_window", "start_token", "stop_token", "mg_score", "gold_token_boundary", "predicted_token_boundary", "token_count"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"SemEval profile rows are missing columns: {missing}")
    current = rows.loc[pd.to_numeric(rows.text_id, errors="raise").eq(int(text_id))].copy()
    if current.empty:
        raise ValueError(f"no SemEval profile rows found for text {text_id}")
    gold_boundaries = current.gold_token_boundary.dropna().unique()
    token_counts = current.token_count.dropna().unique()
    if len(gold_boundaries) != 1 or len(token_counts) != 1:
        raise ValueError("SemEval profile must have one gold boundary and token count")
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for (training_window, evaluation_window), group in current.groupby(["training_window", "window_length"], sort=True):
        group = group.sort_values("start_token")
        midpoint = (group.start_token.astype(float) + group.stop_token.astype(float)) / 2
        axis.plot(midpoint, group.mg_score, marker="o", markersize=3,
                  label=f"train {training_window} / eval {evaluation_window}")
        predicted = group.predicted_token_boundary.dropna()
        if not predicted.empty:
            axis.axvline(float(predicted.iloc[0]), linestyle=":", linewidth=1.2,
                         label=f"межа прогнозу {training_window}/{evaluation_window}")
    gold = float(gold_boundaries[0])
    token_count = int(token_counts[0])
    axis.axvspan(gold, token_count, color="tab:green", alpha=.12, label="еталонний MG-регіон")
    axis.axvline(gold, color="tab:green", linestyle="--", label="еталонна межа HG→MG")
    axis.set_xlim(0, token_count)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Позиція токена")
    axis.set_ylabel("MG-score вікна")
    axis.set_title(f"{figure_id} · {title} · SemEval · text_id={text_id}\n{selection_rule}")
    axis.legend(fontsize=8, ncol=2)
    return save_figure_pair(figure, figure_root, f"{figure_id}_semeval_text_{text_id}")


def plot_f07_semeval_profile(rows: pd.DataFrame, figure_root: Path, *, text_id: int, selection_rule: str) -> list[Path]:
    return _plot_semeval_profile(
        rows, figure_root, text_id=text_id, selection_rule=selection_rule,
        figure_id="F07", title="Успішна локалізація",
    )


def plot_f08_semeval_failure(rows: pd.DataFrame, figure_root: Path, *, text_id: int, selection_rule: str) -> list[Path]:
    return _plot_semeval_profile(
        rows, figure_root, text_id=text_id, selection_rule=selection_rule,
        figure_id="F08", title="Помилкова локалізація",
    )


def semeval_window_error_metrics(labeled_windows: pd.DataFrame) -> pd.DataFrame:
    """Report pure-window error fractions by scorer/configuration and gold region."""
    configuration_columns = ["scorer_llm", "model_family", "training_window", "evaluation_window"]
    required = {*configuration_columns, "text_id", "prediction", "gold_label"}
    missing = sorted(required.difference(labeled_windows.columns))
    if missing:
        raise ValueError(f"labeled SemEval windows are missing columns: {missing}")
    result_columns = [
        *configuration_columns, "gold_label", "incorrect_fraction", "incorrect_window_count",
        "window_count", "document_count",
    ]
    if labeled_windows.empty:
        return pd.DataFrame(columns=result_columns)
    rows = labeled_windows.copy()
    for column in ("prediction", "gold_label"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if rows[column].isna().any() or not rows[column].isin((0, 1)).all():
            raise ValueError(f"SemEval {column} values must be binary 0/1 labels")
    rows["incorrect"] = rows.prediction.ne(rows.gold_label)
    return (
        rows.groupby([*configuration_columns, "gold_label"], as_index=False, sort=True)
        .agg(
            incorrect_fraction=("incorrect", "mean"),
            incorrect_window_count=("incorrect", "sum"),
            window_count=("text_id", "size"),
            document_count=("text_id", "nunique"),
        )
        .loc[:, result_columns]
    )


def semeval_token_error_metrics(
    scored_windows: pd.DataFrame,
    token_counts: dict[tuple[str, int], int],
    gold_boundaries: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] | np.ndarray,
) -> pd.DataFrame:
    """Measure scorer/config token error fractions across MG-score thresholds."""
    configuration_columns = ["scorer_llm", "model_family", "training_window", "evaluation_window"]
    required = {*configuration_columns, "text_id", "start_token", "stop_token", "mg_score"}
    missing = sorted(required.difference(scored_windows.columns))
    if missing:
        raise ValueError(f"scored SemEval windows are missing columns: {missing}")
    if not {"scorer_llm", "text_id", "gold_token_boundary"}.issubset(gold_boundaries.columns):
        raise ValueError("gold boundaries need scorer_llm, text_id, and gold_token_boundary")
    threshold_values = np.asarray(thresholds, dtype="float64")
    if threshold_values.ndim != 1 or not len(threshold_values) or not np.isfinite(threshold_values).all():
        raise ValueError("thresholds must be a non-empty sequence of finite numbers")
    if (threshold_values < 0).any() or (threshold_values > 1).any() or len(np.unique(threshold_values)) != len(threshold_values):
        raise ValueError("thresholds must be unique and within [0, 1]")

    boundaries = gold_boundaries[["scorer_llm", "text_id", "gold_token_boundary"]].copy()
    boundaries["scorer_llm"] = boundaries.scorer_llm.astype(str)
    boundaries["text_id"] = pd.to_numeric(boundaries.text_id, errors="raise").astype("int64")
    boundaries["gold_token_boundary"] = pd.to_numeric(boundaries.gold_token_boundary, errors="raise")
    if boundaries[["scorer_llm", "text_id"]].duplicated().any():
        raise ValueError("gold boundaries must have one row per scorer/document")
    if not np.isfinite(boundaries.gold_token_boundary).all():
        raise ValueError("gold token boundaries must be finite")
    boundary_by_key = {
        (row.scorer_llm, int(row.text_id)): int(row.gold_token_boundary)
        for row in boundaries.itertuples(index=False)
    }

    result_columns = [
        *configuration_columns, "gold_label", "threshold", "incorrect_fraction",
        "incorrect_token_count", "scored_token_count", "expected_token_count", "coverage", "document_count",
    ]
    if scored_windows.empty:
        return pd.DataFrame(columns=result_columns)
    groups = scored_windows.groupby([*configuration_columns, "text_id"], sort=True, dropna=False)
    document_rows = []
    for group_key, windows in tqdm(
        groups, total=groups.ngroups, desc="Evaluating SemEval token thresholds", leave=False,
    ):
        scorer, model_family, training_window, evaluation_window, text_id = group_key
        scorer = str(scorer)
        text_id = int(text_id)
        boundary_key = (scorer, text_id)
        if boundary_key not in boundary_by_key:
            continue
        token_key = boundary_key
        if token_key not in token_counts:
            raise KeyError(f"missing token count for SemEval {scorer}/{text_id}")
        token_count = token_counts[token_key]
        if isinstance(token_count, bool) or not isinstance(token_count, (int, np.integer)) or token_count <= 0:
            raise ValueError(f"token count must be positive for SemEval {scorer}/{text_id}")
        gold_boundary = boundary_by_key[boundary_key]
        if gold_boundary < 0 or gold_boundary > token_count:
            raise ValueError(f"gold boundary is outside token range for SemEval {scorer}/{text_id}")

        expanded = expand_model_scores_to_tokens(
            windows, int(token_count), model_column="scorer_llm",
        ).sort_values("token_position")
        scores = expanded.mg_score.to_numpy(dtype="float64")
        positions = expanded.token_position.to_numpy(dtype="int64")
        labels = (positions >= gold_boundary).astype("int64")
        finite = np.isfinite(scores)
        for gold_label in (0, 1):
            region_mask = labels == gold_label
            expected_count = int(region_mask.sum())
            if expected_count == 0:
                continue
            scored_mask = region_mask & finite
            scored_count = int(scored_mask.sum())
            for threshold in threshold_values:
                predicted = scores[scored_mask] >= threshold
                incorrect_count = int(np.count_nonzero(predicted != bool(gold_label)))
                document_rows.append({
                    "scorer_llm": scorer, "model_family": model_family,
                    "training_window": training_window, "evaluation_window": evaluation_window,
                    "text_id": text_id, "gold_label": gold_label, "threshold": float(threshold),
                    "incorrect_token_count": incorrect_count, "scored_token_count": scored_count,
                    "expected_token_count": expected_count,
                })
    if not document_rows:
        return pd.DataFrame(columns=result_columns)
    documents = pd.DataFrame(document_rows)
    grouped = documents.groupby(
        [*configuration_columns, "gold_label", "threshold"], as_index=False, sort=True,
    ).agg(
        incorrect_token_count=("incorrect_token_count", "sum"),
        scored_token_count=("scored_token_count", "sum"),
        expected_token_count=("expected_token_count", "sum"),
        document_count=("text_id", "nunique"),
    )
    grouped["incorrect_fraction"] = grouped.incorrect_token_count.div(
        grouped.scored_token_count.where(grouped.scored_token_count.gt(0))
    )
    grouped["coverage"] = grouped.scored_token_count.div(grouped.expected_token_count)
    return grouped.loc[:, result_columns]


def build_semeval_local_error_inputs(
    data_root: Path,
    *,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], int], pd.DataFrame, pd.DataFrame]:
    """Load and score every SemEval configuration listed as available in a saved run summary."""
    data_root = Path(data_root)
    evaluation_root = data_root / "experiment-results" / "classification" / "evaluations" / run_id
    summary_path = evaluation_root / "semeval_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"SemEval configuration summary is missing: {summary_path}")
    summary = pd.read_csv(summary_path)
    required_summary = {
        "run_id", "scorer_llm", "model_family", "training_window", "window_length",
        "feature_set", "training_regime", "evaluation_dataset", "semeval_split",
    }
    missing = sorted(required_summary.difference(summary.columns))
    if missing:
        raise ValueError(f"SemEval configuration summary is missing columns: {missing}")
    available = summary.loc[
        summary.run_id.astype(str).eq(str(run_id))
        & summary.feature_set.eq("simple_similarity")
        & summary.training_regime.eq("Kaggle+MAGE+Outfox")
        & summary.evaluation_dataset.eq("semeval2024_task8_subtask_c_train")
        & summary.semeval_split.eq("train"),
        ["scorer_llm", "model_family", "training_window", "window_length"],
    ].drop_duplicates().copy()
    if available.empty:
        raise ValueError(f"no supported SemEval configurations are listed in {summary_path}")
    for column in ("training_window", "window_length"):
        available[column] = pd.to_numeric(available[column], errors="raise").astype("int64")
    available["scorer_llm"] = available.scorer_llm.astype(str)
    available["model_family"] = available.model_family.astype(str)
    annotation_path = (
        data_root / "raw" / "semevals_2024-task8" / "SemEval2024-task8" / "subtaskC"
        / "SemEval2024_task8_C_train_MG.csv"
    )
    if not annotation_path.is_file():
        raise FileNotFoundError(f"SemEval annotations are missing: {annotation_path}")
    annotations = pd.read_csv(annotation_path)

    token_labels_by_scorer: dict[str, dict[int, list[str]]] = {}
    token_counts: dict[tuple[str, int], int] = {}
    scored_frames = []
    pure_frames = []
    boundary_frames = []
    availability_rows = []
    for config in tqdm(
        available.itertuples(index=False), total=len(available),
        desc="Scoring available SemEval local-error configurations", leave=False,
    ):
        scorer = config.scorer_llm
        family = config.model_family
        training_window = int(config.training_window)
        evaluation_window = int(config.window_length)
        config_details = {
            "scorer_llm": scorer, "model_family": family,
            "training_window": training_window, "evaluation_window": evaluation_window,
        }
        try:
            if scorer not in token_labels_by_scorer:
                token_labels_by_scorer[scorer] = load_semeval_token_labels(data_root, scorer_llm=scorer)
                token_counts.update({
                    (scorer, int(text_id)): len(labels)
                    for text_id, labels in token_labels_by_scorer[scorer].items()
                })
            scored = score_prepared_window_partition(
                data_root, run_id=run_id, scorer_llm=scorer,
                dataset="semeval2024_task8_subtask_c_train", split="MG",
                model_family=family, training_regime="Kaggle+MAGE+Outfox",
                training_window=training_window, evaluation_window=evaluation_window,
            )
            if scored.empty:
                raise ValueError("saved model produced no SemEval window scores")
            for column, value in config_details.items():
                scored[column] = value
            labeled, exclusions = label_semeval_windows(
                scored, annotations, token_labels_by_scorer[scorer],
            )
            if labeled.empty:
                raise ValueError("no pure, gold-labeled windows are available for window-error metrics")
            scored_frames.append(scored)
            pure_frames.append(labeled)
            valid_boundaries = exclusions.loc[
                pd.to_numeric(exclusions.gold_token_boundary, errors="coerce").notna(),
                ["text_id", "gold_token_boundary"],
            ].copy()
            valid_boundaries["scorer_llm"] = scorer
            valid_boundaries["text_id"] = pd.to_numeric(valid_boundaries.text_id, errors="raise").astype("int64")
            boundary_frames.append(valid_boundaries[["scorer_llm", "text_id", "gold_token_boundary"]])
            excluded_count = int(exclusions.gold_token_boundary.isna().sum())
            reasons = sorted(set(exclusions.loc[exclusions.gold_token_boundary.isna(), "exclusion_reason"].dropna()))
            availability_rows.append({
                **config_details, "status": "available", "scored_window_count": int(len(scored)),
                "pure_window_count": int(len(labeled)), "gold_document_count": int(valid_boundaries.text_id.nunique()),
                "excluded_document_count": excluded_count,
                "reason": "" if not reasons else f"unalignable documents excluded: {' | '.join(reasons)}",
            })
        except (FileNotFoundError, KeyError, ValueError) as error:
            availability_rows.append({
                **config_details, "status": "unavailable", "scored_window_count": 0,
                "pure_window_count": 0, "gold_document_count": 0, "excluded_document_count": 0,
                "reason": str(error),
            })
    if not scored_frames:
        reasons = "; ".join(
            f"{row['scorer_llm']}/{row['model_family']}/{row['training_window']}/{row['evaluation_window']}: {row['reason']}"
            for row in availability_rows
        )
        raise ValueError(f"no available SemEval configurations could be scored: {reasons}")
    scored_windows = pd.concat(scored_frames, ignore_index=True)
    pure_windows = pd.concat(pure_frames, ignore_index=True)
    gold_boundaries = pd.concat(boundary_frames, ignore_index=True)
    if gold_boundaries.duplicated(["scorer_llm", "text_id"]).any():
        boundary_counts = gold_boundaries.groupby(["scorer_llm", "text_id"]).gold_token_boundary.nunique()
        if boundary_counts.gt(1).any():
            raise ValueError("SemEval gold boundaries differ across configurations for the same scorer/document")
        gold_boundaries = gold_boundaries.drop_duplicates(["scorer_llm", "text_id"])
    return (
        scored_windows.reset_index(drop=True), pure_windows.reset_index(drop=True), token_counts,
        gold_boundaries.reset_index(drop=True), pd.DataFrame(availability_rows),
    )


def build_semeval_window_boundary_comparison(
    scored_windows: pd.DataFrame,
    token_boundary_rows: pd.DataFrame,
    *,
    threshold: float = 0.5,
    run_length: int = 3,
) -> pd.DataFrame:
    """Compare token-projected and window-run boundary metrics on matching documents."""
    from notebooks.utils.reconstructed_token_mg_score_visualization_utils import (
        detect_window_run_boundaries,
    )

    scored_required = {
        "scorer_llm", "model_family", "training_window", "evaluation_window",
        "text_id", "start_token", "stop_token", "mg_score",
    }
    boundary_required = {
        "scorer_llm", "model_family", "training_window", "window_length", "text_id",
        "token_count", "gold_token_boundary", "predicted_token_boundary", "no_crossing",
        "feature_set", "training_regime", "evaluation_dataset", "semeval_split",
    }
    missing_scored = sorted(scored_required.difference(scored_windows.columns))
    missing_boundary = sorted(boundary_required.difference(token_boundary_rows.columns))
    if missing_scored or missing_boundary:
        raise ValueError(
            f"window boundary comparison inputs are missing columns: scored={missing_scored}, "
            f"token_boundaries={missing_boundary}"
        )

    boundaries = token_boundary_rows.loc[
        token_boundary_rows["feature_set"].eq("simple_similarity")
        & token_boundary_rows["training_regime"].eq("Kaggle+MAGE+Outfox")
        & token_boundary_rows["evaluation_dataset"].eq("semeval2024_task8_subtask_c_train")
        & token_boundary_rows["semeval_split"].eq("train")
    ].copy()
    for column in ("training_window", "window_length", "text_id", "token_count", "gold_token_boundary", "absolute_token_error"):
        boundaries[column] = pd.to_numeric(boundaries[column], errors="coerce")
    boundaries = boundaries.loc[
        boundaries["training_window"].eq(boundaries["window_length"])
        & boundaries["window_length"].isin((32, 64))
    ].copy()
    if boundaries.empty:
        raise ValueError("no matched 32/32 or 64/64 SemEval token-boundary rows are available")
    if boundaries.duplicated(["scorer_llm", "model_family", "window_length", "text_id"]).any():
        raise ValueError("token-boundary rows must be unique by scorer/model/window/text")
    if boundaries[["token_count", "gold_token_boundary", "absolute_token_error"]].isna().any().any():
        raise ValueError("token-boundary rows contain missing token counts or boundary metrics")

    scores = scored_windows.copy()
    for column in ("training_window", "evaluation_window", "text_id"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores = scores.loc[
        scores["training_window"].eq(scores["evaluation_window"])
        & scores["evaluation_window"].isin((32, 64))
    ].copy()
    if "model" not in scores:
        scores["model"] = scores["model_family"].astype(str)

    rows_by_model: dict[tuple[str, str], dict[str, object]] = {}
    config_columns = ["scorer_llm", "model_family", "training_window", "window_length"]
    for configuration, configuration_boundaries in boundaries.groupby(config_columns, sort=True):
        scorer, family, training_window, window_length = configuration
        matching_scores = scores.loc[
            scores["scorer_llm"].astype(str).eq(str(scorer))
            & scores["model_family"].astype(str).eq(str(family))
            & scores["training_window"].eq(training_window)
            & scores["evaluation_window"].eq(window_length)
        ]
        token_errors: list[float] = []
        window_errors: list[float] = []
        detected_window_errors: list[float] = []
        token_no_crossings: list[bool] = []
        window_no_transitions: list[bool] = []
        for boundary in configuration_boundaries.itertuples(index=False):
            text_windows = matching_scores.loc[matching_scores["text_id"].eq(boundary.text_id)]
            if text_windows.empty:
                raise ValueError(
                    "missing scored windows for eligible token-boundary text "
                    f"{scorer}/{family}/{window_length}/{int(boundary.text_id)}"
                )
            prediction_input = text_windows.assign(model=str(family))
            prediction = detect_window_run_boundaries(
                prediction_input, threshold=threshold, run_length=run_length,
            )
            if len(prediction) != 1:
                raise ValueError(
                    f"expected one window-run prediction for {scorer}/{family}/{window_length}/{int(boundary.text_id)}"
                )
            not_found = prediction.iloc[0]["status"] == "not_found"
            predicted_boundary = (
                0.0
                if not_found
                else float(prediction.iloc[0]["predicted_token_boundary"])
            )
            token_predicted_boundary = 0.0 if bool(boundary.no_crossing) else float(boundary.predicted_token_boundary)
            token_errors.append(abs(token_predicted_boundary - float(boundary.gold_token_boundary)))
            window_error = abs(predicted_boundary - float(boundary.gold_token_boundary))
            window_errors.append(window_error)
            if not not_found:
                detected_window_errors.append(window_error)
            token_no_crossings.append(bool(boundary.no_crossing))
            window_no_transitions.append(bool(not_found))

        result_row = rows_by_model.setdefault((str(scorer), str(family)), {
            "scorer_llm": str(scorer), "model_family": str(family),
        })
        suffix = str(int(window_length))
        result_row[f"n_{suffix}"] = len(configuration_boundaries)
        result_row[f"token_mae_{suffix}"] = float(np.mean(token_errors))
        result_row[f"window_mae_{suffix}"] = float(np.mean(window_errors))
        result_row[f"token_no_crossing_pct_{suffix}"] = 100.0 * float(np.mean(token_no_crossings))
        result_row[f"window_no_transition_pct_{suffix}"] = 100.0 * float(np.mean(window_no_transitions))
        result_row[f"n_window_detected_{suffix}"] = len(detected_window_errors)
        result_row[f"window_detected_mae_{suffix}"] = (
            float(np.mean(detected_window_errors)) if detected_window_errors else np.nan
        )
        result_row[f"window_detection_pct_{suffix}"] = (
            100.0 * len(detected_window_errors) / len(configuration_boundaries)
        )

    result_columns = ["scorer_llm", "model_family"]
    for suffix in ("32", "64"):
        result_columns.extend(
            [
                f"n_{suffix}", f"token_mae_{suffix}", f"window_mae_{suffix}",
                f"token_no_crossing_pct_{suffix}", f"window_no_transition_pct_{suffix}",
                f"n_window_detected_{suffix}", f"window_detected_mae_{suffix}",
                f"window_detection_pct_{suffix}",
            ]
        )
    result = pd.DataFrame(rows_by_model.values())
    for column in result_columns:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, result_columns].sort_values(
        ["scorer_llm", "model_family"], kind="stable",
    ).reset_index(drop=True)


def semeval_boundary_threshold_sweep(
    scored_windows: pd.DataFrame,
    token_counts: dict[int, int],
    gold_boundaries: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] | np.ndarray,
    smoothing_window: int = 5,
    sustained_length: int = 3,
) -> pd.DataFrame:
    """Sweep token-score thresholds using the SemEval sustained-run boundary rule."""
    required = {"model", "window_length", "text_id", "start_token", "stop_token", "mg_score"}
    missing = sorted(required.difference(scored_windows.columns))
    if missing:
        raise ValueError(f"scored windows are missing columns: {missing}")
    if not {"text_id", "gold_token_boundary"}.issubset(gold_boundaries.columns):
        raise ValueError("gold boundaries need text_id and gold_token_boundary")
    threshold_values = np.asarray(thresholds, dtype="float64")
    if threshold_values.ndim != 1 or not len(threshold_values) or not np.isfinite(threshold_values).all():
        raise ValueError("thresholds must be a non-empty sequence of finite numbers")
    if (threshold_values < 0).any() or (threshold_values > 1).any() or len(np.unique(threshold_values)) != len(threshold_values):
        raise ValueError("thresholds must be unique and within [0, 1]")
    if smoothing_window <= 0 or smoothing_window % 2 == 0 or sustained_length <= 0:
        raise ValueError("smoothing_window must be positive and odd; sustained_length must be positive")
    gold = gold_boundaries[["text_id", "gold_token_boundary"]].copy()
    gold["_text_id_key"] = gold.text_id.astype(str).str.strip()
    gold["gold_token_boundary"] = pd.to_numeric(gold.gold_token_boundary, errors="coerce")
    gold = gold.loc[gold.gold_token_boundary.notna()]
    if gold._text_id_key.duplicated().any():
        raise ValueError("gold boundaries must have one row per document")
    if gold.empty:
        raise ValueError("no valid gold boundaries are available for the threshold sweep")

    prediction_rows = []
    groups = scored_windows.groupby(["model", "window_length", "text_id"], sort=True)
    for (model, window_length, text_id), windows in tqdm(
        groups, total=groups.ngroups, desc="Sweeping SemEval boundary thresholds", leave=False,
    ):
        text_id_int = int(text_id)
        if text_id_int not in token_counts:
            raise KeyError(f"missing token count for SemEval text {text_id_int}")
        token_count = int(token_counts[text_id_int])
        expanded = expand_model_scores_to_tokens(windows, token_count)
        values = expanded.sort_values("token_position").mg_score.to_numpy(dtype="float64")
        smoothed = pd.Series(values).rolling(smoothing_window, center=True, min_periods=1).median().to_numpy()
        finite_values = np.where(np.isfinite(smoothed), smoothed, -np.inf)
        if len(finite_values) >= sustained_length:
            run_minimums = np.lib.stride_tricks.sliding_window_view(finite_values, sustained_length).min(axis=1)
        else:
            run_minimums = np.asarray([], dtype="float64")
        for threshold in threshold_values:
            crossing_starts = np.flatnonzero(run_minimums >= threshold)
            crossed = bool(len(crossing_starts))
            predicted_boundary = int(crossing_starts[0]) if crossed else 0
            prediction_rows.append({
                "model": model, "window_length": int(window_length), "text_id": text_id_int,
                "threshold": float(threshold), "predicted_token_boundary": predicted_boundary,
                "token_count": token_count, "no_crossing": not crossed,
            })
    predictions = pd.DataFrame(prediction_rows)
    predictions["_text_id_key"] = predictions.text_id.astype(str)
    predictions = predictions.merge(gold["_text_id_key"].to_frame().join(gold.gold_token_boundary),
                                    on="_text_id_key", how="inner", validate="many_to_one")
    if predictions.empty:
        raise ValueError("scored windows and gold boundaries have no common documents")
    predictions["absolute_token_error"] = (
        predictions.predicted_token_boundary - predictions.gold_token_boundary
    ).abs()
    return (
        predictions.groupby(["model", "window_length", "threshold"], as_index=False, sort=True)
        .agg(
            mean_absolute_token_error=("absolute_token_error", "mean"),
            median_absolute_token_error=("absolute_token_error", "median"),
            no_crossing_rate=("no_crossing", "mean"),
            document_count=("text_id", "nunique"),
        )
    )


def build_semeval_threshold_sweep_inputs(
    data_root: Path,
    *,
    run_id: str,
    scorer_llm: str = "gemma_3_270m_it",
    model_family: str = "logistic_regression",
    window_pairs: tuple[tuple[int, int], ...] = ((32, 32), (64, 64)),
) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    """Score cached SemEval feature windows and align them to common valid gold boundaries."""
    data_root = Path(data_root)
    dataset = "semeval2024_task8_subtask_c_train"
    annotation_path = data_root / "raw" / "semevals_2024-task8" / "SemEval2024-task8" / "subtaskC" / "SemEval2024_task8_C_train_MG.csv"
    annotations = pd.read_csv(annotation_path)
    token_labels = load_semeval_token_labels(data_root, scorer_llm=scorer_llm)
    token_counts = {int(text_id): len(labels) for text_id, labels in token_labels.items()}
    scored_frames = []
    gold_frames = []
    valid_id_sets = []
    for train_window, eval_window in tqdm(window_pairs, desc="Scoring SemEval threshold-sweep windows", leave=False):
        scored = score_prepared_window_partition(
            data_root, run_id=run_id, scorer_llm=scorer_llm, dataset=dataset, split="MG",
            model_family=model_family, training_regime="Kaggle+MAGE+Outfox",
            training_window=train_window, evaluation_window=eval_window,
        )
        config_label = f"{model_family} train{train_window}/eval{eval_window}"
        scored["model"] = config_label
        scored["window_length"] = int(eval_window)
        _, exclusions = label_semeval_windows(scored, annotations, token_labels)
        gold = exclusions.loc[
            pd.to_numeric(exclusions.gold_token_boundary, errors="coerce").notna(),
            ["text_id", "gold_token_boundary"],
        ].copy()
        gold["text_id"] = gold.text_id.astype(int)
        if gold.text_id.duplicated().any():
            raise ValueError(f"duplicate gold boundaries for SemEval train/eval {train_window}/{eval_window}")
        scored_frames.append(scored)
        gold_frames.append(gold)
        valid_id_sets.append(set(gold.text_id))
    if not scored_frames:
        raise ValueError("at least one SemEval window pair is required")
    common_ids = set.intersection(*valid_id_sets)
    if not common_ids:
        raise ValueError("no common SemEval documents have valid gold boundaries across window pairs")
    scored = pd.concat(scored_frames, ignore_index=True)
    scored = scored.loc[scored.text_id.astype(int).isin(common_ids)].copy()
    gold_boundaries = pd.concat(gold_frames, ignore_index=True)
    if gold_boundaries.groupby("text_id").gold_token_boundary.nunique().gt(1).any():
        raise ValueError("SemEval gold boundaries differ across scored window configurations")
    gold_boundaries = gold_boundaries.drop_duplicates("text_id").reset_index(drop=True)
    return scored, token_counts, gold_boundaries


def plot_f09_boundary_threshold_sweep(metrics: pd.DataFrame, figure_root: Path) -> list[Path]:
    required = {"model", "window_length", "threshold", "mean_absolute_token_error", "no_crossing_rate"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"threshold sweep metrics are missing columns: {missing}")
    if metrics.empty:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for (model, window), group in metrics.groupby(["model", "window_length"], sort=True):
        group = group.sort_values("threshold")
        label = f"{model} · eval window {window}"
        axes[0].plot(group.threshold, group.mean_absolute_token_error, marker=".", markersize=3, label=label)
        axes[1].plot(group.threshold, group.no_crossing_rate, marker=".", markersize=3, label=label)
    for axis in axes:
        axis.axvline(.5, color="black", linestyle="--", linewidth=1, label="поточний поріг 0,5")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Поріг MG-score")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Середня абсолютна похибка межі, токени")
    axes[0].set_title("MAE межі SemEval")
    axes[1].set_ylabel("Частка без знайденої межі")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("No-crossing rate")
    axes[1].legend(fontsize=7)
    figure.suptitle("F09 · SemEval · trigger: 3 consecutive tokens · median smoothing=5")
    return save_figure_pair(figure, figure_root, "F09_semeval_boundary_threshold_sweep")


def plot_f10_semeval_local_errors(
    window_metrics: pd.DataFrame,
    token_metrics: pd.DataFrame,
    figure_root: Path,
) -> list[Path]:
    """Plot pure-window error fractions and token errors over thresholds by scorer."""
    configuration_columns = ["scorer_llm", "model_family", "training_window", "evaluation_window"]
    required_windows = {*configuration_columns, "gold_label", "incorrect_fraction"}
    required_tokens = {*configuration_columns, "gold_label", "threshold", "incorrect_fraction"}
    missing_windows = sorted(required_windows.difference(window_metrics.columns))
    missing_tokens = sorted(required_tokens.difference(token_metrics.columns))
    if missing_windows or missing_tokens:
        raise ValueError(
            f"F10 metrics are missing columns: windows={missing_windows}, tokens={missing_tokens}"
        )
    if window_metrics.empty or token_metrics.empty:
        raise ValueError("F10 requires non-empty window and token error metrics")
    scorers = sorted(set(window_metrics.scorer_llm.astype(str)) | set(token_metrics.scorer_llm.astype(str)))
    config_rows = pd.concat([
        window_metrics[configuration_columns], token_metrics[configuration_columns],
    ], ignore_index=True).drop_duplicates()
    config_rows["scorer_llm"] = config_rows.scorer_llm.astype(str)
    config_rows = config_rows.sort_values(configuration_columns, kind="stable").reset_index(drop=True)
    config_keys = [tuple(row) for row in config_rows[configuration_columns].itertuples(index=False, name=None)]
    window_config_keys = set(
        map(tuple, window_metrics[configuration_columns].drop_duplicates().itertuples(index=False, name=None))
    )
    token_config_keys = set(
        map(tuple, token_metrics[configuration_columns].drop_duplicates().itertuples(index=False, name=None))
    )
    palette = sns.color_palette("tab10", n_colors=max(1, len(config_rows)))
    config_colors = dict(zip(config_keys, palette))
    figure, axes = plt.subplots(
        len(scorers), 2, figsize=(17, max(4.2, 3.8 * len(scorers))),
        squeeze=False, constrained_layout=True,
    )
    for row_index, scorer in enumerate(scorers):
        window_axis, token_axis = axes[row_index]
        scorer_windows = window_metrics.loc[window_metrics.scorer_llm.astype(str).eq(scorer)]
        scorer_tokens = token_metrics.loc[token_metrics.scorer_llm.astype(str).eq(scorer)]
        scorer_configs = [
            key for key in config_keys if key[0] == scorer and key in window_config_keys | token_config_keys
        ]
        labels = [f"{family.replace('_', ' ')} {train}/{evaluation}" for _, family, train, evaluation in scorer_configs]
        x_positions = np.arange(len(scorer_configs), dtype="float64")
        bar_width = .38
        for gold_label, label, offset in ((0, "HG", -bar_width / 2), (1, "MG", bar_width / 2)):
            fractions = []
            for config in scorer_configs:
                selected = scorer_windows.loc[
                    scorer_windows.scorer_llm.astype(str).eq(config[0])
                    & scorer_windows.model_family.eq(config[1])
                    & scorer_windows.training_window.eq(config[2])
                    & scorer_windows.evaluation_window.eq(config[3])
                    & scorer_windows.gold_label.eq(gold_label)
                ]
                fractions.append(float(selected.incorrect_fraction.iloc[0]) if len(selected) == 1 else np.nan)
            window_axis.bar(x_positions + offset, fractions, width=bar_width, label=f"gold {label}")
        window_axis.set_xticks(x_positions, labels, rotation=35, ha="right")
        window_axis.set_ylim(0, 1)
        window_axis.set_ylabel("Incorrect-window fraction")
        window_axis.set_title(f"{scorer} · pure windows")
        window_axis.grid(axis="y", alpha=.2)
        window_axis.legend(fontsize=7)

        for config in scorer_configs:
            color = config_colors[config]
            config_label = f"{config[1].replace('_', ' ')} {config[2]}/{config[3]}"
            config_tokens = scorer_tokens.loc[
                scorer_tokens.model_family.eq(config[1])
                & scorer_tokens.training_window.eq(config[2])
                & scorer_tokens.evaluation_window.eq(config[3])
            ]
            for gold_label, region, linestyle in ((0, "HG", "-"), (1, "MG", "--")):
                curve = config_tokens.loc[config_tokens.gold_label.eq(gold_label)].sort_values("threshold")
                if not curve.empty:
                    token_axis.plot(
                        curve.threshold, curve.incorrect_fraction, color=color, linestyle=linestyle,
                        marker=".", markersize=2, linewidth=1.2, label=f"{config_label} · {region}",
                    )
        token_axis.axvline(.5, color="black", linestyle=":", linewidth=1, label="threshold 0.5")
        token_axis.set_xlim(0, 1)
        token_axis.set_ylim(0, 1)
        token_axis.set_xlabel("Token MG-score threshold")
        token_axis.set_ylabel("Mislabeled-token fraction")
        token_axis.set_title(f"{scorer} · token threshold sweep")
        token_axis.grid(alpha=.2)
        token_axis.legend(fontsize=6, ncol=2)
    axes[-1, 0].set_xlabel("Classifier / train-eval window")
    figure.suptitle("F10 · SemEval local HG/MG classification errors")
    return save_figure_pair(figure, figure_root, "F10_semeval_local_error_threshold_analysis")


def plot_f01_window_transfer(metrics: pd.DataFrame, figure_root: Path) -> list[Path]:
    rows = metrics.loc[metrics.block.eq("W")]
    output_paths = []
    for scorer in sorted(rows.scorer_llm.dropna().unique()):
        figure, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        image = None
        for row, family in enumerate(("logistic_regression", "random_forest")):
            for column, regime in enumerate(("Kaggle+MAGE+Outfox", "Kaggle+MAGE+Outfox+BBC")):
                axis = axes[row, column]
                matrix = window_transfer_matrix(
                    rows, scorer_llm=scorer, model_family=family, evaluation_regime=regime,
                )
                values = matrix.to_numpy(dtype="float64")
                image = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap="viridis")
                for y in range(2):
                    for x in range(2):
                        value = values[y, x]
                        axis.text(x, y, "NA" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", color="white" if np.isfinite(value) and value < .6 else "black")
                axis.set_xticks([0, 1], labels=["32", "64"])
                axis.set_yticks([0, 1], labels=["32", "64"])
                axis.set_xlabel("Eval window")
                axis.set_ylabel("Train window")
                axis.set_title(f"{family.replace('_', ' ')} · {regime}")
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Window macro-F1", shrink=.8)
        figure.suptitle(f"F01 · {scorer} · window transfer")
        output_paths.extend(save_figure_pair(figure, figure_root, f"F01_window_transfer__{scorer}"))
    return output_paths


def plot_f02_confusion_matrices(metrics: pd.DataFrame, figure_root: Path) -> list[Path]:
    rows = metrics.loc[
        metrics.block.eq("W")
        & metrics.scorer_llm.eq("gemma_3_270m_it")
        & metrics.training_window.eq(metrics.evaluation_window)
    ]
    output_paths = []
    for window in (32, 64):
        figure, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        window_rows = rows.loc[rows.training_window.eq(window)]
        for row, regime in enumerate(("Kaggle+MAGE+Outfox", "Kaggle+MAGE+Outfox+BBC")):
            for column, family in enumerate(("logistic_regression", "random_forest")):
                axis = axes[row, column]
                selected = window_rows.loc[
                    window_rows.evaluation_regime.eq(regime) & window_rows.model_family.eq(family)
                ]
                axis.set_title(f"{regime} · {family.replace('_', ' ')}")
                if selected.empty:
                    axis.text(.5, .5, "NA", ha="center", va="center", transform=axis.transAxes)
                    axis.set_axis_off()
                    continue
                if len(selected) != 1:
                    raise ValueError("F02 expected exactly one row per configuration")
                result = selected.iloc[0]
                normalized = normalize_confusion_counts(
                    tn=int(result.tn), fp=int(result.fp), fn=int(result.fn), tp=int(result.tp),
                )
                sns.heatmap(normalized, ax=axis, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False,
                            xticklabels=["HG", "MG"], yticklabels=["HG", "MG"], linewidths=.5)
                counts = ((int(result.tn), int(result.fp)), (int(result.fn), int(result.tp)))
                for y in range(2):
                    for x in range(2):
                        axis.text(x + .5, y + .5, f"{counts[y][x]}\n{normalized[y, x]:.0%}", ha="center", va="center")
                axis.set_xlabel("Прогнозований клас")
                axis.set_ylabel("Справжній клас")
                axis.set_title(f"{axis.get_title()}\nN_HG={int(result.hg_window_count)}, N_MG={int(result.mg_window_count)}")
        figure.suptitle(f"F02 · Gemma 3 270M · {window}/{window}")
        output_paths.extend(save_figure_pair(figure, figure_root, f"F02_confusion__window_{window}"))
    return output_paths


def plot_f03_document_tradeoff(metrics: pd.DataFrame, figure_root: Path) -> list[Path]:
    rows = metrics.loc[
        metrics.evaluation_regime.eq("Kaggle+MAGE+Outfox+BBC")
        & metrics.model.eq("mean_score_baseline")
    ]
    if rows.empty:
        return []
    output_paths = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True, sharex=True, sharey=True)
    for axis, family in zip(axes, ("logistic_regression", "random_forest")):
        current = rows.loc[rows.model_family.eq(family)]
        for scorer, group in current.groupby("scorer_llm", sort=True):
            axis.scatter(group.false_positive_rate, group.recall, label=scorer, s=40)
            for _, item in group.iterrows():
                axis.annotate(f"{item.training_window}/{item.evaluation_window}",
                              (item.false_positive_rate, item.recall), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set_title(family.replace("_", " "))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Документний FPR")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Повнота виявлення MG")
    if not rows.empty:
        axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("F03 · BBC-only HG · Mixed+BBC evaluation")
    return save_figure_pair(figure, figure_root, "F03_document_fpr_vs_mg_recall")


def plot_f04_document_length(
    metrics: pd.DataFrame,
    figure_root: Path,
    *,
    minimum_class_support: int = 20,
) -> list[Path]:
    primary = metrics.loc[metrics.scorer_llm.isin(("gemma_3_270m_it", "littlelamb"))]
    output_paths = []
    for scorer in ("gemma_3_270m_it", "littlelamb"):
        for family in ("logistic_regression", "random_forest"):
            for regime in ("Kaggle+MAGE+Outfox", "Kaggle+MAGE+Outfox+BBC"):
                rows = primary.loc[
                    primary.scorer_llm.eq(scorer)
                    & primary.model_family.eq(family)
                    & primary.evaluation_regime.eq(regime)
                ]
                rows = eligible_length_metrics(rows, minimum_class_support=minimum_class_support)
                if rows.empty:
                    continue
                bins = [label for label in ("<=250", "251-500", "501-1000", ">1000") if label in rows.word_length_bin.astype(str).values]
                methods = sorted(rows.method_variant.dropna().unique())
                values = rows.pivot(index="word_length_bin", columns="method_variant", values="macro_f1").reindex(index=bins, columns=methods)
                figure, axis = plt.subplots(figsize=(max(8, .9 * len(methods) + 4), 4.8), constrained_layout=True)
                x = np.arange(len(bins), dtype=float)
                width = .8 / max(len(methods), 1)
                for index, method in enumerate(methods):
                    axis.bar(x - .4 + width * (index + .5), values[method], width, label=method)
                axis.set_xticks(x, labels=bins)
                axis.set_xlabel("Довжина документа, слів")
                axis.set_ylabel("Document macro-F1")
                axis.set_ylim(0, 1)
                axis.set_title(f"F04 · {scorer} · {family} · {regime} · N/class ≥ {minimum_class_support}")
                axis.legend(fontsize=7, ncol=2)
                output_paths.extend(save_figure_pair(figure, figure_root, f"F04_length__{scorer}__{family}__{regime.replace('+', '_')}"))
    return output_paths
