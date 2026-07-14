# Evaluation And Diagnostics

## Evaluation Question

The question is not “can a classifier separate these rows?” It is:

> For a specified scorer configuration and declared target population, how accurately can the detector distinguish HG from MG text that it did not influence during design, tuning, preprocessing, or threshold selection?

Use `0 = HG` and `1 = MG` consistently. Always state which class is considered positive in ROC, PR, and threshold reporting.

## Split Design

### Required Partitions

| Partition | Purpose | May influence model choice? |
| --- | --- | --- |
| Training | Fit transformations and models. | Yes. |
| Validation/CV | Select features, hyperparameters, and operating threshold. | Yes. |
| In-domain test | Final estimate within the same source/domain. | No. |
| External test | Estimate portability to another corpus/domain/generator. | No. |

The repository compares Kaggle and OUTFOX datasets and mixed datasets. Those experiments demonstrate why a same-source score must not be interpreted as a universal detector score: a change of source can substantially change class performance.

### Reference Split Code

```python
from sklearn.model_selection import train_test_split


def stratified_split(frame, test_size=0.2, seed=42):
    return train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
```

The helper in `scripts/tools.py` historically uses a stratified split with `random_state=42` and a configurable test fraction. In a new project, use grouped splitting if texts share prompts, source documents, paraphrases, or generation sessions.

## Candidate Classifiers

The notebooks compare Random Forest, RBF SVM, linear SVM, and XGBoost. Treat them as baselines with different assumptions:

| Model | Strength | Requirement/Risk |
| --- | --- | --- |
| Random Forest | Captures nonlinear interactions; little feature scaling needed. | Can overfit source artifacts; tune depth and leaf size. |
| RBF SVM | Strong compact baseline for nonlinear feature spaces. | Scale features inside the pipeline; tune `C` and `gamma` only with CV. |
| Linear SVM | Simple, fast reference. | May underfit nonlinear boundaries; still requires scaling. |
| XGBoost | Flexible nonlinear learner. | High overfitting risk on small datasets; tune conservatively and fix seeds. |

### Reference Pipelines

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def candidate_models(seed=42):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "svm_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
    }
```

Add XGBoost only when it is installed and its parameters, version, objective, seed, and early-stopping split are all recorded. Do not select a model because it is best on a single final test run.

## Cross-Validated Model Selection

```python
from sklearn.model_selection import StratifiedKFold, cross_validate


def cross_validate_model(model, X, y, seed=42):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_validate(
        model,
        X,
        y,
        cv=folds,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
            "mg_f1": "f1",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
    )
```

Cross-validation estimates variation within the development corpus. It does not replace a held-out test or external-domain evaluation. All preprocessing, including scaling, imputation, feature selection, and outlier treatment, belongs in the model pipeline or within CV folds.

## Metrics

For class `c`:

\[
\operatorname{precision}_c=\frac{TP_c}{TP_c+FP_c},
\qquad
\operatorname{recall}_c=\frac{TP_c}{TP_c+FN_c}
\]

\[
F1_c=\frac{2\cdot\operatorname{precision}_c\cdot\operatorname{recall}_c}{\operatorname{precision}_c+\operatorname{recall}_c}
\]

Macro F1 gives each class equal weight:

\[
F1_{\mathrm{macro}}=\frac{F1_{HG}+F1_{MG}}{2}
\]

Weighted F1 weights class F1 values by support. Accuracy is the fraction of all correct labels. Macro F1 is usually safer than accuracy when labels are imbalanced or the cost of HG and MG errors differs.

### Required Report

```python
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_probability):
    return {
        "report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["HG", "MG"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
    }
```

Report support with every per-class metric. A high weighted score can hide poor HG performance when MG dominates, and vice versa.

## Thresholds And Calibration

The notebooks explore hand-selected thresholds for individual probability-like values. A raw likelihood or discrepancy threshold is scorer- and corpus-specific; it is not a general MG probability.

Preferred process:

1. Fit a model on training data.
2. Predict probabilities on a validation set only.
3. Choose a threshold based on a declared objective, such as minimum HG recall, maximum macro F1, or a false-positive-rate limit.
4. Freeze the threshold.
5. Evaluate it once on each test set.

```python
import numpy as np
from sklearn.metrics import f1_score


def select_macro_f1_threshold(y_validation, mg_probability):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_validation, mg_probability >= value, average="macro") for value in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])
```

If predictions will be shown to people, inspect calibration with a reliability plot or `CalibratedClassifierCV`. A probability threshold is meaningful only when its calibration and operating cost are documented.

## Diagnostics

The visual notebook uses grouped bar charts, faceted charts, heatmaps, box plots, and classification-report aggregation. Use the following diagnostics to understand a detector rather than to inflate its headline score.

| Diagnostic | Question answered | Correct use |
| --- | --- | --- |
| HG/MG feature histograms | Are feature distributions separated? | Use training data or label descriptive only; do not tune on final test. |
| Mean/sigma scatter | Does a scorer configuration separate distributions? | Compare only matching feature definitions. |
| `d_score` vs prompt mean scatter | Are classes or domains forming clusters? | Color by both origin and dataset in separate plots. |
| Confusion matrix | Which error type is dominant? | Report absolute counts and normalized rates. |
| Train/test heatmap | Does performance change by train/test source? | Keep model and feature config fixed across cells. |
| ROC and PR curves | How does performance vary with threshold? | Use test scores only after threshold is frozen. |
| Calibration curve | Do predicted probabilities match observed frequency? | Necessary before probability-based decisions. |

## Cross-Domain Protocol

For every selected scorer-model pair, create a matrix where rows are train domains and columns are test domains. Use a train domain's validation split to choose all settings, then evaluate unchanged on each test domain.

```text
train: Kaggle-like corpus
  validate: held-out Kaggle-like corpus
  test: Kaggle-like corpus, OUTFOX-like corpus, mixed corpus

train: OUTFOX-like corpus
  validate: held-out OUTFOX-like corpus
  test: Kaggle-like corpus, OUTFOX-like corpus, mixed corpus
```

For each cell report support, HG/MG precision-recall-F1, macro F1, ROC-AUC, PR-AUC, and selected threshold. Do not average across cells unless domains have a defensible, declared weighting scheme.

## Do Not Reproduce These Historical Shortcuts

- Do not filter outliers using the entire labeled dataset before splitting.
- Do not choose thresholds after viewing final test reports.
- Do not rely on same-dataset validation as a portability claim.
- Do not average results from incompatible `fdgpt_` and top-`k` scorer modes.
- Do not use a deprecated `DataFrame.append` result-aggregation pattern; accumulate dictionaries in a list and create one DataFrame.
- Do not interpret a custom-text report with extremely small or highly imbalanced support as a general model evaluation.
