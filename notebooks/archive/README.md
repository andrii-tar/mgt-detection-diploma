# Archived Notebooks

This directory contains superseded training and evaluation notebooks retained
for historical reference. They are not part of the active notebook workflow.

| Notebook | Historical purpose | Why inactive | Replacement |
|---|---|---|---|
| `bayesian_model_comparison.ipynb` | Compare Bayesian and deterministic local-evidence classifiers. | Superseded by the active feature-vector workflow. | `prepare_and_train_classifiers.ipynb` |
| `cross_dataset_classification_poc.ipynb` | Prototype cross-dataset whole-text and local-evidence matrices. | Replaced by resumable classification notebooks. | `prepare_and_train_classifiers.ipynb` and `evaluate_classifiers.ipynb` |
| `gemma_cross_dataset_boundary_validation.ipynb` | Validate Gemma local-evidence boundaries on SemEval text. | Depends on the retired one-off experiment layout. | `evaluate_classifiers.ipynb` |
| `local_evidence_classification.ipynb` | Train and score local-evidence window classifiers. | Replaced by the feature-vector classifier pipeline. | `prepare_and_train_classifiers.ipynb` |
| `local_evidence_segmentation.ipynb` | Inspect token-level local-evidence segmentation. | Tied to artifacts from the retired local-evidence workflow. | `evaluate_classifiers.ipynb` |
| `text_classification.ipynb` | Run the original scalar baseline classification experiment. | Superseded by prepared feature-vector artifacts. | `prepare_and_train_classifiers.ipynb` |
