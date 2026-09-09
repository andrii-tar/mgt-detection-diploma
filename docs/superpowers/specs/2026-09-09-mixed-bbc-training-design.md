# Mixed+BBC Training Design

## Goal

Add a separate `Mixed+BBC` training regime to the cross-dataset classification
experiment while preserving the existing `Mixed` baseline.

## Split Protocol

BBC is an HG-only source. Its available HG parent cohort is partitioned
deterministically into train, validation, and test using the existing seed and
cap settings.

- `Mixed`: Kaggle + MAGE + Outfox.
- `Mixed+BBC`: Kaggle + MAGE + Outfox + BBC.
- `Mixed+BBC` fitting uses only the four source train partitions.
- Source validation uses Kaggle, MAGE, Outfox, and BBC validation partitions.
- BBC test remains a final HG-only holdout and is never used for fitting or
  threshold selection.
- SemEval train and dev are unseen external validation datasets and are not
  used for fitting, threshold selection, or model selection.

## Classifier Coverage

The new regime applies to both existing classifier families:

- whole-text scalar logistic regression;
- local-evidence logistic regression with configured windows and aggregations.

BBC contributes only label 0 examples. Training remains binary because the
other three sources contribute MG examples. BBC evaluation continues to report
HG false-positive behavior; binary ROC-AUC, PR-AUC, and balanced accuracy are
undefined for BBC-only targets.

## Implementation

Extend the notebook's training and evaluation regime mappings with
`Mixed+BBC`. Keep the existing `Mixed` mapping unchanged. Ensure the source
loader's explicit `SOURCE_SPLITS` configuration continues to request only HG
for BBC. Add BBC to the new regime's training and validation cohort mappings,
while retaining separate evaluation entries for BBC test and SemEval.

The notebook displays and saved result frames must preserve `training_regime`
so `Mixed` and `Mixed+BBC` can be compared directly. Existing source coverage,
BBC false-positive summaries, and SemEval outputs remain available.

## Validation

- Add notebook contract tests for the new regime and its source composition.
- Add utility-level tests proving a mixed binary training cohort can include an
  HG-only BBC source without allowing BBC test rows into training.
- Run focused and full pytest suites.
- Keep the notebook output-free after source edits.
- Do not run the full notebook or large processing jobs during automated tests.
