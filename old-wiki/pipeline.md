# Processing Pipeline

## Purpose

Transform labeled texts into one reproducible feature table per scorer configuration. A scorer configuration is part of the data identity: changing the model revision, tokenizer, vocabulary mode, `top_k`, or normalization makes features incomparable with prior runs.

## Inputs And Outputs

```text
Text table
  + scorer configuration
  -> token likelihood evidence
  -> derived scorer features
  -> feature table for model training and evaluation
```

### Input Text Table

At minimum, retain `text_id`, raw scoring text, binary label, and `dataset_id`. Store the text normalization version with the text or scoring run. A detector can accidentally learn formatting differences, source templates, or length distributions instead of origin; provenance is required to diagnose this.

### Token Evidence Contract

Use a scorer adapter that returns evidence for a single text. Keep raw evidence separately from derived scalar metrics so formulas can be corrected and recomputed later.

```python
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TokenEvidence:
    text_id: str
    scorer_id: str
    observed_logprobs: np.ndarray       # shape: (n_tokens,)
    candidate_logprobs: np.ndarray | None  # shape: (n_tokens, k), or None
    token_ids: list[int] | None
    scorer_metadata: dict[str, Any]


def score_text(text_id: str, text: str, scorer_id: str) -> TokenEvidence:
    """Implement this adapter for the local LM or serving endpoint in a new project."""
    raise NotImplementedError
```

The first token has no preceding context in an autoregressive model. Exclude it from all calculations or use an explicit validity mask. Do not use a fabricated sentinel value in new work.

### Feature Table Contract

One row represents one `(text_id, scorer_id)` pair.

```text
text_id, dataset_id, label, scorer_id,
perplexity, prompt_mu_score, prompt_sigma_score,
d_score, d_mu_score, d_sigma_score, num_tokens,
text_len_symbols, text_len_words, text_len_sentences,
scorer_revision, vocabulary_mode, top_k, sample_count
```

Never merge features from different scorer configurations as though they were interchangeable. If several scorers will be combined, include their configuration identities and fit the combination only inside training folds.

## Historical Storage Model

The repository stores original records in a `texts` table and scorer outputs in `processed_texts`:

```text
texts
  id, content, generated, origin, dataset, dataset_id

processed_texts
  id, text_id, model, perplexity, num_tokens,
  prompt_logprobs, log_probs, raw response,
  d_score, d_mu_score, d_sigma_score,
  prompt_mu_score, prompt_sigma_score
```

This schema is useful because raw arrays remain available after the definition of a scalar feature changes. In a new project, store arrays in object storage, Parquet-compatible nested columns, or a database only after considering volume, retention, and access-control requirements.

## Scorer Modes

### Mode A: Echoed Prompt Scores With Top-K Alternatives

The historical API path sends an OpenAI-compatible completion request with the prompt echoed and zero generated tokens. It requests prompt token log probabilities and the top alternatives at each position.

```json
{
  "model": "scorer-model-id",
  "prompt": ["text to score"],
  "echo": true,
  "max_tokens": 0,
  "temperature": 0,
  "logprobs": 20,
  "prompt_logprobs": 20
}
```

This produces an approximation because all probability operations are constrained to candidates returned by the API. The historical implementation labels this path `top 20`.

Requirements for a valid adapter:

- Align observed token log probabilities and candidate rows to the same token positions.
- Preserve candidate rank/order and whether the observed token is present.
- Reject or explicitly flag rows where the observed token is absent from the candidate set.
- Record endpoint version, tokenizer/model revision, `top_k`, and serving settings.
- Never assume log probabilities from different tokenizers are directly comparable.

### Mode B: Local Full-Vocabulary Fast-DetectGPT-Style Scoring

The historical `fdgpt_` path loads a Hugging Face causal LM locally, computes logits for the input, and samples from the entire vocabulary. It compares the log likelihood of observed next tokens with log likelihoods of sampled alternatives. The `fdgpt_` prefix is a storage convention identifying this path, not a different ground-truth label.

For a full vocabulary distribution at position `t`, let `q_t(v)` be the scorer probability of token `v`, `x_t` be the observed token, and `s_{t,j}` be sampled alternative token `j`. Define average observed and sampled log likelihoods:

\[
\ell(x)=\frac{1}{T}\sum_{t=1}^{T}\log q_t(x_t),
\qquad
\ell_j=\frac{1}{T}\sum_{t=1}^{T}\log q_t(s_{t,j})
\]

The discrepancy is:

\[
d_{\mathrm{full}}=\frac{\ell(x)-\operatorname{mean}_j(\ell_j)}{\operatorname{std}_j(\ell_j)+\epsilon}
\]

This is closer to a standard Fast-DetectGPT-style criterion than a top-`k` endpoint approximation, but it still requires careful model, tokenizer, sampling, and domain validation.

## End-To-End Procedure

### 1. Freeze And Split Text IDs First

Deduplicate and group related texts before splitting. Examples derived from the same prompt, an original text and its paraphrase, or chunked sections of one document must not land in different splits.

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(
    splitter.split(texts, y=texts["label"], groups=texts["split_group"])
)
train_texts = texts.iloc[train_index].copy()
test_texts = texts.iloc[test_index].copy()
```

Use `StratifiedGroupKFold` for grouped cross-validation when each group has enough examples. If there are no related groups, use a stratified split.

### 2. Score Every Split Without Looking At Labels

Feature extraction must be label-blind. Do not choose the scorer, its `top_k`, length filter, or outlier rule after inspecting held-out labels.

```python
def score_frame(text_frame, scorer_id):
    evidence = []
    for row in text_frame.itertuples(index=False):
        evidence.append(score_text(row.text_id, row.text, scorer_id))
    return evidence
```

### 3. Derive Scalar Features

Convert token evidence into features with the pure functions in [Features](features.md). Persist the feature schema version and every scorer setting alongside the resulting table.

### 4. Check Coverage Across Scorers

When comparing scorers, use only texts successfully scored by every scorer in that comparison. Otherwise class balance and sample composition differ by scorer, invalidating a direct comparison.

```python
def shared_text_ids(features, scorer_ids):
    scored = features.loc[features["scorer_id"].isin(scorer_ids)]
    counts = scored.groupby("text_id")["scorer_id"].nunique()
    return counts[counts.eq(len(scorer_ids))].index
```

### 5. Apply Length Policy Deliberately

The notebook filters texts to more than 15 sentences in some experiments. Length filtering can stabilize token-level metrics, but it changes the target population. Define the policy before evaluation and report exclusion rates by class and domain.

### 6. Fit And Evaluate

Train on feature rows from the training IDs only. Apply scalers, outlier policy, feature selection, hyperparameter selection, and threshold calibration inside the training/CV process. See [Evaluation](evaluation.md).

## Failure Handling

| Failure | Required behavior |
| --- | --- |
| Endpoint/model request fails | Record the failure with `text_id` and scorer metadata; do not silently substitute a score. |
| Empty or one-token input | Mark invalid for autoregressive likelihood features and report the exclusion. |
| Candidate row lacks observed token | Mark discrepancy as unavailable for top-`k` mode; do not compare unrelated candidate probabilities. |
| NaN/Inf derived feature | Reject the row from that run and include it in the data-quality report. |
| Tokenizer/model revision changes | Start a new scorer configuration and recompute features. |
| Class-specific failure rate | Stop and investigate; the failure can itself create a spurious detector. |

## Artifacts To Preserve

For every run preserve: text-ID manifest, split manifest, source data version, scorer config, raw evidence location/checksum, feature table, training config, fitted pipeline, threshold calibration data, predictions, and metric report. These artifacts make an apparent detection result auditable and reproducible.
