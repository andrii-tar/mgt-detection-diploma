# Features And Metrics

## Feature Philosophy

Features describe how a specific scorer model evaluates a text. They are not generator fingerprints. Their distributions can move with language, domain, topic, text length, decoding settings, edits, and scorer revision. Always inspect feature distributions by label and domain before fitting a classifier.

Let an input have `T` valid next-token positions. At position `t`, `l_t` is the scorer log probability of the observed token and `p_t = exp(l_t)` is its probability.

## Core Likelihood Features

### Perplexity

\[
\operatorname{PPL}(x)=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}l_t\right)
\]

Lower perplexity means the scorer assigns higher average likelihood to the text. It does **not** mean a text is machine-generated. Perplexity is especially sensitive to the scorer, tokenizer, text length, and domain.

### Prompt Log-Probability Mean And Standard Deviation

\[
\mu_{\mathrm{prompt}}=\frac{1}{T}\sum_{t=1}^{T} l_t
\]

\[
\sigma_{\mathrm{prompt}}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(l_t-\mu_{\mathrm{prompt}})^2}
\]

The historical code uses population standard deviation (`ddof=0`) for `prompt_sigma_score`. Keep that choice explicit because it differs from the common sample standard deviation.

### Likelihood Feature Reference Implementation

```python
import numpy as np


def likelihood_features(observed_logprobs: np.ndarray) -> dict[str, float]:
    """Compute features from valid autoregressive next-token log probabilities."""
    logprobs = np.asarray(observed_logprobs, dtype=float)
    if logprobs.ndim != 1 or len(logprobs) == 0:
        raise ValueError("expected at least one valid token log probability")
    if not np.isfinite(logprobs).all():
        raise ValueError("log probabilities must be finite")

    return {
        "perplexity": float(np.exp(-logprobs.mean())),
        "prompt_mu_score": float(logprobs.mean()),
        "prompt_sigma_score": float(logprobs.std(ddof=0)),
        "num_tokens": int(logprobs.size),
    }
```

## Monte-Carlo Sampling Discrepancy

The repository's top-`k` path computes a custom normalized discrepancy score. Its purpose is to compare the observed token probability pattern with probability patterns obtained by sampling alternatives from the scorer distribution.

### Inputs

For each position, create a vector of candidate log probabilities. In top-`k` mode this is the returned candidate set; in full-vocabulary mode it can be the model logits/log probabilities across all tokens. The observed token must be identified within that candidate row.

Normalize candidate log probabilities within each row:

\[
q_t(i)=\frac{\exp(a_{t,i})}{\sum_{r=1}^{K}\exp(a_{t,r})}
\]

where `a[t, i]` is the candidate log probability and `K` is the candidate count. For top-`k` data this is a **truncated, renormalized** distribution, not the original full-vocabulary distribution.

Let `o_t` be the index of the observed token in the candidate row. Define its normalized probability and its sequence mean:

\[
p_{\mathrm{obs}}=\frac{1}{T}\sum_{t=1}^{T}q_t(o_t)
\]

Draw `M` independently sampled candidate indices at each token position. For sample `j`:

\[
s_j=\frac{1}{T}\sum_{t=1}^{T}q_t(z_{t,j}), \qquad z_{t,j}\sim q_t
\]

Then:

\[
\mu_{\mathrm{sample}}=\frac{1}{M}\sum_{j=1}^{M}s_j
\]

\[
\sigma_{\mathrm{sample}}=\sqrt{\frac{1}{M}\sum_{j=1}^{M}(s_j-\mu_{\mathrm{sample}})^2}
\]

\[
d_{\mathrm{score}}=\frac{p_{\mathrm{obs}}-\mu_{\mathrm{sample}}}{\sigma_{\mathrm{sample}}+\epsilon}
\]

The historical implementation uses 1,000 samples and `epsilon = 1e-10`. It stores `d_score`, `d_mu_score`, and `d_sigma_score`.

### Sampling Discrepancy Reference Implementation

```python
import numpy as np


def _row_softmax(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - log_values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def sampling_discrepancy(
    candidate_logprobs: np.ndarray,
    observed_candidate_index: np.ndarray,
    sample_count: int = 1_000,
    seed: int = 42,
    epsilon: float = 1e-10,
) -> dict[str, float]:
    """Reproduce the top-k normalized-probability discrepancy definition.

    candidate_logprobs has shape (tokens, candidates). observed_candidate_index
    identifies the actual token in each candidate row. It is invalid if an actual
    token is absent from a top-k row; exclude or handle that token explicitly.
    """
    candidates = np.asarray(candidate_logprobs, dtype=float)
    observed_index = np.asarray(observed_candidate_index, dtype=int)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidate_logprobs must have shape (tokens, candidates)")
    if observed_index.shape != (candidates.shape[0],):
        raise ValueError("one observed index is required per token")
    if (observed_index < 0).any() or (observed_index >= candidates.shape[1]).any():
        raise ValueError("observed token absent from one or more candidate rows")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate log probabilities must be finite")

    probabilities = _row_softmax(candidates)
    token_index = np.arange(candidates.shape[0])
    observed_mean = probabilities[token_index, observed_index].mean()

    rng = np.random.default_rng(seed)
    sampled_probabilities = np.empty((candidates.shape[0], sample_count))
    for token_position, distribution in enumerate(probabilities):
        sampled_index = rng.choice(len(distribution), size=sample_count, p=distribution)
        sampled_probabilities[token_position] = distribution[sampled_index]

    sampled_sequence_means = sampled_probabilities.mean(axis=0)
    sample_mu = sampled_sequence_means.mean()
    sample_sigma = sampled_sequence_means.std(ddof=0)
    return {
        "d_score": float((observed_mean - sample_mu) / (sample_sigma + epsilon)),
        "d_mu_score": float(sample_mu),
        "d_sigma_score": float(sample_sigma),
    }
```

### Interpretation

- `d_score` is a standardized relative position, not a calibrated probability that text is MG.
- `d_mu_score` and `d_sigma_score` depend on vocabulary support and sample count; compare only matching scorer configurations.
- A low sample standard deviation can make the standardized score numerically unstable. Log and inspect these cases.
- Top-`k` discrepancy should be named as an approximation. It can change if the endpoint changes its candidate ranking or omits the observed token.

## Full-Vocabulary Log-Likelihood Discrepancy

The local `fdgpt_` path follows a related but distinct definition based on sampled **log likelihoods**. It samples candidate token IDs from the complete scorer distribution and compares their mean log likelihood with the observed sequence mean:

\[
d_{\mathrm{full}}=\frac{\ell(x)-\mu(\ell(\tilde{x}))}{\sigma(\ell(\tilde{x}))+\epsilon}
\]

Do not pool this feature with the top-`k` probability discrepancy under a shared unqualified name. Store `vocabulary_mode` as `full` or `top_k`, and retain the method version.

## Text Controls

The source notebooks inspect character, word, and sentence counts. These controls are useful for matching datasets and detecting shortcuts.

```python
import re


def text_controls(text: str) -> dict[str, int]:
    # Replace this simple sentence splitter with a language-appropriate segmenter.
    nonempty_sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return {
        "text_len_symbols": len(text),
        "text_len_words": len(text.split()),
        "text_len_sentences": len(nonempty_sentences),
    }
```

Use a language-aware tokenizer/sentence segmenter for final work. A rule such as “more than 15 sentences” should be treated as a declared eligibility criterion, not quietly applied after data inspection.

## Outliers

The repository removes outliers sequentially per feature and label using an absolute z-score threshold of 3. This is an exploratory visualization aid, not a safe default for model evaluation. Sequential removal changes the sample after each feature and can leak test-distribution information if performed before splitting.

If an outlier policy is justified, fit it only on the training partition and apply its learned bounds to validation/test rows. First report results without removal; then report the policy, retained counts, and sensitivity analysis.

```python
def fit_zscore_bounds(frame, feature_columns, threshold=3.0):
    means = frame[feature_columns].mean()
    stds = frame[feature_columns].std(ddof=0).replace(0, np.nan)
    return means, stds, threshold


def apply_zscore_bounds(frame, means, stds, threshold):
    z_scores = (frame[means.index] - means) / stds
    return frame.loc[z_scores.abs().le(threshold).all(axis=1)].copy()
```

## Feature Quality Checks

Before model training, assert all of the following:

- no feature is derived from a label;
- each `text_id` belongs to one split only;
- feature rows are unique by `(text_id, scorer_id, feature_version)`;
- all selected scalar features are finite;
- labels have the expected class mapping;
- scorer configuration is constant within the feature table;
- class balance, length, and scoring-failure rates are reported by dataset;
- no text-dependent threshold was selected using the final test set.
