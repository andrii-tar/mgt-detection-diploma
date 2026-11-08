from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from notebooks.utils.token_feature_scatter_utils import (
    TOKEN_FEATURE_COLUMNS,
    build_token_feature_frame,
    load_balanced_token_feature_frame,
    load_balanced_token_feature_group_frame,
    plot_all_feature_pairs,
)


def test_build_token_feature_frame_exposes_four_token_features() -> None:
    stage1 = pd.Series(
        {
            "id": 7,
            "tokens": [
                {
                    "token": "A",
                    "logprob": -0.5,
                    "top_logprobs": [{"logprob": -0.6}, {"logprob": -1.2}],
                },
                {
                    "token": "B",
                    "logprob": -1.0,
                    "top_logprobs": [{"logprob": -1.1}],
                },
            ],
        }
    )
    stage2 = pd.Series(
        {
            "id": 7,
            "token_similarities": [
                {"position": 0, "top_logprobs": [{"similarity_to_gt": 0.8}]},
                {"position": 1, "top_logprobs": [{"similarity_to_gt": 0.2}]},
            ],
        }
    )

    result = build_token_feature_frame(stage1, stage2, split="HG", text_index=3)

    assert list(TOKEN_FEATURE_COLUMNS) == [
        "observed_logprob",
        "mean_candidate_logprob_delta",
        "mean_candidate_similarity",
        "top20_coverage",
    ]
    assert result["text_index"].unique().tolist() == [3]
    assert result["split"].unique().tolist() == ["HG"]
    assert np.allclose(result["mean_candidate_similarity"], [0.8, 0.2])
    assert np.allclose(result["top20_coverage"], [np.exp(-0.6) + np.exp(-1.2), np.exp(-1.1)])


def test_plot_all_feature_pairs_returns_six_axes() -> None:
    frame = pd.DataFrame(
        {
            "split": ["HG", "MG"] * 3,
            **{feature: np.arange(6, dtype=float) for feature in TOKEN_FEATURE_COLUMNS},
        }
    )

    figure, axes = plot_all_feature_pairs(frame)

    assert len(axes) == 6
    assert len(figure.axes) == 6


def test_loader_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        load_balanced_token_feature_frame(
            None,  # type: ignore[arg-type]
            "dataset",
            "model",
            batch_size=0,
        )


def test_group_loader_rejects_empty_dataset_group() -> None:
    with pytest.raises(ValueError, match="datasets"):
        load_balanced_token_feature_group_frame(None, (), "model")  # type: ignore[arg-type]
