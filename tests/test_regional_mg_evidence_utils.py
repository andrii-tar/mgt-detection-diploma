import numpy as np
import pandas as pd

from notebooks.utils.regional_mg_evidence_utils import (
    build_multiscale_token_evidence,
    extract_hysteresis_regions,
    fit_probability_calibrator,
    summarize_window_contributions,
)


def test_multiscale_projection_preserves_mean_max_support_and_agreement():
    windows = pd.DataFrame(
        [
            {"start_token": 0, "stop_token": 4, "window_length": 4, "mg_score": 0.8},
            {"start_token": 2, "stop_token": 6, "window_length": 4, "mg_score": 0.6},
            {"start_token": 0, "stop_token": 6, "window_length": 6, "mg_score": 0.9},
        ]
    )

    evidence = build_multiscale_token_evidence(
        windows,
        token_count=6,
        window_threshold=0.7,
    )

    assert evidence["token_position"].tolist() == list(range(6))
    assert evidence.loc[2, "mean_score"] == np.mean([0.8, 0.6, 0.9])
    assert evidence.loc[2, "max_score"] == 0.9
    assert evidence.loc[2, "support_count"] == 3
    assert evidence.loc[2, "length_agreement"] == 1.0
    assert evidence.loc[5, "support_count"] == 2
    assert not np.isnan(evidence.loc[5, "score_std"])


def test_hysteresis_regions_bridge_low_score_gaps_and_drop_short_regions():
    evidence = pd.DataFrame(
        {
            "token_position": range(10),
            "robust_score": [0.2, 0.8, 0.7, 0.52, 0.6, 0.3, 0.76, 0.4, 0.3, 0.2],
            "mean_score": [0.2, 0.8, 0.7, 0.52, 0.6, 0.3, 0.76, 0.4, 0.3, 0.2],
            "max_score": [0.2, 0.8, 0.7, 0.52, 0.6, 0.3, 0.76, 0.4, 0.3, 0.2],
        }
    )

    regions = extract_hysteresis_regions(
        evidence,
        score_column="robust_score",
        start_threshold=0.75,
        continue_threshold=0.5,
        minimum_length=3,
        merge_gap=1,
    )

    assert regions[["start_token", "stop_token"]].to_dict("records") == [
        {"start_token": 1, "stop_token": 7}
    ]
    assert regions.loc[0, "token_count"] == 6
    assert regions.loc[0, "max_score"] == 0.8


def test_window_contribution_summary_ranks_signed_feature_evidence():
    scored = pd.DataFrame(
        {
            "contribution__observed_logprob__mean": [0.4, 0.2],
            "contribution__mean_candidate_similarity__max": [-0.1, 0.3],
            "mg_score": [0.8, 0.9],
        }
    )

    summary = summarize_window_contributions(scored, top_n=2)

    assert summary["feature"].tolist() == [
        "observed_logprob__mean",
        "mean_candidate_similarity__max",
    ]
    assert np.allclose(summary["mean_signed_contribution"], [0.3, 0.1])


def test_probability_calibrator_returns_monotonic_validation_probabilities():
    calibrator = fit_probability_calibrator(
        raw_scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
    )

    calibrated = calibrator.predict([0.15, 0.85])

    assert calibrated[0] < 0.5
    assert calibrated[1] > 0.5
    assert np.all((calibrated >= 0) & (calibrated <= 1))
