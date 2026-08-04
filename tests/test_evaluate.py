"""Unit tests for src/evaluate.py ranking and fairness metrics, using
hand-checkable synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_by_subgroup, evaluate_ranking


def test_perfect_ranking_gives_precision_and_recall_of_one():
    # 10 schools, the top-scored 3 are exactly the 3 true high-risk schools.
    y_pred = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    high_risk_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    result = evaluate_ranking(y_pred, high_risk_true, k_fractions=(0.3,))
    top_30 = result["by_k"]["top_30pct"]
    assert top_30["precision"] == 1.0
    assert top_30["recall"] == 1.0
    assert top_30["lift"] == pytest.approx(1.0 / 0.3)


def test_random_ranking_gives_lift_near_one():
    rng = np.random.default_rng(0)
    n = 2000
    high_risk_true = (rng.uniform(size=n) < 0.2).astype(int)
    y_pred = rng.uniform(size=n)  # uncorrelated with the label
    result = evaluate_ranking(y_pred, high_risk_true, k_fractions=(0.2,))
    lift = result["by_k"]["top_20pct"]["lift"]
    assert 0.7 < lift < 1.3  # should hover near 1.0, not be a large effect


def test_inverted_ranking_gives_lift_below_one():
    # The worst-scored rows are the true high-risk ones — picking the "risk"
    # score's top decile catches none of them.
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    high_risk_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    result = evaluate_ranking(y_pred, high_risk_true, k_fractions=(0.3,))
    assert result["by_k"]["top_30pct"]["precision"] == 0.0


def test_base_rate_matches_label_mean():
    high_risk_true = np.array([1, 0, 0, 0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = evaluate_ranking(y_pred, high_risk_true, k_fractions=(0.5,))
    assert result["base_rate_high_risk"] == 0.25


def test_evaluate_by_subgroup_detects_a_systematically_biased_group():
    # Group "A": predictions are unbiased (small random noise around truth).
    # Group "B": predictions are consistently 3 points too low — a bias that
    # a single pooled MAE would blend away and hide.
    rng = np.random.default_rng(0)
    n_per_group = 50
    y_true = pd.Series(np.concatenate([rng.normal(2, 0.1, n_per_group), rng.normal(2, 0.1, n_per_group)]))
    y_pred = np.concatenate(
        [
            y_true.iloc[:n_per_group].to_numpy() + rng.normal(0, 0.1, n_per_group),
            y_true.iloc[n_per_group:].to_numpy() - 3.0,
        ]
    )
    subgroup = pd.Series(["A"] * n_per_group + ["B"] * n_per_group)
    rows = evaluate_by_subgroup(y_true, y_pred, subgroup, min_n=10)
    by_group = {r["group"]: r for r in rows}
    assert by_group["B"]["mae"] > by_group["A"]["mae"]
    assert by_group["B"]["mean_error"] == pytest.approx(-3.0, abs=0.05)
    assert abs(by_group["A"]["mean_error"]) < 0.5


def test_evaluate_by_subgroup_drops_small_groups():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    subgroup = pd.Series(["big", "big", "big", "big", "tiny"])
    rows = evaluate_by_subgroup(y_true, y_pred, subgroup, min_n=4)
    groups = {r["group"] for r in rows}
    assert groups == {"big"}
