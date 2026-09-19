"""Unit tests for src/evaluate.py ranking and fairness metrics, using
hand-checkable synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    evaluate_by_subgroup,
    evaluate_ranking,
    evaluate_ranking_within_group,
)


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


def test_within_group_perfect_ranking_gives_macro_precision_of_one():
    # Two networks of 30 schools; in each, the 5 top-scored are the 5 true
    # high-risk ones.
    preds, labels, groups = [], [], []
    for network in ("A", "B"):
        preds.extend(range(30, 0, -1))
        labels.extend([1] * 5 + [0] * 25)
        groups.extend([network] * 30)
    result = evaluate_ranking_within_group(
        np.array(preds), np.array(labels), pd.Series(groups), top_ns=(5,), min_group_size=10
    )
    stats = result["by_top_n"]["top_5"]
    assert stats["n_groups_evaluated"] == 2
    assert stats["precision_at_n_macro"] == 1.0
    assert stats["base_rate_macro"] == pytest.approx(5 / 30)
    assert stats["lift_macro"] == pytest.approx(6.0)


def test_within_group_skips_groups_no_bigger_than_n():
    """A group of exactly N schools has precision@N == its base rate, so lift
    is 1.0 by construction. Including such groups manufactures a fake 'no
    signal' result out of arithmetic."""
    preds = np.arange(30, 0, -1)
    labels = np.array([1] * 5 + [0] * 25)
    # One group of 20 (>= floor of 11), one of exactly 10 (== N, must be cut).
    groups = pd.Series(["big"] * 20 + ["exactly_n"] * 10)
    result = evaluate_ranking_within_group(
        preds, labels, groups, top_ns=(10,), min_group_size=1
    )
    stats = result["by_top_n"]["top_10"]
    assert stats["min_group_size_applied"] == 11  # raised above the caller's 1
    assert stats["n_groups_evaluated"] == 1
    assert stats["n_groups_skipped_too_small"] == 1
    assert result["n_groups_total"] == 2


def test_within_group_size_weighting_follows_the_large_network():
    """Macro counts each network once (the decision is taken per network);
    size-weighted follows the big ones. A gap between them means the rule
    works better at one network size than another."""
    # Small network: ranking is perfect. Large network: ranking is inverted.
    small_pred = np.arange(20, 0, -1)
    small_label = np.array([1] * 5 + [0] * 15)
    large_pred = np.arange(400)  # ascending => the top-scored are the last rows
    large_label = np.array([1] * 100 + [0] * 300)
    result = evaluate_ranking_within_group(
        np.concatenate([small_pred, large_pred]),
        np.concatenate([small_label, large_label]),
        pd.Series(["small"] * 20 + ["large"] * 400),
        top_ns=(5,),
        min_group_size=10,
    )
    stats = result["by_top_n"]["top_5"]
    assert stats["n_groups_evaluated"] == 2
    assert stats["precision_at_n_macro"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    # 420 schools, 400 of them in the network where precision is 0.
    assert stats["precision_at_n_size_weighted"] == pytest.approx(20 / 420)


def test_within_group_random_ranking_gives_lift_near_one():
    rng = np.random.default_rng(1)
    n_groups, size = 40, 200
    preds = rng.uniform(size=n_groups * size)
    labels = (rng.uniform(size=n_groups * size) < 0.3).astype(int)
    groups = pd.Series(np.repeat([f"g{i}" for i in range(n_groups)], size))
    result = evaluate_ranking_within_group(preds, labels, groups, top_ns=(20,))
    assert 0.8 < result["by_top_n"]["top_20"]["lift_macro"] < 1.2


def test_within_group_single_group_matches_the_national_metric():
    """Ties the two functions together: one group with N = 10% of the rows must
    reproduce evaluate_ranking's top-10% precision."""
    rng = np.random.default_rng(5)
    n = 500
    preds = rng.uniform(size=n)
    labels = (preds + rng.normal(0, 0.2, n) > 0.6).astype(int)
    national = evaluate_ranking(preds, labels, k_fractions=(0.1,))
    within = evaluate_ranking_within_group(
        preds, labels, pd.Series(["only"] * n), top_ns=(50,), min_group_size=51
    )
    assert within["by_top_n"]["top_50"]["precision_at_n_macro"] == pytest.approx(
        national["by_k"]["top_10pct"]["precision"]
    )


def test_within_group_reports_observed_dropout_when_given():
    preds = np.arange(30, 0, -1)
    labels = np.array([1] * 5 + [0] * 25)
    observed = np.array([12.0] * 5 + [0.5] * 25)
    result = evaluate_ranking_within_group(
        preds, labels, pd.Series(["net"] * 30), top_ns=(5,), min_group_size=10,
        observed_rate=observed,
    )
    stats = result["by_top_n"]["top_5"]
    assert stats["mean_observed_dropout_selected"] == pytest.approx(12.0)
    assert stats["mean_observed_dropout_pool"] == pytest.approx((5 * 12.0 + 25 * 0.5) / 30)


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
