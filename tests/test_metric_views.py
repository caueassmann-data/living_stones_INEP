"""The reshaping behind the plain-language metric views (app/metric_views.py)."""

from __future__ import annotations

import json

import pytest

from app.metric_views import (
    error_comparison,
    pct_lower,
    ranking_rows,
    within_network_rows,
)
from src.utils import model_dir


def test_pct_lower_reads_as_a_share_of_the_baseline():
    assert pct_lower(1.14, 1.65) == pytest.approx(0.309, abs=1e-3)
    assert pct_lower(2.0, 1.0) < 0  # worse than baseline is negative, not clipped
    assert pct_lower(None, 1.0) is None
    assert pct_lower(1.0, 0) is None  # no division by zero


def test_error_comparison_marks_which_direction_is_better():
    rows = {
        r["key"]: r
        for r in error_comparison(
            {"mae": 1.1, "rmse": 2.9, "r2": 0.29}, {"mae": 1.6, "rmse": 3.5, "r2": 0.0}
        )
    }
    assert rows["mae"]["lower_is_better"] and rows["rmse"]["lower_is_better"]
    assert not rows["r2"]["lower_is_better"]
    assert rows["mae"]["baseline"] == 1.6


def test_error_comparison_tolerates_a_missing_baseline():
    rows = error_comparison({"mae": 1.1}, {})
    assert len(rows) == 1 and rows[0]["baseline"] is None


def test_model_comparison_labels_roles_and_sorts_best_first():
    from app.metric_views import model_comparison_rows

    board = [
        {"model": "dummy_mean", "cv_mae_mean": 1.63, "cv_mae_std": 0.01, "is_baseline": True},
        {"model": "random_forest", "cv_mae_mean": 1.12, "cv_mae_std": 0.01, "is_baseline": False},
        {"model": "ridge", "cv_mae_mean": 1.36, "cv_mae_std": 0.01, "is_baseline": False},
    ]
    df = model_comparison_rows(board, "random_forest")
    assert list(df["model"]) == ["Random forest", "Ridge regression", "Always guess the average"]
    assert list(df["role"]) == ["Chosen model", "Other candidate", "Do-nothing baseline"]
    assert model_comparison_rows([], None).empty


def _ranking():
    return {
        "base_rate_high_risk": 0.35,
        "by_k": {
            "top_10pct": {"k_rows": 100, "precision": 0.76, "recall": 0.22, "lift": 2.18},
            "top_20pct": {"k_rows": 200, "precision": 0.71, "recall": 0.40, "lift": 2.02},
        },
    }


def test_ranking_rows_compare_against_the_random_rate_not_the_baseline_model():
    """The baseline model predicts a constant, so its 'ranking' just reflects
    row order in the file. Random selection is the honest comparison."""
    df = ranking_rows(_ranking())
    assert list(df["cut"]) == ["Top 10% of schools", "Top 20% of schools"]
    assert (df["random"] == 0.35).all()
    assert df.loc[0, "precision"] == 0.76


def test_ranking_rows_handles_missing_input():
    assert ranking_rows(None).empty
    assert ranking_rows({}).empty


def _within():
    stats = {
        "n_groups_evaluated": 500,
        "precision_at_n_macro": 0.5,
        "base_rate_macro": 0.35,
        "lift_macro": 1.45,
        "mean_observed_dropout_selected": 2.5,
        "mean_observed_dropout_pool": 0.9,
    }
    return {
        "state_network": {"by_top_n": {"top_20": stats, "top_50": stats}},
        "municipal_network": {
            "by_top_n": {"top_20": stats, "top_50": {"n_groups_evaluated": 0}}
        },
    }


def test_within_network_rows_drop_sizes_with_nothing_to_measure():
    df = within_network_rows(_within())
    assert len(df) == 3  # the empty municipal top_50 is excluded
    assert set(df["n"]) == {20, 50}
    assert set(df["kind_key"]) == {"state_network", "municipal_network"}


def test_within_network_rows_handle_missing_input():
    assert within_network_rows(None).empty


def test_real_metrics_json_reshapes_for_both_levels():
    """Guards the views against the shape of the artifacts actually shipped."""
    for level in ("fundamental", "medio"):
        path = model_dir(level) / "metrics.json"
        if not path.exists():
            pytest.skip("models not trained")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert not ranking_rows(metrics["test_ranking_metrics"]).empty
        assert not within_network_rows(metrics.get("test_ranking_within_network")).empty
        assert error_comparison(metrics["test_metrics"], metrics["baseline_test_metrics"])
