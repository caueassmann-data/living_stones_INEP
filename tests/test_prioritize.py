"""Unit tests for src/prioritize.py — the Top-N decision rule.

These run without a trained model on disk: every test passes a frame that
already carries `pred_dropout_rate`, which is also how the Streamlit app calls
`prioritize()` (it scores a whole year once and reuses it across filters).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.prioritize import (
    BACKEND_LABEL_COLUMNS,
    DEFAULT_SCOPE,
    SCOPES,
    add_scope_key,
    coverage_message,
    low_signal_message,
    prioritize,
    rank_within_scope,
    scope_key_columns,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a mart-shaped frame with sensible defaults for unset columns."""
    defaults = {
        "year": 2025,
        "state_code": "BA",
        "municipality_id": 2900000,
        "municipality_name": "Somewhere",
        "admin_dependency_type": 3,
        "school_name": "School",
        "enrollment_level": 200.0,
        "dropout_rate_lag1": 1.0,
        "target_dropout_rate": 1.0,
        "has_history": 1,
        "high_risk": 0,
        "risk_band": "low",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _pool(n: int, *, state: str = "BA", muni: int = 2900000, dep: int = 3) -> pd.DataFrame:
    return _frame(
        [
            {
                "school_id": 1000 + i,
                "state_code": state,
                "municipality_id": muni,
                "admin_dependency_type": dep,
                # Descending predictions so the expected order is obvious.
                "pred_dropout_rate": float(n - i),
            }
            for i in range(n)
        ]
    )


def test_returns_exactly_top_n_when_scope_is_larger():
    result = prioritize("fundamental", df=_pool(200), year=2025, top_n=50)
    assert result.n_selected == 50
    assert result.frame["priority_rank"].tolist() == list(range(1, 51))
    assert result.frame["pred_dropout_rate"].iloc[0] == 200.0
    assert not result.frame["scope_fully_covered"].any()
    assert coverage_message(result) is None


def test_small_network_returns_whole_scope_and_says_so():
    """The small-network contract: 94% of Brazilian municipalities have fewer
    than 50 schools, so N=50 must not be presented as a prioritized subset."""
    result = prioritize("fundamental", df=_pool(7), year=2025, top_n=50)
    assert result.n_selected == 7
    assert result.frame["scope_fully_covered"].all()
    assert result.fully_covered_scopes == 1
    message = coverage_message(result)
    assert message is not None and "7 schools" in message


def test_ties_break_deterministically_regardless_of_input_order():
    """~1.9% of real 2025 predictions are exact duplicates. The same query must
    return the same schools, or last month's export cannot be reconciled."""
    tied = _frame(
        [
            {"school_id": sid, "pred_dropout_rate": 5.0, "dropout_rate_lag1": 1.0}
            for sid in [9, 3, 7, 1, 5, 2, 8, 4, 6, 10]
        ]
    )
    first = prioritize("fundamental", df=tied, year=2025, top_n=4)
    shuffled = tied.sample(frac=1.0, random_state=7)
    second = prioritize("fundamental", df=shuffled, year=2025, top_n=4)
    assert first.frame["school_id"].tolist() == second.frame["school_id"].tolist()
    # Tie-break falls through to school_id ascending once the earlier keys tie.
    assert first.frame["school_id"].tolist() == [1, 2, 3, 4]


def test_tied_predictions_never_return_more_than_n_rows():
    """Regression guard: Series.rank(method="min") would return 10 rows here."""
    tied = _frame(
        [{"school_id": i, "pred_dropout_rate": 5.0} for i in range(10)]
    )
    result = prioritize("fundamental", df=tied, year=2025, top_n=3)
    assert result.n_selected == 3


def test_ranking_happens_within_scope_not_globally():
    """A weak network still contributes its own Top-N — that is the whole point
    of prioritizing inside a network the field team actually works in."""
    strong = _pool(10, muni=2900001)
    strong["municipality_name"] = "Strongtown"
    strong["pred_dropout_rate"] = [50.0 + i for i in range(10)]
    weak = _pool(10, muni=2900002)
    weak["municipality_name"] = "Weaktown"
    weak["pred_dropout_rate"] = [0.1 * i for i in range(10)]
    weak["school_id"] = weak["school_id"] + 500

    result = prioritize(
        "fundamental", df=pd.concat([strong, weak]), year=2025, scope="network", top_n=3
    )
    assert result.n_selected == 6
    assert result.meta["n_scopes"] == 2
    per_scope = result.frame.groupby("scope_label").size()
    assert set(per_scope.unique()) == {3}


def test_selection_scope_pools_municipalities_into_one_list():
    """The consortium case: several municipalities, one shared worklist."""
    a = _pool(10, muni=2900001)
    b = _pool(10, muni=2900002)
    b["school_id"] = b["school_id"] + 500
    b["pred_dropout_rate"] = b["pred_dropout_rate"] * 10  # b dominates

    result = prioritize(
        "fundamental",
        df=pd.concat([a, b]),
        year=2025,
        scope="selection",
        municipality_ids=[2900001, 2900002],
        top_n=5,
    )
    assert result.meta["n_scopes"] == 1
    assert result.n_selected == 5
    # Pooled: the 5 strongest come from b only.
    assert result.frame["school_id"].min() >= 1500


def test_filters_compose_and_do_not_leak_rows():
    ba_mun = _pool(30, state="BA", muni=2900001, dep=3)
    ba_state = _pool(30, state="BA", muni=2900001, dep=2)
    ba_state["school_id"] = ba_state["school_id"] + 1000
    sp = _pool(30, state="SP", muni=3500001, dep=3)
    sp["school_id"] = sp["school_id"] + 2000

    result = prioritize(
        "fundamental",
        df=pd.concat([ba_mun, ba_state, sp]),
        year=2025,
        scope="network",
        state_codes=["BA"],
        municipality_ids=[2900001],
        admin_dependency_types=[3],
        top_n=10,
    )
    assert result.pool_size == 30
    assert set(result.frame["state_code"]) == {"BA"}
    assert set(result.frame["admin_dependency_label"]) == {"Municipal"}


def test_full_pool_is_ranked_no_silent_truncation():
    """app/ui_common.py used to score only filtered.head(20000) before ranking,
    so a large selection's 'top 50' was not the top 50. Lock that shut."""
    big = _pool(25_000)
    # Put the true maximum at the very end of the frame — beyond the old
    # 20,000-row cap, where it used to be invisible to the ranking.
    big.loc[big.index[-1], "pred_dropout_rate"] = 1e6
    big.loc[big.index[-1], "school_id"] = 999_999
    result = prioritize("fundamental", df=big, year=2025, top_n=5)
    assert result.pool_size == 25_000
    assert result.frame["school_id"].iloc[0] == 999_999


def test_backend_labels_are_hidden_from_the_field_team_by_default():
    """The Foundation kept `high_risk` as a historical/evaluation label and
    asked that the end user see the prioritized list instead."""
    result = prioritize("fundamental", df=_pool(20), year=2025, top_n=5)
    assert not set(BACKEND_LABEL_COLUMNS) & set(result.frame.columns)

    with_labels = prioritize(
        "fundamental", df=_pool(20), year=2025, top_n=5, include_backend_labels=True
    )
    assert set(BACKEND_LABEL_COLUMNS) <= set(with_labels.frame.columns)


def test_students_at_risk_estimate_scales_with_enrollment():
    df = _frame(
        [
            {"school_id": 1, "pred_dropout_rate": 6.0, "enrollment_level": 900.0},
            {"school_id": 2, "pred_dropout_rate": 6.0, "enrollment_level": 15.0},
        ]
    )
    result = prioritize("fundamental", df=df, year=2025, top_n=2)
    by_school = result.frame.set_index("school_id")["students_at_risk_estimate"]
    assert by_school[1] == pytest.approx(54.0)
    assert by_school[2] == pytest.approx(0.9)


def test_year_filter_selects_a_single_year():
    df = pd.concat([_pool(10), _pool(10).assign(year=2024, school_id=range(2000, 2010))])
    result = prioritize("fundamental", df=df, year=2025, top_n=100)
    assert set(result.frame["year"]) == {2025}
    assert result.pool_size == 10


def test_require_history_drops_schools_without_a_track_record():
    df = _pool(10)
    df.loc[df.index[:4], "has_history"] = 0
    result = prioritize("fundamental", df=df, year=2025, top_n=100, require_history=True)
    assert result.pool_size == 6


def test_input_frame_is_not_mutated():
    df = _pool(20)
    before = df.copy()
    prioritize("fundamental", df=df, year=2025, top_n=5)
    pd.testing.assert_frame_equal(df, before)


def test_empty_selection_returns_an_empty_result_not_an_error():
    result = prioritize("fundamental", df=_pool(10), year=1999, top_n=50)
    assert result.n_selected == 0
    assert result.meta["n_scopes"] == 0
    assert coverage_message(result) is None


@pytest.mark.parametrize("scope", SCOPES)
def test_every_scope_produces_a_usable_key(scope):
    keyed = add_scope_key(_pool(5), scope)
    assert keyed["scope_key"].notna().all()
    assert keyed["scope_label"].notna().all()
    assert isinstance(scope_key_columns(scope), list)


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="Unknown scope"):
        prioritize("fundamental", df=_pool(5), year=2025, scope="galaxy")


def test_rank_within_scope_marks_percentile_from_the_top():
    ranked = rank_within_scope(add_scope_key(_pool(4), DEFAULT_SCOPE), top_n=2)
    top = ranked.loc[ranked["priority_rank"] == 1].iloc[0]
    assert top["pred_percentile_in_scope"] == pytest.approx(1.0)
    assert top["scope_size"] == 4


def test_low_signal_warning_fires_in_a_near_zero_network():
    """Salvador's municipal Fundamental network averages 0.05% observed dropout;
    its top 50 has *lower* observed dropout than the network as a whole. A
    Top-N list always returns N schools, so nothing in the output reveals that
    there was nothing to prioritize unless we say so."""
    quiet = _pool(100)
    quiet["pred_dropout_rate"] = 0.02
    result = prioritize("fundamental", df=quiet, year=2025, top_n=50)
    assert result.n_selected == 50
    message = low_signal_message(result)
    assert message is not None and "margin of error" in message


def test_low_signal_warning_stays_quiet_when_risk_is_real():
    busy = _pool(100)
    busy["pred_dropout_rate"] = 8.0
    assert low_signal_message(prioritize("fundamental", df=busy, year=2025, top_n=50)) is None


def test_scope_summary_reports_pool_and_selection_per_network():
    strong = _pool(100, muni=2900001)
    small = _pool(3, muni=2900002)
    small["school_id"] = small["school_id"] + 500
    result = prioritize(
        "fundamental", df=pd.concat([strong, small]), year=2025, scope="network", top_n=50
    )
    summary = result.scope_summary.set_index("scope_size")
    assert summary.loc[100, "n_selected"] == 50
    assert not summary.loc[100, "scope_fully_covered"]
    assert summary.loc[3, "n_selected"] == 3
    assert summary.loc[3, "scope_fully_covered"]
