"""Unit tests for src/etl/build_school_risk_marts.py::_add_history_features.

This is the core of the v0.3.0 validity fix (see docs/validity_and_english_revision.md):
using a school's own dropout-rate history as a predictor. These tests build a
tiny, hand-checkable attainment panel instead of depending on the multi-GB
real INEP extracts, so they run anywhere and catch regressions in the lag
math directly.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.etl.build_school_risk_marts import LEVELS, _add_history_features


def _panel() -> pd.DataFrame:
    # One school (1) with 4 years of dropout rates: 1.0, 2.0, 3.0, 4.0
    # A second school (2) with a single year (no history should ever leak in).
    return pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021, 2020],
            "school_id": [1, 1, 1, 1, 2],
            "municipality_id": [10, 10, 10, 10, 10],
            "state_code": ["SP", "SP", "SP", "SP", "SP"],
            "dropout_rate_fundamental": [1.0, 2.0, 3.0, 4.0, 5.0],
            "approval_rate_fundamental": [90.0, 88.0, 86.0, 84.0, 80.0],
            "failure_rate_fundamental": [9.0, 10.0, 11.0, 12.0, 15.0],
        }
    )


def test_lag1_and_lag2_use_only_strictly_earlier_years():
    hist = _add_history_features(_panel(), "fundamental")
    row_2021 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2021)].iloc[0]
    assert row_2021["dropout_rate_lag1"] == 3.0  # 2020's rate
    assert row_2021["dropout_rate_lag2"] == 2.0  # 2019's rate


def test_first_year_has_no_history_and_is_flagged():
    hist = _add_history_features(_panel(), "fundamental")
    row_2018 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2018)].iloc[0]
    assert math.isnan(row_2018["dropout_rate_lag1"])
    assert row_2018["has_history"] == 0

    row_2019 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2019)].iloc[0]
    assert row_2019["has_history"] == 1
    assert row_2019["dropout_rate_lag1"] == 1.0


def test_3yr_trailing_average_excludes_current_year():
    hist = _add_history_features(_panel(), "fundamental")
    row_2021 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2021)].iloc[0]
    # Trailing average of 2018-2020 (years strictly before 2021): (1+2+3)/3
    assert row_2021["dropout_rate_3yr_avg"] == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def test_trend_is_lag1_minus_lag2():
    hist = _add_history_features(_panel(), "fundamental")
    row_2021 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2021)].iloc[0]
    assert row_2021["dropout_rate_trend"] == 3.0 - 2.0


def test_approval_and_failure_lag_come_from_prior_year_only():
    hist = _add_history_features(_panel(), "fundamental")
    row_2019 = hist.loc[(hist["school_id"] == 1) & (hist["year"] == 2019)].iloc[0]
    assert row_2019["approval_rate_lag1"] == 90.0
    assert row_2019["failure_rate_lag1"] == 9.0


def test_second_school_never_sees_first_schools_history():
    hist = _add_history_features(_panel(), "fundamental")
    row = hist.loc[hist["school_id"] == 2].iloc[0]
    assert math.isnan(row["dropout_rate_lag1"])
    assert row["has_history"] == 0


def test_both_levels_are_registered_with_matching_column_names():
    for level, meta in LEVELS.items():
        assert meta["dropout_col"].startswith("dropout_rate_")
        assert meta["approval_col"].startswith("approval_rate_")
        assert meta["failure_col"].startswith("failure_rate_")
