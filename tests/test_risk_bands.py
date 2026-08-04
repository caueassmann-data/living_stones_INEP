"""Unit tests for src/etl/build_school_risk_marts.py::assign_risk_bands."""

from __future__ import annotations

import pandas as pd

from src.etl.build_school_risk_marts import assign_risk_bands


def test_zero_heavy_distribution_does_not_collapse_bands():
    """Real marts have a large mass of schools at exactly 0% dropout. A naive
    tertile split on the raw value would put both the 33rd and 66th percentile
    cutpoints at 0, making every nonzero school look 'high'. Rank-percentile
    avoids that."""
    rates = pd.Series([0.0] * 70 + [1.0] * 15 + [10.0] * 15)
    risk_band, high_risk, q66 = assign_risk_bands(rates)
    assert (risk_band.loc[rates == 0.0] == "low").all()
    assert (risk_band.loc[rates == 10.0] == "high").all()
    assert set(risk_band.unique()) <= {"low", "moderate", "high"}


def test_absolute_override_flags_high_rate_even_if_rank_is_low():
    """A school at 6% dropout is unambiguously high-risk in absolute terms,
    even in a hypothetical distribution where most other schools are worse
    (so its rank percentile alone would not clear the 0.66 cutoff)."""
    rates = pd.Series([20.0] * 90 + [6.0] * 10)
    _, high_risk, _ = assign_risk_bands(rates)
    assert (high_risk.loc[rates == 6.0] == 1).all()


def test_high_risk_is_binary_int():
    rates = pd.Series([0.0, 1.0, 2.0, 3.0, 8.0])
    _, high_risk, _ = assign_risk_bands(rates)
    assert set(high_risk.unique()) <= {0, 1}
