"""Unit tests for src/data_loader.py::prepare_xy using a small synthetic mart-like frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import prepare_xy
from src.utils import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


def _synthetic_mart(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target_dropout_rate": rng.uniform(0, 10, n)})
    for c in NUMERIC_FEATURE_COLUMNS:
        df[c] = rng.normal(1.0, 1.0, n)
    for c in CATEGORICAL_FEATURE_COLUMNS:
        df[c] = "A"
    return df


def test_prepare_xy_keeps_rows_with_missing_features():
    """Missing feature values (e.g. a school's first year, with no lag history)
    must NOT be dropped — only rows with a missing target are dropped. Earlier
    versions of this function dropped any row with any missing feature, which
    silently discarded exactly the schools with the least history."""
    df = _synthetic_mart(20)
    df.loc[0, "dropout_rate_lag1"] = np.nan
    df.loc[1, "enrollment_level"] = np.nan
    X, y = prepare_xy(df)
    assert len(X) == 20
    assert len(y) == 20
    assert X["dropout_rate_lag1"].isna().sum() == 1


def test_prepare_xy_drops_rows_with_missing_target():
    df = _synthetic_mart(10)
    df.loc[0, "target_dropout_rate"] = np.nan
    X, y = prepare_xy(df)
    assert len(X) == 9
    assert len(y) == 9


def test_prepare_xy_preserves_index_for_alignment():
    """train.py aligns school_id/year/high_risk back onto X via X.index — this
    only works if prepare_xy does not reset the index."""
    df = _synthetic_mart(10)
    df["school_id"] = range(100, 110)
    X, y = prepare_xy(df)
    aligned = df.loc[X.index, "school_id"]
    assert list(aligned) == list(df["school_id"])


def test_categorical_columns_not_coerced_to_numeric():
    df = _synthetic_mart(5)
    df["state_code"] = "SP"
    X, _ = prepare_xy(df)
    if "state_code" in X.columns:
        assert X["state_code"].iloc[0] == "SP"
