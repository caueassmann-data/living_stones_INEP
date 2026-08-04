"""Tests for Brazil school-risk data contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_school_mart, prepare_xy
from src.utils import FEATURE_COLUMNS, mart_path


@pytest.mark.parametrize("level", ["fundamental", "medio"])
def test_mart_exists_and_has_official_target(level: str):
    path = mart_path(level)
    if not path.exists():
        pytest.skip(f"Mart not built yet: {path}")
    df = load_school_mart(level)
    assert "target_dropout_rate" in df.columns
    assert df["target_definition"].iloc[0] == "inep_official_dropout_rate"
    assert df["education_level"].iloc[0] == level
    assert df["target_dropout_rate"].notna().all()


@pytest.mark.parametrize("level", ["fundamental", "medio"])
def test_prepare_xy_numeric(level: str):
    path = mart_path(level)
    if not path.exists():
        pytest.skip(f"Mart not built yet: {path}")
    X, y = prepare_xy(load_school_mart(level))
    assert len(X) == len(y)
    assert len(X) > 100
    assert set(X.columns).issubset(set(FEATURE_COLUMNS))
    assert y.between(0, 100).all()


def test_no_kaggle_in_active_data_loader_source():
    text = Path("src/data_loader.py").read_text(encoding="utf-8")
    assert "dropoutKaggle" not in text
    assert "y_dropout" not in text
