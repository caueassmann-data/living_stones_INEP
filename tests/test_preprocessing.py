"""Unit tests for src/preprocessing.py using small synthetic data (no mart needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.preprocessing import build_model_pipeline


def _synthetic_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "enrollment_level": rng.normal(200, 50, n),
            "student_teacher_ratio": rng.normal(15, 3, n),
            "admin_dependency_type": rng.choice([1, 2, 3, 4], n),
            "state_code": rng.choice(["SP", "RJ", "MG"], n),
        }
    )
    # Sprinkle in missing values to exercise the imputers.
    df.loc[0, "enrollment_level"] = np.nan
    df.loc[1, "admin_dependency_type"] = np.nan
    return df


def test_pipeline_fits_and_predicts_with_missing_values():
    X = _synthetic_frame()
    y = pd.Series(np.random.default_rng(1).normal(2.0, 1.0, len(X)))
    pipe = build_model_pipeline(
        Ridge(alpha=1.0),
        numeric_columns=["enrollment_level", "student_teacher_ratio"],
        categorical_columns=["admin_dependency_type", "state_code"],
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(X)
    assert np.isfinite(pred).all()


def test_categorical_columns_are_one_hot_not_scaled():
    X = _synthetic_frame()
    y = pd.Series(np.random.default_rng(1).normal(2.0, 1.0, len(X)))
    pipe = build_model_pipeline(
        Ridge(alpha=1.0),
        numeric_columns=["enrollment_level", "student_teacher_ratio"],
        categorical_columns=["admin_dependency_type", "state_code"],
    )
    pipe.fit(X, y)
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    # One column per (numeric feature) but multiple columns per categorical
    # feature (one-hot expansion) — state_code has 3 distinct values here.
    state_code_columns = [f for f in feature_names if f.startswith("cat__state_code_")]
    assert len(state_code_columns) == 3


def test_unknown_category_at_predict_time_does_not_raise():
    """handle_unknown='ignore' must be set — a real-world CSV upload can contain a
    state code or dependency code the training data never saw."""
    X = _synthetic_frame()
    y = pd.Series(np.random.default_rng(1).normal(2.0, 1.0, len(X)))
    pipe = build_model_pipeline(
        Ridge(alpha=1.0),
        numeric_columns=["enrollment_level", "student_teacher_ratio"],
        categorical_columns=["admin_dependency_type", "state_code"],
    )
    pipe.fit(X, y)
    unseen = X.iloc[:2].copy()
    unseen["state_code"] = "ZZ"  # not a real Brazilian state, and not in training data
    pred = pipe.predict(unseen)
    assert len(pred) == 2
