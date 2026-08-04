"""Load Brazil school-risk marts and build the model-ready feature matrix."""

from __future__ import annotations

import pandas as pd

from src.utils import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS, mart_path


def load_school_mart(level: str) -> pd.DataFrame:
    """Load a level-specific school x year mart with the official dropout-rate target."""
    path = mart_path(level)
    # Avoid Path.exists(): can hang on Desktop/OneDrive notebook kernels (see src/utils.py).
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing mart {path}. Run: python -m src.etl.run_etl --level {level}"
        ) from exc
    if "target_dropout_rate" not in df.columns:
        raise ValueError(f"Mart missing target_dropout_rate: {path}")
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix (X) and continuous dropout-rate target (y).

    Only rows with a known target are dropped. Missing feature values are
    left as NaN — the modeling pipeline's imputer (src/preprocessing.py)
    handles them — rather than dropped here, because history features
    (dropout_rate_lag1, etc.) are legitimately missing for a school's first
    year in the panel, and dropping those rows would silently remove exactly
    the schools with the least track record, which is a biased thing to do.
    """
    out = df.copy()
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in out.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in out.columns]

    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in categorical_cols:
        # Keep categorical columns as-is (e.g. state_code is a string state
        # abbreviation); the one-hot encoder in the pipeline handles them.
        out[c] = out[c].astype("object").where(out[c].notna(), None)

    cols = [c for c in FEATURE_COLUMNS if c in out.columns]
    X = out[cols]
    y = pd.to_numeric(out["target_dropout_rate"], errors="coerce")
    mask = y.notna()
    # Original DataFrame index is preserved (not reset) so callers can align
    # extra columns not in FEATURE_COLUMNS — e.g. school_id for group-aware
    # splitting, or high_risk for ranking metrics — via df.loc[X.index].
    return X.loc[mask], y.loc[mask]
