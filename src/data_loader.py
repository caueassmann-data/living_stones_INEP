"""Load Brazil school-risk marts for Fundamental / Médio modeling."""

from __future__ import annotations

import pandas as pd

from src.utils import FEATURE_COLUMNS, mart_path


def load_school_mart(level: str) -> pd.DataFrame:
    """Load a level-specific school×year mart with official abandonment target."""
    path = mart_path(level)
    # Avoid Path.exists(): can hang on Desktop/OneDrive notebook kernels.
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
    """Build model matrix and continuous abandonment target."""
    out = df.copy()
    if "student_teacher_ratio" not in out.columns:
        teachers = pd.to_numeric(out.get("qt_doc_bas"), errors="coerce")
        enroll = pd.to_numeric(out.get("enrollment_level"), errors="coerce")
        out["student_teacher_ratio"] = enroll / teachers.replace({0: pd.NA})
    cols = [c for c in FEATURE_COLUMNS if c in out.columns]
    X = out[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(out["target_dropout_rate"], errors="coerce")
    mask = y.notna() & X.notna().all(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
