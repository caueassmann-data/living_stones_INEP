"""EDA helpers for Brazil school-risk marts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import PROJECT_ROOT, mart_path


def load_level_mart(level: str) -> pd.DataFrame:
    return pd.read_parquet(mart_path(level))


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().rename("pct_missing").to_frame()
    miss["n_missing"] = df.isna().sum()
    miss["dtype"] = df.dtypes.astype(str)
    return miss.sort_values("pct_missing", ascending=False)


def summary_by_uf(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    g = (
        df.groupby("uf", as_index=False)
        .agg(
            n=("target_dropout_rate", "size"),
            mean_abandono=("target_dropout_rate", "mean"),
            median_abandono=("target_dropout_rate", "median"),
        )
        .sort_values("mean_abandono", ascending=False)
    )
    return g.loc[g["n"] >= min_n]


def compare_levels() -> pd.DataFrame:
    rows = []
    for level in ("fundamental", "medio"):
        path = mart_path(level)
        # Avoid Path.exists() here: on OneDrive/Desktop it can hang notebook kernels.
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            continue
        rows.append(
            {
                "level": level,
                "rows": len(df),
                "schools": df["school_id"].nunique(),
                "mean_abandono": float(df["target_dropout_rate"].mean()),
                "median_abandono": float(df["target_dropout_rate"].median()),
                "p90_abandono": float(df["target_dropout_rate"].quantile(0.9)),
                "public_share": float(df["is_public"].mean()) if "is_public" in df else None,
                "rural_share": float(df["is_rural"].mean()) if "is_rural" in df else None,
            }
        )
    return pd.DataFrame(rows)


def validation_json_path() -> Path:
    return PROJECT_ROOT / "latam_education_data" / "dq" / "rendimento_validation.json"
