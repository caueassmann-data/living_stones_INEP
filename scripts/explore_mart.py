"""Quick exploration helpers for Brazil school-risk marts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "latam_education_data" / "marts"


def load(level: str) -> pd.DataFrame:
    name = f"school_risk_br_{level}"
    path = MARTS / name / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Brazil school-risk marts")
    parser.add_argument("--level", choices=["fundamental", "medio"], default="fundamental")
    parser.add_argument(
        "--query",
        choices=["summary", "top_states", "head"],
        default="summary",
    )
    args = parser.parse_args()
    df = load(args.level)
    if args.query == "summary":
        print(
            {
                "level": args.level,
                "rows": len(df),
                "schools": int(df["school_id"].nunique()),
                "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
                "mean_dropout_rate": float(df["target_dropout_rate"].mean()),
                "median_dropout_rate": float(df["target_dropout_rate"].median()),
            }
        )
    elif args.query == "top_states":
        g = (
            df.groupby("state_code")
            .agg(n=("target_dropout_rate", "size"), mean=("target_dropout_rate", "mean"))
            .sort_values("mean", ascending=False)
            .head(15)
        )
        print(g)
    else:
        print(df.head(20))


if __name__ == "__main__":
    main()
