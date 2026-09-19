"""Build the slim artifacts a hosted Streamlit deployment needs.

The full marts are ~50 MB on disk but ~620 MB once loaded into pandas, and
they are gitignored build output, so a fresh clone cannot run the app at all.
Streamlit Community Cloud deploys from a plain `git clone`, so it needs both
problems solved: the artifacts have to be in the repository, and they have to
fit in the free tier's memory.

This writes a two-year slice of each mart to latam_education_data/marts_app/,
which is committed. Measured: 10.6 MB on disk and ~154 MB in memory, against
49.7 MB / ~620 MB for the full marts.

Only the recent years are kept because the app's prioritized list is built for
one year at a time and its year picker is derived from whatever is loaded. The
full marts stay on the machine that trains; nothing here touches them.

    python scripts/build_deploy_artifacts.py
    python scripts/build_deploy_artifacts.py --years 3

Run it again after any retrain or ETL rebuild, then commit the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import APP_MARTS_ROOT, EDUCATION_LEVELS, app_mart_path, mart_path, model_dir

DEFAULT_YEARS_KEPT = 2

# Estimators whose unpickling needs no library beyond the deployment
# requirements. An XGBoost winner would import fine locally and fail on the
# host, where xgboost is deliberately not installed (it is a large wheel and
# only src/train.py needs it), so the mismatch is caught here instead.
RUNTIME_SUPPORTED_MODELS = {"random_forest", "ridge", "dummy_mean", "dummy_median"}


def build_slim_mart(level: str, years_kept: int) -> dict:
    source = mart_path(level)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing full mart for {level}: {source}. Run: python -m src.etl.run_etl"
        )

    full = pd.read_parquet(source)
    years = sorted(pd.to_numeric(full["year"], errors="coerce").dropna().astype(int).unique())
    keep = years[-years_kept:]
    slim = full.loc[pd.to_numeric(full["year"], errors="coerce").isin(keep)].copy()

    target = app_mart_path(level)
    target.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(target, compression="zstd", index=False)

    return {
        "level": level,
        "path": str(target.relative_to(PROJECT_ROOT)),
        "years": [int(y) for y in keep],
        "rows_full": int(len(full)),
        "rows_slim": int(len(slim)),
        "schools": int(slim["school_id"].nunique()),
        "disk_mb": round(target.stat().st_size / 1e6, 2),
        "memory_mb": round(float(slim.memory_usage(deep=True).sum()) / 1e6, 1),
    }


def check_model_artifacts(level: str) -> dict:
    d = model_dir(level)
    pipeline = d / "pipeline.joblib"
    metrics_path = d / "metrics.json"
    if not pipeline.exists():
        raise FileNotFoundError(
            f"Missing {pipeline}. Run: python -m src.train --level {level}"
        )
    selected = json.loads(metrics_path.read_text(encoding="utf-8")).get("selected_model")
    if selected not in RUNTIME_SUPPORTED_MODELS:
        raise RuntimeError(
            f"[{level}] selected_model is '{selected}', which the deployment requirements do "
            f"not cover (supported: {sorted(RUNTIME_SUPPORTED_MODELS)}). Add the library to "
            "requirements.txt before deploying, or the hosted app will fail loading the model."
        )
    return {
        "level": level,
        "selected_model": selected,
        "pipeline_mb": round(pipeline.stat().st_size / 1e6, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS_KEPT,
        help=f"How many of the most recent years to ship (default {DEFAULT_YEARS_KEPT}).",
    )
    parser.add_argument("--levels", nargs="*", default=list(EDUCATION_LEVELS))
    args = parser.parse_args()

    marts = [build_slim_mart(level, args.years) for level in args.levels]
    models = [check_model_artifacts(level) for level in args.levels]

    print(f"Deploy marts written to {APP_MARTS_ROOT.relative_to(PROJECT_ROOT)}\n")
    for m in marts:
        print(
            f"  {m['level']:12s} years {m['years']}  "
            f"{m['rows_slim']:>7,} of {m['rows_full']:,} rows  "
            f"{m['disk_mb']:>6.1f} MB disk  {m['memory_mb']:>6.1f} MB memory"
        )
    print()
    for m in models:
        print(f"  {m['level']:12s} model '{m['selected_model']}'  {m['pipeline_mb']:>6.1f} MB")

    payload = sum(m["disk_mb"] for m in marts) + sum(m["pipeline_mb"] for m in models)
    memory = sum(m["memory_mb"] for m in marts)
    print(f"\n  Repository payload: {payload:.1f} MB   App memory (marts): {memory:.1f} MB")
    print(
        "\nNext: commit latam_education_data/marts_app/ and models/*/pipeline.joblib, "
        "push, and point Streamlit Community Cloud at app/main.py."
    )


if __name__ == "__main__":
    main()
