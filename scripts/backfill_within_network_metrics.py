"""Add within-network ranking metrics to an existing models/<level>/metrics.json.

`src/train.py` computes these natively, but training a level takes ~18 minutes
and the pipeline itself does not change here — only the metric we report about
it. This reloads the saved pipeline, reproduces the exact same held-out split
(GroupShuffleSplit is seeded with RANDOM_STATE, so the partition is
deterministic), predicts, and merges the new keys in. No refit.

Run it once after adding the metric; after that, normal training keeps it
current.

    python scripts/backfill_within_network_metrics.py
    python scripts/backfill_within_network_metrics.py --level fundamental
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_school_mart, prepare_xy
from src.prioritize import (
    DEFAULT_SCOPE,
    DEFAULT_TOP_N,
    PRIORITIZATION_VERSION,
    TOP_N_PRESETS,
)
from src.train import within_network_ranking_metrics
from src.utils import EDUCATION_LEVELS, RANDOM_STATE, model_dir, write_json


def backfill(level: str) -> dict:
    out_dir = model_dir(level)
    metrics_path = out_dir / "metrics.json"
    pipe_path = out_dir / "pipeline.joblib"
    if not pipe_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for {level} in {out_dir}. Run: python -m src.train --level {level}"
        )

    df = load_school_mart(level)
    X, y = prepare_xy(df)
    groups = df.loc[X.index, "school_id"]
    high_risk = df.loc[X.index, "high_risk"] if "high_risk" in df.columns else None

    # Must mirror src/train.py exactly, or the "held-out" rows would include
    # rows the saved pipeline was fitted on and the metric would be inflated.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(X, y, groups=groups))
    X_test = X.iloc[test_idx]
    high_risk_test = high_risk.iloc[test_idx] if high_risk is not None else None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("n_test", 0)) != len(X_test):
        raise RuntimeError(
            f"Split mismatch for {level}: metrics.json records n_test="
            f"{metrics.get('n_test')} but this split yields {len(X_test)}. The mart has "
            "changed since training — retrain instead of backfilling."
        )

    pipe = joblib.load(pipe_path)
    pred = pipe.predict(X_test)

    metrics["test_ranking_within_network"] = within_network_ranking_metrics(
        df, X_test.index, pred, high_risk_test
    )
    metrics["prioritization_policy"] = {
        "rule": "top_n_within_network",
        "default_scope": DEFAULT_SCOPE,
        "default_top_n": DEFAULT_TOP_N,
        "presets": list(TOP_N_PRESETS),
        "prioritization_version": PRIORITIZATION_VERSION,
        "backend_label_unchanged": "high_risk (assign_risk_bands) is evaluation-only",
        "backfilled_without_retraining": True,
    }
    write_json(metrics, metrics_path)
    return metrics["test_ranking_within_network"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=[*EDUCATION_LEVELS, "both"], default="both")
    args = parser.parse_args()

    levels = list(EDUCATION_LEVELS) if args.level == "both" else [args.level]
    for level in levels:
        result = backfill(level)
        print(f"\n[{level}] within-network ranking metrics written:")
        for scope, payload in (result or {}).items():
            for key, stats in payload["by_top_n"].items():
                if not stats.get("n_groups_evaluated"):
                    continue
                print(
                    f"  {scope:18s} {key:7s} "
                    f"networks={stats['n_groups_evaluated']:5d} "
                    f"precision={stats['precision_at_n_macro']:.3f} "
                    f"lift={stats['lift_macro']:.2f} "
                    f"observed_in_list={stats.get('mean_observed_dropout_selected', float('nan')):.2f}%"
                )


if __name__ == "__main__":
    main()
