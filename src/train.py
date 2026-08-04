"""Train level-specific school abandonment regressors and export artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from src.data_loader import load_school_mart, prepare_xy
from src.evaluate import evaluate_on_test, export_global_importance
from src.preprocessing import build_model_pipeline
from src.utils import (
    MODEL_VERSION,
    RANDOM_STATE,
    model_dir,
    write_json,
)


def _candidates() -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cross_validate_candidates(
    X: pd.DataFrame, y: pd.Series, feature_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    rows = []
    best_name = None
    best_mae = float("inf")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, est in _candidates().items():
        pipe = build_model_pipeline(est, feature_columns)
        scores = -cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        mae = float(scores.mean())
        rows.append({"model": name, "cv_mae_mean": mae, "cv_mae_std": float(scores.std())})
        if mae < best_mae:
            best_mae = mae
            best_name = name
    return pd.DataFrame(rows).sort_values("cv_mae_mean"), best_name or "ridge"


def train_level(level: str, max_rows: int = 80000) -> dict[str, Any]:
    t0 = time.time()
    df = load_school_mart(level)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)
    X, y = prepare_xy(df)
    feature_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv_table, winner = cross_validate_candidates(X_train, y_train, feature_columns)
    est = _candidates()[winner]
    pipe = build_model_pipeline(est, feature_columns)
    pipe.fit(X_train, y_train)
    test_metrics = evaluate_on_test(pipe, X_test, y_test)

    out = model_dir(level)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "pipeline.joblib")
    write_json({"features": feature_columns, "level": level}, out / "feature_list.json")
    importance = export_global_importance(
        pipe, X_train, y_train, level=level, out_dir=out / "figures"
    )
    meta = {
        "model_version": MODEL_VERSION,
        "education_level": level,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "selected_model": winner,
        "cv_leaderboard": cv_table.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "target": "target_dropout_rate",
        "target_definition": "inep_taxa_abandono_official",
        "seconds": round(time.time() - t0, 2),
        "feature_importance_top": importance[:10],
    }
    write_json(meta, out / "metrics.json")
    write_json(
        {
            "model_version": MODEL_VERSION,
            "level": level,
            "selected_model": winner,
            "limitation": (
                "School-level Brazil INEP abandonment model; not student-level scoring."
            ),
        },
        out / "model_card_meta.json",
    )
    print(f"[{level}] selected={winner} test_mae={test_metrics['mae']:.3f} r2={test_metrics['r2']:.3f}")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Brazil school abandonment models")
    parser.add_argument(
        "--level",
        choices=["fundamental", "medio", "both"],
        default="both",
    )
    args = parser.parse_args(argv)
    levels = ["fundamental", "medio"] if args.level == "both" else [args.level]
    for level in levels:
        train_level(level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
