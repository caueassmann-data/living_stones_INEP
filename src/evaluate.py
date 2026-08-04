"""Evaluation and explainability for school-level regressors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.utils import RANDOM_STATE, write_json


def evaluate_on_test(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = pipe.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def export_global_importance(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    level: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Export global feature importance using native weights or permutation importance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    features = list(X_train.columns)
    model = pipe.named_steps["model"]
    method = "permutation_mae"

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "tree_feature_importances"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        method = "abs_linear_coefficients"
    else:
        result = permutation_importance(
            pipe,
            X_train,
            y_train,
            n_repeats=5,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        values = np.asarray(result.importances_mean, dtype=float)

    rows = [
        {"feature": f, "importance": float(v)}
        for f, v in sorted(zip(features, values), key=lambda t: t[1], reverse=True)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "global_importance_top.csv", index=False)
    write_json(
        {"level": level, "method": method, "rows": rows[:15]},
        out_dir / "global_importance.json",
    )
    return rows
