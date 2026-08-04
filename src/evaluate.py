"""Evaluation and explainability for the Brazil school-level dropout-rate models.

Four kinds of metrics are computed, because they answer different questions:

  1. `evaluate_on_test` — regression error (MAE/RMSE/R2). Answers "how far
     off is a single prediction, in percentage points of dropout rate."
  2. `evaluate_ranking` — precision@k, recall@k, lift, and average precision
     against the mart's `high_risk` label. Answers the question the product
     actually needs: "if we sort schools by predicted risk and act on the
     top decile, how many of the truly highest-risk schools do we catch."
     A model can have a mediocre MAE and still be a genuinely useful triage
     tool if its ranking is good — this is why both are reported.
  3. `evaluate_by_subgroup` — error broken out by rural/urban, public/private,
     and enrollment size. A model with a good *average* error can still be
     systematically worse for one of these groups; the model card
     (docs/model_card.md) explicitly warns against punitive use, and that
     warning is not backed by anything unless someone actually checks whether
     errors are evenly distributed. This checks it.
  4. `export_global_importance` — which features the winning model relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.utils import RANDOM_STATE, write_json


def evaluate_on_test(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = pipe.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def evaluate_ranking(
    y_pred: np.ndarray,
    high_risk_true: np.ndarray,
    *,
    k_fractions: tuple[float, ...] = (0.1, 0.2, 0.33),
) -> dict[str, Any]:
    """Triage-oriented ranking metrics.

    `high_risk_true` is the mart's own `high_risk` label (top-third by
    dropout rate, or >=5% absolute — see src/etl/build_school_risk_marts.py).
    For each k in `k_fractions`, schools are ranked by predicted dropout rate
    and the top k-fraction is treated as "the schools we would act on":

      precision@k = share of the selected schools that are truly high-risk
      recall@k    = share of all truly high-risk schools the selection catches
      lift@k      = precision@k divided by the overall high-risk base rate
                    (lift of 1.0 = no better than picking schools at random)

    average_precision is the area under the precision-recall curve treating
    the continuous prediction as a risk score — a single ranking-quality
    number that does not require picking a k.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    high_risk_true = np.asarray(high_risk_true, dtype=int)
    n = len(y_pred)
    base_rate = float(high_risk_true.mean()) if n else float("nan")
    order = np.argsort(-y_pred)  # descending predicted risk

    by_k: dict[str, dict[str, float]] = {}
    for k_frac in k_fractions:
        k = max(1, int(round(n * k_frac)))
        selected = order[:k]
        precision = float(high_risk_true[selected].mean())
        total_positive = int(high_risk_true.sum())
        recall = float(high_risk_true[selected].sum() / total_positive) if total_positive else float("nan")
        lift = precision / base_rate if base_rate else float("nan")
        by_k[f"top_{int(round(k_frac * 100))}pct"] = {
            "k_rows": k,
            "precision": precision,
            "recall": recall,
            "lift": lift,
        }

    return {
        "base_rate_high_risk": base_rate,
        "average_precision": float(average_precision_score(high_risk_true, y_pred))
        if len(np.unique(high_risk_true)) > 1
        else float("nan"),
        "by_k": by_k,
    }


def evaluate_by_subgroup(
    y_true: pd.Series, y_pred: np.ndarray, subgroup: pd.Series, *, min_n: int = 30
) -> list[dict[str, Any]]:
    """MAE/RMSE/mean-error broken out by a categorical subgroup.

    `mean_error` (signed, pred - true) is reported alongside MAE because a
    model can have an unremarkable MAE for a group while still being biased
    in one direction for it (e.g. systematically under-predicting rural
    schools) — MAE alone cannot distinguish "noisy but centered" from
    "consistently wrong in one direction," and only the second is a fairness
    concern in the sense docs/model_card.md warns about.

    Groups with fewer than `min_n` rows are skipped: a subgroup MAE from a
    handful of rows is noise, not a finding, and reporting it invites
    over-interpretation.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float), "y_pred": np.asarray(y_pred, dtype=float)})
    df["group"] = pd.Series(subgroup).reset_index(drop=True).astype("string").fillna("(missing)")
    df["error"] = df["y_pred"] - df["y_true"]
    rows = []
    for group_value, g in df.groupby("group"):
        if len(g) < min_n:
            continue
        rows.append(
            {
                "group": group_value,
                "n": int(len(g)),
                "mae": float(g["error"].abs().mean()),
                "rmse": float(np.sqrt((g["error"] ** 2).mean())),
                "mean_error": float(g["error"].mean()),
            }
        )
    return sorted(rows, key=lambda r: r["mae"], reverse=True)


def _readable_feature_names(pipe: Pipeline) -> list[str]:
    """Strip the ColumnTransformer's 'num__'/'cat__' prefixes for display.

    One-hot-encoded categorical columns (e.g. state_code) expand into many
    columns like 'cat__state_code_SP'; this keeps them readable as
    'state_code_SP' in exported importance tables and app charts.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    out = []
    for n in names:
        for prefix in ("num__", "cat__"):
            if n.startswith(prefix):
                n = n[len(prefix) :]
                break
        out.append(n)
    return out


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
    features = _readable_feature_names(pipe)
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

    if len(values) != len(features):
        # Defensive: should not happen, but avoids a confusing zip-truncation
        # if a future estimator's coefficient shape does not match 1:1.
        n = min(len(values), len(features))
        features, values = features[:n], values[:n]

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
