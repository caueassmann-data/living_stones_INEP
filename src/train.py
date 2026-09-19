"""Train level-specific school dropout-rate regressors and export artifacts.

WHAT CHANGED IN v0.3.0 AND WHY
--------------------------------
The previous version of this file compared three real candidates (Ridge,
RandomForest, XGBoost) against *each other* only. On the actual marts, all
three lost to the trivial "always predict the training mean" baseline on
MAE — the metric used to pick the winner — because nothing in the leaderboard
ever checked that. This file now:

  1. Adds `dummy_mean` / `dummy_median` baselines to the leaderboard and
     refuses to silently celebrate a "winner" that does not beat them
     (`beats_baseline` in the exported metrics; a loud warning is printed).
  2. Splits by `school_id` (GroupShuffleSplit / GroupKFold), not by row.
     With ~6-8 years of data per school, a row-level split lets the same
     school appear in both train and test, which inflates every metric.
  3. Adds a *temporal* holdout (train on years <= 2023, test on 2024-2025)
     in addition to the group holdout. This is the scenario the product
     actually promises — predicting a *future* year — and it is a strictly
     harder, more honest test than holding out random schools.
  4. Reports ranking metrics (precision@k, recall@k, lift, average
     precision) against the mart's `high_risk` label, because the product
     is a triage tool: a modest MAE can still support a genuinely useful
     ranking of "which schools to look at first."
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from xgboost import XGBRegressor

from src.data_loader import load_school_mart, prepare_xy
from src.evaluate import (
    evaluate_by_subgroup,
    evaluate_on_test,
    evaluate_ranking,
    evaluate_ranking_within_group,
    export_global_importance,
)
from src.preprocessing import build_model_pipeline
from src.prioritize import (
    DEFAULT_SCOPE,
    DEFAULT_TOP_N,
    PRIORITIZATION_VERSION,
    TOP_N_PRESETS,
)
from src.utils import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_VERSION,
    NUMERIC_FEATURE_COLUMNS,
    RANDOM_STATE,
    model_dir,
    write_json,
)

# Out-of-time validation years: never used for model selection, only as a
# final "would this have worked on the future" check on the chosen model.
TEMPORAL_HOLDOUT_YEARS = (2024, 2025)

BASELINE_NAMES = {"dummy_mean", "dummy_median"}


def _candidates() -> dict[str, Any]:
    # RandomForest's min_samples_leaf/max_depth are deliberately conservative:
    # on the full mart (600k+ training rows), an unconstrained forest grows
    # into a 150+MB pipeline.joblib and takes much longer to train than
    # XGBoost for a CV-MAE gain of only ~4-5% — not a good trade for a model
    # that has to be loaded on every Streamlit app interaction. These settings
    # keep the artifact in the tens-of-MB range with negligible accuracy loss.
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=25,
            max_depth=18,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, str, float]:
    """Compare candidates (including trivial baselines) with school-grouped 5-fold CV.

    GroupKFold groups by school_id so that a school's rows never straddle
    both the training and validation side of a single fold — see the module
    docstring for why a plain row-level KFold was invalidating the results.
    """
    rows = []
    best_name = None
    best_mae = float("inf")
    cv = GroupKFold(n_splits=5)
    for name, est in _candidates().items():
        pipe = build_model_pipeline(est, numeric_columns, categorical_columns)
        scores = -cross_val_score(
            pipe, X, y, groups=groups, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        mae = float(scores.mean())
        rows.append(
            {
                "model": name,
                "cv_mae_mean": mae,
                "cv_mae_std": float(scores.std()),
                "is_baseline": name in BASELINE_NAMES,
            }
        )
        if name not in BASELINE_NAMES and mae < best_mae:
            best_mae = mae
            best_name = name
    table = pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)
    baseline_mae = float(table.loc[table["is_baseline"], "cv_mae_mean"].min())
    return table, best_name or "ridge", baseline_mae


def _fit_and_score(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    high_risk_test: pd.Series | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
    within_network_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit one candidate on a train partition and score it on a held-out partition.

    `within_network_df` is the full mart; pass it to also report ranking
    quality inside each network, which is the metric that matches the deployed
    Top-N rule. Skipped for baselines, where it would be noise.
    """
    est = _candidates()[name]
    pipe = build_model_pipeline(est, numeric_columns, categorical_columns)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    out: dict[str, Any] = evaluate_on_test(pipe, X_test, y_test)
    if high_risk_test is not None and high_risk_test.notna().any():
        out["ranking"] = evaluate_ranking(pred, high_risk_test.fillna(0).to_numpy())
        if within_network_df is not None:
            out["ranking_within_network"] = within_network_ranking_metrics(
                within_network_df, X_test.index, pred, high_risk_test
            )
    return out


def _evaluate_fairness_subgroups(
    df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, test_pred: np.ndarray
) -> dict[str, list[dict[str, Any]]]:
    """Break the held-out error down by rural/urban, public/private, and school size.

    See docs/model_card.md "Equity notes": a model with a good average error
    can still be systematically worse for one of these groups, and that is
    exactly the scenario the model card warns against using punitively.
    """
    subgroups: dict[str, list[dict[str, Any]]] = {}
    is_rural = df.loc[X_test.index].get("is_rural")
    if is_rural is not None:
        labels = is_rural.map({1: "Rural", 0: "Urban", 1.0: "Rural", 0.0: "Urban"})
        subgroups["location"] = evaluate_by_subgroup(y_test, test_pred, labels)
    is_public = df.loc[X_test.index].get("is_public")
    if is_public is not None:
        labels = is_public.map({1: "Public", 0: "Private", 1.0: "Public", 0.0: "Private"})
        subgroups["admin_network"] = evaluate_by_subgroup(y_test, test_pred, labels)
    enrollment = df.loc[X_test.index].get("enrollment_level")
    if enrollment is not None:
        size_bucket = pd.cut(
            pd.to_numeric(enrollment, errors="coerce"),
            bins=[-1, 100, 300, 700, float("inf")],
            labels=["<=100 students", "101-300 students", "301-700 students", ">700 students"],
        )
        subgroups["school_size"] = evaluate_by_subgroup(y_test, test_pred, size_bucket)
    return subgroups


def within_network_ranking_metrics(
    df: pd.DataFrame,
    index: pd.Index,
    y_pred: np.ndarray,
    positives: pd.Series | None,
) -> dict[str, Any] | None:
    """Ranking quality inside each education network, on a held-out partition.

    This is the metric that matches the deployed decision rule: a field team
    picks one network and takes the top N schools in it (src/prioritize.py).
    The national `test_ranking_metrics` answers a different question and reads
    higher, because schools across Brazil differ far more than schools inside
    one network.

    The group key includes `year`. The grouped split holds out whole schools,
    not whole years, so a network's test rows span several years; ranking a
    2019 school against a 2025 one is not the operational setting and would
    mix a COVID year into a normal one.
    """
    if positives is None or not positives.notna().any():
        return None

    rows = df.loc[index]
    year = rows["year"].astype(str)
    out: dict[str, Any] = {}
    group_specs = {
        "state_network": ["state_code", "admin_dependency_type"],
        "municipal_network": ["municipality_id", "admin_dependency_type"],
    }
    for name, cols in group_specs.items():
        if not all(c in rows.columns for c in cols):
            continue
        key = year.str.cat([rows[c].astype(str) for c in cols], sep="|")
        out[name] = evaluate_ranking_within_group(
            y_pred,
            positives.fillna(0).to_numpy(),
            key,
            top_ns=(20, 50),
            observed_rate=pd.to_numeric(rows["target_dropout_rate"], errors="coerce").to_numpy(),
        )
    return out or None


def train_level(level: str, max_rows: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    df = load_school_mart(level)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)

    X, y = prepare_xy(df)
    numeric_columns = [c for c in NUMERIC_FEATURE_COLUMNS if c in X.columns]
    categorical_columns = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in X.columns]
    groups = df.loc[X.index, "school_id"]
    years = pd.to_numeric(df.loc[X.index, "year"], errors="coerce")
    high_risk = df.loc[X.index, "high_risk"] if "high_risk" in df.columns else None

    # --- Primary evaluation: held-out SCHOOLS (group split) ---------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    high_risk_test = high_risk.iloc[test_idx] if high_risk is not None else None

    cv_table, winner, baseline_cv_mae = cross_validate_candidates(
        X_train, y_train, groups_train, numeric_columns, categorical_columns
    )
    beats_baseline_cv = bool(cv_table.loc[cv_table["model"] == winner, "cv_mae_mean"].iloc[0] < baseline_cv_mae)
    if not beats_baseline_cv:
        print(
            f"[{level}] WARNING: selected model '{winner}' does NOT beat the trivial "
            f"baseline on cross-validated MAE ({baseline_cv_mae:.3f}). Treat this model "
            "as not yet production-ready for triage."
        )

    est = _candidates()[winner]
    pipe = build_model_pipeline(est, numeric_columns, categorical_columns)
    pipe.fit(X_train, y_train)
    test_metrics = evaluate_on_test(pipe, X_test, y_test)
    test_pred = pipe.predict(X_test)
    ranking_metrics = (
        evaluate_ranking(test_pred, high_risk_test.fillna(0).to_numpy())
        if high_risk_test is not None
        else None
    )
    within_network_metrics = within_network_ranking_metrics(
        df, X_test.index, test_pred, high_risk_test
    )
    baseline_test_metrics = _fit_and_score(
        "dummy_mean", X_train, y_train, X_test, y_test, high_risk_test, numeric_columns, categorical_columns
    )
    beats_baseline_test = bool(test_metrics["mae"] < baseline_test_metrics["mae"])
    subgroup_metrics = _evaluate_fairness_subgroups(df, X_test, y_test, test_pred)

    # --- Secondary evaluation: held-out FUTURE YEARS (temporal holdout) ---
    temporal_test_mask = years.isin(TEMPORAL_HOLDOUT_YEARS)
    temporal_train_mask = ~temporal_test_mask
    temporal_metrics: dict[str, Any] | None = None
    temporal_baseline_metrics: dict[str, Any] | None = None
    if temporal_test_mask.sum() >= 50 and temporal_train_mask.sum() >= 50:
        Xt_train, Xt_test = X.loc[temporal_train_mask], X.loc[temporal_test_mask]
        yt_train, yt_test = y.loc[temporal_train_mask], y.loc[temporal_test_mask]
        hr_test = high_risk.loc[temporal_test_mask] if high_risk is not None else None
        temporal_metrics = _fit_and_score(
            winner,
            Xt_train,
            yt_train,
            Xt_test,
            yt_test,
            hr_test,
            numeric_columns,
            categorical_columns,
            within_network_df=df,
        )
        temporal_baseline_metrics = _fit_and_score(
            "dummy_mean", Xt_train, yt_train, Xt_test, yt_test, hr_test, numeric_columns, categorical_columns
        )

    out = model_dir(level)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "pipeline.joblib", compress=3)
    write_json(
        {"numeric_features": numeric_columns, "categorical_features": categorical_columns, "level": level},
        out / "feature_list.json",
    )
    importance = (
        export_global_importance(pipe, X_train, y_train, level=level, out_dir=out / "figures")
        if winner not in BASELINE_NAMES
        else []
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
        "beats_baseline_cv": beats_baseline_cv,
        "beats_baseline_test": beats_baseline_test,
        "test_metrics": test_metrics,
        "test_ranking_metrics": ranking_metrics,
        "test_ranking_within_network": within_network_metrics,
        "prioritization_policy": {
            "rule": "top_n_within_network",
            "default_scope": DEFAULT_SCOPE,
            "default_top_n": DEFAULT_TOP_N,
            "presets": list(TOP_N_PRESETS),
            "prioritization_version": PRIORITIZATION_VERSION,
            "backend_label_unchanged": "high_risk (assign_risk_bands) is evaluation-only",
        },
        "baseline_test_metrics": baseline_test_metrics,
        "subgroup_metrics": subgroup_metrics,
        "temporal_holdout": {
            "description": (
                f"Trained on years <= {min(TEMPORAL_HOLDOUT_YEARS) - 1}, evaluated on "
                f"years {list(TEMPORAL_HOLDOUT_YEARS)} — the model never sees these rows "
                "or any row from these years during training."
            ),
            "n_train": int(temporal_train_mask.sum()),
            "n_test": int(temporal_test_mask.sum()),
            "metrics": temporal_metrics,
            "baseline_metrics": temporal_baseline_metrics,
        }
        if temporal_metrics is not None
        else None,
        "target": "target_dropout_rate",
        "target_definition": "inep_official_dropout_rate",
        "seconds": round(time.time() - t0, 2),
        "feature_importance_top": importance[:10],
    }
    write_json(meta, out / "metrics.json")
    write_json(
        {
            "model_version": MODEL_VERSION,
            "level": level,
            "selected_model": winner,
            "beats_trivial_baseline": beats_baseline_cv and beats_baseline_test,
            "limitation": (
                "School-level Brazil INEP dropout-rate model; not student-level scoring."
            ),
        },
        out / "model_card_meta.json",
    )
    baseline_note = "OK beats baseline" if beats_baseline_test else "DOES NOT beat baseline"
    print(
        f"[{level}] selected={winner} test_mae={test_metrics['mae']:.3f} "
        f"r2={test_metrics['r2']:.3f} ({baseline_note}, baseline_mae={baseline_test_metrics['mae']:.3f})"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Brazil school dropout-rate models")
    parser.add_argument(
        "--level",
        choices=["fundamental", "medio", "both"],
        default="both",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional cap on training rows (random sample). Default: use the full mart "
            "— earlier versions silently capped this at 80,000, discarding ~90%% of the "
            "Fundamental mart."
        ),
    )
    args = parser.parse_args(argv)
    levels = ["fundamental", "medio"] if args.level == "both" else [args.level]
    for level in levels:
        train_level(level, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
