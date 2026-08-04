"""Shared visualization helpers for the model-results Streamlit app and PNG export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import model_dir

FEATURE_LABELS = {
    "is_rural": "Rural school",
    "is_public": "Public network",
    "has_internet": "Has internet",
    "has_internet_for_students": "Has internet for students",
    "has_computer_lab": "Has computer lab",
    "has_sports_court": "Has sports court",
    "has_library": "Has library",
    "has_water": "Has running water",
    "has_electricity": "Has electricity",
    "electricity_absent": "No electricity at all",
    "has_sewage": "Has sewage system",
    "sewage_absent": "No sewage system at all",
    "enrollment_level": "Enrollment (this level)",
    "enrollment_basic_ed": "Total basic-ed enrollment",
    "teacher_count_basic_ed": "Number of teachers",
    "student_teacher_ratio": "Students per teacher",
    "avg_class_size": "Average class size",
    "overage_enrollment_share": "Share of over-age students",
    "eja_enrollment_share": "Share of adult-education (EJA) students",
    "is_covid_year": "COVID-affected school year (2020-2021)",
    "admin_dependency_type": "Administrative network (code)",
    "location_type": "Urban/rural (code)",
    "state_code": "State (UF)",
    "target_dropout_rate": "Official dropout rate (%)",
    "dropout_rate_lag1": "Dropout rate, prior year",
    "dropout_rate_lag2": "Dropout rate, 2 years prior",
    "dropout_rate_3yr_avg": "Dropout rate, 3-year trailing average",
    "dropout_rate_trend": "Dropout rate trend (prior year vs 2 years prior)",
    "has_history": "School has prior-year data",
    "approval_rate_lag1": "Approval rate, prior year",
    "failure_rate_lag1": "Failure/repetition rate, prior year",
    "municipal_dropout_rate_lag1": "Municipality's average dropout rate, prior year",
    "state_dropout_rate_lag1": "State's average dropout rate, prior year",
}

LEVEL_TITLE = {
    "fundamental": "Ensino Fundamental",
    "medio": "Ensino Medio",
}


def label_feature(name: str) -> str:
    """Translate a raw one-hot-encoded or base feature name into a readable label."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    # One-hot encoded categorical columns look like "state_code_SP" or
    # "admin_dependency_type_3.0"; fall back to a readable split.
    for prefix, readable in (
        ("state_code_", "State: "),
        ("admin_dependency_type_", "Admin network code "),
        ("location_type_", "Location code "),
    ):
        if name.startswith(prefix):
            return readable + name[len(prefix) :]
    return name


def load_metrics(level: str) -> dict[str, Any]:
    path = model_dir(level) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_importance(level: str) -> pd.DataFrame:
    path = model_dir(level) / "figures" / "global_importance_top.csv"
    df = pd.read_csv(path)
    df["feature_label"] = df["feature"].map(label_feature)
    return df


def fig_cv_leaderboard(metrics: dict[str, Any], level: str):
    rows = metrics.get("cv_leaderboard") or []
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No CV leaderboard", ha="center", va="center")
    else:
        winner = metrics.get("selected_model")
        colors = []
        for _, r in df.iterrows():
            if r["model"] == winner:
                colors.append("#1f4e79")
            elif r.get("is_baseline"):
                colors.append("#c65911")  # baselines stand out distinctly
            else:
                colors.append("#9db4c8")
        ax.barh(df["model"], df["cv_mae_mean"], color=colors, xerr=df.get("cv_mae_std"))
        ax.set_xlabel("Cross-validation MAE (lower is better)")
        ax.set_title(f"{LEVEL_TITLE.get(level, level)} — why this model won (5-fold, grouped by school)")
        ax.invert_yaxis()
        beats = metrics.get("beats_baseline_cv")
        if beats is not None:
            note = "beats the trivial baseline" if beats else "DOES NOT beat the trivial baseline"
            ax.text(
                0.98,
                0.02,
                f"Orange bars = trivial baselines (predict the mean/median). Winner {note}.",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#c65911" if not beats else "#555555",
            )
    fig.tight_layout()
    return fig


def fig_feature_importance(imp: pd.DataFrame, level: str, top_n: int = 10):
    df = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature_label"], df["importance"], color="#2a9d8f")
    ax.set_xlabel("Importance (model relies on this more →)")
    ax.set_title(f"{LEVEL_TITLE.get(level, level)} — school factors driving predictions")
    fig.tight_layout()
    return fig


def fig_metrics_summary(metrics: dict[str, Any], level: str):
    tm = metrics.get("test_metrics") or {}
    bm = metrics.get("baseline_test_metrics") or {}
    labels = ["MAE", "RMSE", "R2"]
    values = [tm.get("mae", 0), tm.get("rmse", 0), tm.get("r2", 0)]
    baseline_values = [bm.get("mae", 0), bm.get("rmse", 0), None]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], values, width=width, color="#1f4e79", label="Model")
    ax.bar(
        [i + width / 2 for i in x],
        [v if v is not None else 0 for v in baseline_values],
        width=width,
        color="#c65911",
        label="Trivial baseline (predict the mean)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(
        f"{LEVEL_TITLE.get(level, level)} — hold-out metrics vs. baseline "
        f"({metrics.get('selected_model', 'n/a')})"
    )
    ax.set_ylabel("Value (MAE/RMSE in dropout-rate percentage points)")
    ax.legend(fontsize=8)
    for i, v in enumerate(values):
        ax.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(baseline_values):
        if v is not None:
            ax.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def fig_uf_risk(scored: pd.DataFrame, level: str, top_n: int = 12):
    """Predicted vs. actual mean dropout rate by state (UF).

    Earlier versions of this chart only plotted the predicted mean, even
    though the actual mean was already computed — which hid exactly the gap
    a reviewer would want to see. Both bars are shown here side by side.
    """
    g = (
        scored.groupby("state_code", as_index=False)
        .agg(
            n=("pred_dropout_rate", "size"),
            mean_pred=("pred_dropout_rate", "mean"),
            mean_actual=("target_dropout_rate", "mean"),
        )
        .sort_values("mean_pred", ascending=False)
    )
    g = g.loc[g["n"] >= 20].head(top_n)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(g))
    width = 0.35
    ax.bar([i - width / 2 for i in x], g["mean_pred"], width=width, color="#9c2a2a", label="Predicted")
    ax.bar([i + width / 2 for i in x], g["mean_actual"], width=width, color="#4a4a4a", label="Actual")
    ax.set_xticks(list(x))
    ax.set_xticklabels(g["state_code"].astype(str))
    ax.set_xlabel("State (UF)")
    ax.set_ylabel("Mean dropout rate (%)")
    ax.set_title(
        f"{LEVEL_TITLE.get(level, level)} — predicted vs. actual by state "
        "(mean over a sampled set of schools, n>=20 in sample)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, g


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def export_level_figures(level: str) -> list[Path]:
    """Write PNG figures for one education level under models/{level}/figures/."""
    metrics = load_metrics(level)
    imp = load_importance(level)
    out_dir = model_dir(level) / "figures"
    written = [
        save_fig(fig_metrics_summary(metrics, level), out_dir / "metrics_summary.png"),
        save_fig(fig_cv_leaderboard(metrics, level), out_dir / "cv_leaderboard.png"),
        save_fig(fig_feature_importance(imp, level), out_dir / "feature_importance.png"),
    ]
    return written
