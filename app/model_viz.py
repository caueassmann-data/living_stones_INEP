"""Shared visualization helpers for model-results Streamlit and PNG export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import PROJECT_ROOT, model_dir

FEATURE_LABELS = {
    "is_rural": "Rural school",
    "is_public": "Public network",
    "in_internet": "Has internet",
    "in_lab_info": "Has computer lab",
    "in_quadra": "Has sports court",
    "in_biblioteca": "Has library",
    "in_agua": "Has water",
    "in_energia": "Has electricity",
    "in_esgoto": "Has sewage",
    "enrollment_level": "Enrollment (this level)",
    "qt_mat_bas": "Basic-ed enrollment",
    "qt_doc_bas": "Number of teachers",
    "student_teacher_ratio": "Students per teacher",
    "tp_dependencia": "Admin network code",
    "tp_localizacao": "Urban/rural code",
    "target_dropout_rate": "Official abandonment (%)",
}

LEVEL_TITLE = {
    "fundamental": "Ensino Fundamental",
    "medio": "Ensino Médio",
}


def label_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


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
    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No CV leaderboard", ha="center", va="center")
    else:
        colors = []
        winner = metrics.get("selected_model")
        for m in df["model"]:
            colors.append("#1f4e79" if m == winner else "#9db4c8")
        ax.barh(df["model"], df["cv_mae_mean"], color=colors, xerr=df.get("cv_mae_std"))
        ax.set_xlabel("Cross-validation MAE (lower is better)")
        ax.set_title(f"{LEVEL_TITLE.get(level, level)} — why this model won (5-fold CV)")
        ax.invert_yaxis()
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
    labels = ["MAE", "RMSE", "R²"]
    values = [tm.get("mae", 0), tm.get("rmse", 0), tm.get("r2", 0)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#1f4e79", "#9c2a2a", "#2a9d8f"])
    ax.set_title(
        f"{LEVEL_TITLE.get(level, level)} — hold-out metrics "
        f"({metrics.get('selected_model', 'n/a')})"
    )
    ax.set_ylabel("Value")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def fig_uf_risk(scored: pd.DataFrame, level: str, top_n: int = 12):
    g = (
        scored.groupby("uf", as_index=False)
        .agg(
            n=("pred_taxa_abandono", "size"),
            mean_pred=("pred_taxa_abandono", "mean"),
            mean_actual=("target_dropout_rate", "mean"),
        )
        .sort_values("mean_pred", ascending=False)
    )
    g = g.loc[g["n"] >= 20].head(top_n)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(g["uf"].astype(str), g["mean_pred"], color="#9c2a2a")
    ax.set_xlabel("State (UF)")
    ax.set_ylabel("Mean predicted abandonment (%)")
    ax.set_title(
        f"{LEVEL_TITLE.get(level, level)} — higher-risk states "
        "(mean predicted rate, n≥20 in sample)"
    )
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
