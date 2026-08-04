"""Streamlit — Model Results Lab (closing visual for prediction models)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.model_viz import (
    LEVEL_TITLE,
    fig_cv_leaderboard,
    fig_feature_importance,
    fig_metrics_summary,
    fig_uf_risk,
    label_feature,
    load_importance,
    load_metrics,
)
from src.inference import predict_frame
from src.utils import LIMITATION_STATEMENT, mart_path

st.set_page_config(
    page_title="Brazil School Risk — Model Results",
    layout="wide",
)

st.title("Brazil School Risk — Model Results Lab")
st.caption(LIMITATION_STATEMENT)
st.info(
    "This app is about **school-level** predicted abandonment rates and risk drivers — "
    "not individual student dropout probability."
)

mode = st.sidebar.radio(
    "View",
    ["Fundamental", "Médio", "Compare both"],
    index=0,
)


def _safe_score_sample(level: str, n: int = 8000) -> pd.DataFrame | None:
    path = mart_path(level)
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        st.warning(f"Mart not found for {level}: {path}")
        return None
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    try:
        return predict_frame(level, df)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None


def render_level(level: str) -> None:
    title = LEVEL_TITLE.get(level, level)
    st.header(title)

    try:
        metrics = load_metrics(level)
        imp = load_importance(level)
    except FileNotFoundError as exc:
        st.error(f"Missing model artifacts for {level}. Train first. ({exc})")
        return

    tm = metrics.get("test_metrics") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", str(metrics.get("selected_model", "n/a")))
    c2.metric("Hold-out MAE", f"{tm.get('mae', float('nan')):.3f}")
    c3.metric("Hold-out RMSE", f"{tm.get('rmse', float('nan')):.3f}")
    c4.metric("Hold-out R²", f"{tm.get('r2', float('nan')):.3f}")
    st.caption(
        f"Trained on n={metrics.get('n_rows')} school-year rows "
        f"(train {metrics.get('n_train')} / test {metrics.get('n_test')}). "
        f"Target: official INEP abandonment rate (%)."
    )

    tabs = st.tabs(
        [
            "1. Metrics",
            "2. Why this model (CV)",
            "3. School drivers",
            "4. Risk by state (UF)",
        ]
    )

    with tabs[0]:
        st.subheader("Hold-out performance")
        st.write(
            "MAE is the average absolute error in **percentage points** of abandonment rate. "
            "Lower is better. R² is modest because school structural features only partially "
            "explain abandonment — the tool is for **triage**, not perfect forecasting."
        )
        st.pyplot(fig_metrics_summary(metrics, level), clear_figure=True)
        st.json(tm)

    with tabs[1]:
        st.subheader("Model selection via 5-fold CV MAE")
        st.write(
            "We compared Ridge, Random Forest, and XGBoost with fixed hyperparameters. "
            "The highlighted bar is the winner (lowest CV MAE), then refit on the train set."
        )
        st.pyplot(fig_cv_leaderboard(metrics, level), clear_figure=True)
        st.dataframe(pd.DataFrame(metrics.get("cv_leaderboard") or []))

    with tabs[2]:
        st.subheader("Which school factors drive the predictions?")
        st.write(
            "These are **model importances** (not causal effects). "
            "They answer: when the model predicts a higher abandonment rate, "
            "which school characteristics does it rely on most?"
        )
        st.pyplot(fig_feature_importance(imp, level), clear_figure=True)
        show = imp.head(12).copy()
        show["feature_label"] = show["feature"].map(label_feature)
        st.dataframe(show[["feature_label", "importance"]])

    with tabs[3]:
        st.subheader("Where does predicted risk concentrate?")
        st.write(
            "We score a random sample of schools and average predicted abandonment by UF. "
            "Useful for public-network prioritization — not a final league table."
        )
        scored = _safe_score_sample(level)
        if scored is not None:
            fig, table = fig_uf_risk(scored, level)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(table.rename(
                columns={
                    "uf": "UF",
                    "n": "Schools in sample",
                    "mean_pred": "Mean predicted abandonment (%)",
                    "mean_actual": "Mean actual abandonment (%)",
                }
            ))


def render_compare() -> None:
    st.header("Fundamental vs Médio — why two models?")
    rows = []
    imps = {}
    for level in ("fundamental", "medio"):
        try:
            m = load_metrics(level)
            imps[level] = load_importance(level)
        except FileNotFoundError:
            st.error(f"Missing artifacts for {level}")
            return
        tm = m.get("test_metrics") or {}
        rows.append(
            {
                "Level": LEVEL_TITLE[level],
                "Selected model": m.get("selected_model"),
                "MAE": tm.get("mae"),
                "RMSE": tm.get("rmse"),
                "R²": tm.get("r2"),
            }
        )
    st.dataframe(pd.DataFrame(rows))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fundamental drivers")
        st.pyplot(fig_feature_importance(imps["fundamental"], "fundamental"), clear_figure=True)
    with col_b:
        st.subheader("Médio drivers")
        st.pyplot(fig_feature_importance(imps["medio"], "medio"), clear_figure=True)

    st.success(
        "Different winners and different top drivers support keeping **two** prediction "
        "products (Fundamental and Médio) instead of one pooled Brazil model."
    )


if mode == "Fundamental":
    render_level("fundamental")
elif mode == "Médio":
    render_level("medio")
else:
    render_compare()

st.divider()
st.caption(LIMITATION_STATEMENT)
st.caption(
    "Static PNGs for Slack: `python scripts/export_model_figures.py` "
    "→ `models/{level}/figures/*.png`"
)
