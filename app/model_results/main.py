"""Streamlit — Model Results Lab (closing visual for the prediction models)."""

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

try:
    # See app/ui_common.py run_app() for why this is wrapped: it only
    # succeeds when this page is the top-level script.
    st.set_page_config(page_title="Brazil School Risk — Model Results", layout="wide")
except st.errors.StreamlitAPIException:
    pass

st.title("Brazil School Risk — Model Results Lab")
st.caption(LIMITATION_STATEMENT)
st.info(
    "This app is about **school-level** predicted dropout rates and risk drivers — "
    "not individual student dropout probability."
)

mode = st.sidebar.radio(
    "View",
    ["Fundamental", "Medio", "Compare both"],
    index=0,
)


@st.cache_data(show_spinner="Scoring a sample of schools...")
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
    bm = metrics.get("baseline_test_metrics") or {}
    beats = metrics.get("beats_baseline_test")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected model", str(metrics.get("selected_model", "n/a")))
    c2.metric("Hold-out MAE", f"{tm.get('mae', float('nan')):.3f}", help="Model error, in dropout-rate percentage points.")
    c3.metric("Hold-out RMSE", f"{tm.get('rmse', float('nan')):.3f}")
    c4.metric("Hold-out R2", f"{tm.get('r2', float('nan')):.3f}")
    c5.metric(
        "Beats trivial baseline?",
        "Yes" if beats else "No",
        delta=f"baseline MAE {bm.get('mae', float('nan')):.3f}",
        delta_color="normal" if beats else "inverse",
    )
    st.caption(
        f"Trained on n={metrics.get('n_rows')} school-year rows "
        f"(train {metrics.get('n_train')} / test {metrics.get('n_test')}, held out by "
        f"**school**, not by row). Target: official INEP dropout rate (%)."
    )
    if not beats:
        st.error(
            "This model does not beat the trivial 'always predict the training mean' "
            "baseline on held-out schools. Treat its predictions as exploratory only. "
            "See docs/validity_and_english_revision.md for the full analysis."
        )

    temporal = metrics.get("temporal_holdout")
    tabs = st.tabs(
        [
            "1. Metrics",
            "2. Why this model (CV)",
            "3. School drivers",
            "4. Risk by state (UF)",
            "5. Future-year holdout",
            "6. Equity",
        ]
    )

    with tabs[0]:
        st.subheader("Hold-out performance vs. trivial baseline")
        st.write(
            "MAE is the average absolute error in **percentage points** of dropout rate. "
            "Lower is better. The baseline is a model that ignores every feature and always "
            "predicts the training-set average — any model that cannot beat it is not adding "
            "value. R2 is modest because school structural features only partially explain "
            "dropout — the tool is for **triage**, not precise forecasting."
        )
        st.pyplot(fig_metrics_summary(metrics, level), clear_figure=True)
        col_a, col_b = st.columns(2)
        col_a.json(tm)
        col_b.json(bm)
        ranking = metrics.get("test_ranking_metrics")
        if ranking:
            st.subheader("Triage ranking quality")
            st.write(
                "If we sort schools by predicted risk and act on the top decile, how many of "
                "the truly highest-risk schools (the mart's `high_risk` label) do we catch? "
                "`lift` above 1.0 means the ranking beats picking schools at random."
            )
            st.json(ranking)

    with tabs[1]:
        st.subheader("Model selection via 5-fold CV MAE (grouped by school)")
        st.write(
            "We compare two trivial baselines (predict the mean/median) against Ridge, "
            "Random Forest, and XGBoost. Folds are grouped by `school_id` so the same school "
            "never appears in both the training and validation side of a fold. The highlighted "
            "blue bar is the winner among the real candidates; orange bars are the baselines."
        )
        st.pyplot(fig_cv_leaderboard(metrics, level), clear_figure=True)
        st.dataframe(pd.DataFrame(metrics.get("cv_leaderboard") or []))

    with tabs[2]:
        st.subheader("Which school factors drive the predictions?")
        st.write(
            "These are **model importances** (not causal effects). "
            "They answer: when the model predicts a higher dropout rate, "
            "which school characteristics does it rely on most?"
        )
        st.pyplot(fig_feature_importance(imp, level), clear_figure=True)
        show = imp.head(12).copy()
        show["feature_label"] = show["feature"].map(label_feature)
        st.dataframe(show[["feature_label", "importance"]])

    with tabs[3]:
        st.subheader("Where does predicted risk concentrate?")
        st.write(
            "We score a random sample of schools and average predicted dropout rate by state "
            "(UF), shown next to the actual average so any systematic over/under-prediction is "
            "visible directly. Useful for public-network prioritization — not a final league table."
        )
        scored = _safe_score_sample(level)
        if scored is not None:
            fig, table = fig_uf_risk(scored, level)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(
                table.rename(
                    columns={
                        "state_code": "State (UF)",
                        "n": "Schools in sample",
                        "mean_pred": "Mean predicted dropout rate (%)",
                        "mean_actual": "Mean actual dropout rate (%)",
                    }
                )
            )

    with tabs[4]:
        st.subheader("Out-of-time check: would this have worked on years it never saw?")
        if not temporal or not temporal.get("metrics"):
            st.info("Not enough rows in the recent years to compute a temporal holdout.")
        else:
            st.write(temporal["description"])
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.write(f"**{metrics.get('selected_model')}** (trained on years before the holdout)")
                st.json(temporal["metrics"])
            with tcol2:
                st.write("**Trivial baseline**, same split")
                st.json(temporal.get("baseline_metrics") or {})
            model_mae = temporal["metrics"].get("mae")
            baseline_mae = (temporal.get("baseline_metrics") or {}).get("mae")
            if model_mae is not None and baseline_mae is not None:
                if model_mae < baseline_mae:
                    st.success("The model beats the trivial baseline on unseen future years too.")
                else:
                    st.error(
                        "The model does NOT beat the trivial baseline on unseen future years — "
                        "the strongest and most realistic test for an early-warning tool."
                    )

    with tabs[5]:
        st.subheader("Is the error evenly distributed across school types?")
        st.write(
            "A model with a good **average** error can still be systematically worse for one "
            "group of schools — and that is exactly the scenario `docs/model_card.md` warns "
            "against using punitively. `mean_error` is **signed** (predicted minus actual): a "
            "value near zero with a high MAE means the model is noisy but unbiased for that "
            "group; a mean_error far from zero means it is consistently too high or too low for "
            "that group specifically. Groups with too few schools to be meaningful are omitted."
        )
        subgroup_metrics = metrics.get("subgroup_metrics") or {}
        if not subgroup_metrics:
            st.info("No subgroup breakdown available — retrain with the current src/train.py to generate it.")
        else:
            dimension_titles = {
                "location": "Rural vs. urban",
                "admin_network": "Public vs. private",
                "school_size": "School size",
            }
            for dim, rows in subgroup_metrics.items():
                if not rows:
                    continue
                st.write(f"**{dimension_titles.get(dim, dim)}**")
                table = pd.DataFrame(rows).rename(
                    columns={
                        "group": "Group",
                        "n": "Schools in holdout",
                        "mae": "MAE",
                        "rmse": "RMSE",
                        "mean_error": "Mean error (signed: predicted - actual)",
                    }
                )
                st.dataframe(table)


def render_compare() -> None:
    st.header("Fundamental vs Medio — why two models?")
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
                "R2": tm.get("r2"),
                "Beats baseline": m.get("beats_baseline_test"),
            }
        )
    st.dataframe(pd.DataFrame(rows))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fundamental drivers")
        st.pyplot(fig_feature_importance(imps["fundamental"], "fundamental"), clear_figure=True)
    with col_b:
        st.subheader("Medio drivers")
        st.pyplot(fig_feature_importance(imps["medio"], "medio"), clear_figure=True)

    st.success(
        "Different winners and different top drivers support keeping **two** prediction "
        "products (Fundamental and Medio) instead of one pooled Brazil model."
    )


if mode == "Fundamental":
    render_level("fundamental")
elif mode == "Medio":
    render_level("medio")
else:
    render_compare()

st.divider()
st.caption(LIMITATION_STATEMENT)
st.caption(
    "Static PNGs for Slack: `python scripts/export_model_figures.py` "
    "-> `models/{level}/figures/*.png`"
)
