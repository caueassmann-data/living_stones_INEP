"""Streamlit — Model Results Lab (closing visual for the prediction models)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.metric_views import (
    render_drivers,
    render_error_comparison,
    render_model_comparison,
    render_ranking_quality,
    render_state_comparison,
    render_technical_details,
    render_within_network,
)
from app.model_viz import (
    LEVEL_TITLE,
    fig_uf_risk,
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
    # Three headline answers a non-technical reader asks first. The detailed
    # error numbers live in the comparison cards further down.
    model_name = str(metrics.get("selected_model", "n/a")).replace("_", " ").title()
    c1, c2, c3 = st.columns(3)
    c1.metric("Model used", model_name, help="The algorithm that performed best in testing.")
    c2.metric(
        "Better than guessing the average?",
        "Yes" if beats else "No",
        help=(
            "The comparison point is a model that ignores everything about the school and "
            "always predicts the average dropout rate. A model that cannot beat that adds "
            "no value."
        ),
    )
    c3.metric(
        "Schools tested",
        f"{metrics.get('n_test', 0):,}",
        help=(
            "Schools the model never saw while learning. Testing on unseen schools is what "
            "makes the results trustworthy."
        ),
    )
    st.caption(
        f"Learned from {metrics.get('n_train', 0):,} school-year records and tested on "
        f"{metrics.get('n_test', 0):,} others, kept apart **by school** so the model never "
        "saw a school it was graded on. What it predicts: each school's official INEP "
        "dropout rate (%)."
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
        st.subheader("How far off is the model?")
        st.write(
            "Errors are in **percentage points** of dropout rate. The comparison point is a "
            "“do-nothing” model that ignores everything about the school and always "
            "predicts the average. The model explains only part of why dropout differs between "
            "schools, so it is a tool for deciding **where to look first**, not for forecasting "
            "an exact number."
        )
        render_error_comparison(tm, bm)
        render_technical_details("model vs baseline", {"model": tm, "baseline": bm})

        ranking = metrics.get("test_ranking_metrics")
        if ranking:
            st.subheader("Does sorting by predicted risk find the right schools?")
            render_ranking_quality(ranking, key=f"{level}_lab")
            render_technical_details("ranking metrics", ranking)

        within = metrics.get("test_ranking_within_network")
        if within:
            st.subheader("Ranking inside one network (how the tool is used)")
            render_within_network(within, key=f"{level}_lab")
            render_technical_details("within-network metrics", within)

    with tabs[1]:
        st.subheader("Why this model?")
        st.write(
            "Several approaches were tried against two “do-nothing” options (always "
            "guess the average, or the median). Each was graded five times on schools it had "
            "not learned from, and the chart shows its typical error. The same school never "
            "appears on both the learning and grading side."
        )
        render_model_comparison(
            metrics.get("cv_leaderboard") or [], metrics.get("selected_model"), key=f"{level}_lab"
        )
        render_technical_details("model comparison", metrics.get("cv_leaderboard"))

    with tabs[2]:
        st.subheader("Which school factors drive the predictions?")
        st.write(
            "When the model predicts a higher dropout rate, these are the school "
            "characteristics it leans on most. They describe what the model *uses*, not what "
            "*causes* dropout."
        )
        render_drivers(imp, key=f"{level}_lab", top=12)

    with tabs[3]:
        st.subheader("Where does predicted risk concentrate?")
        st.write(
            "We score a random sample of schools and average predicted dropout rate by state "
            "(UF), shown next to the actual average so any systematic over/under-prediction is "
            "visible directly. Useful for public-network prioritization — not a final league table."
        )
        scored = _safe_score_sample(level)
        if scored is not None:
            _, table = fig_uf_risk(scored, level)
            render_state_comparison(table, key=f"{level}_lab")

    with tabs[4]:
        st.subheader("Out-of-time check: would this have worked on years it never saw?")
        if not temporal or not temporal.get("metrics"):
            st.info("Not enough rows in the recent years to compute a temporal holdout.")
        else:
            st.write(temporal["description"])
            render_error_comparison(
                temporal["metrics"],
                temporal.get("baseline_metrics") or {},
                baseline_caption="always predicting the average school, on the same future years",
            )
            t_ranking = (temporal["metrics"] or {}).get("ranking")
            if t_ranking:
                st.write("**Does the ranking still find the right schools on future years?**")
                render_ranking_quality(t_ranking, key=f"{level}_lab_temporal")
            t_within = (temporal["metrics"] or {}).get("ranking_within_network")
            if t_within:
                st.write("**And inside one network, on future years**")
                render_within_network(t_within, key=f"{level}_lab_temporal")
            render_technical_details(
                "future-year holdout",
                {"model": temporal["metrics"], "baseline": temporal.get("baseline_metrics")},
            )

    with tabs[5]:
        st.subheader("Is the error evenly distributed across school types?")
        st.write(
            "A model that is right on average can still be worse for one kind of school — "
            "and that is the risk if a list like this were used to judge or penalize schools. "
            "Two columns matter below. **Typical error** is how far off the model is for that "
            "group. **Leans too high (+) or too low (−)** shows whether the misses go "
            "consistently one way: near zero means the model is noisy but fair to that group; "
            "far from zero means it systematically over- or under-predicts for them. Groups too "
            "small to be meaningful are left out."
        )
        subgroup_metrics = metrics.get("subgroup_metrics") or {}
        if not subgroup_metrics:
            st.info("No breakdown by school type is available for this model yet.")
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
                        "n": "Schools tested",
                        "mae": "Typical error (points)",
                        "mean_error": "Leans too high (+) or too low (−), in points",
                    }
                ).drop(columns=["rmse"], errors="ignore")
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Schools tested": st.column_config.NumberColumn(format="localized"),
                        "Typical error (points)": st.column_config.NumberColumn(format="%.2f"),
                        "Leans too high (+) or too low (−), in points": (
                            st.column_config.NumberColumn(format="%+.2f")
                        ),
                    },
                )


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
                "Model used": str(m.get("selected_model", "n/a")).replace("_", " ").title(),
                "Typical error (points)": tm.get("mae"),
                "Differences between schools explained": tm.get("r2"),
                "Better than guessing the average?": "Yes" if m.get("beats_baseline_test") else "No",
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Typical error (points)": st.column_config.NumberColumn(format="%.2f"),
            "Differences between schools explained": st.column_config.NumberColumn(
                format="percent"
            ),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fundamental drivers")
        render_drivers(imps["fundamental"], key="compare_fundamental", top=8)
    with col_b:
        st.subheader("Medio drivers")
        render_drivers(imps["medio"], key="compare_medio", top=8)

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
