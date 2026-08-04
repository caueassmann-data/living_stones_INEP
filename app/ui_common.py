"""Shared Streamlit UI for the two Brazil school dropout-risk triage apps.

Both app/fundamental/main.py and app/medio/main.py just call run_app() with
their level; this file has the actual UI so the two apps cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import load_artifacts, predict_frame, score_single
from src.utils import LIMITATION_STATEMENT, mart_path

ADMIN_DEPENDENCY_OPTIONS = {"Federal": 1, "State": 2, "Municipal": 3, "Private": 4}
LOCATION_OPTIONS = {"Urban": 1, "Rural": 2}

# Brazil's 27 states (26 states + the Federal District), abbreviation -> name.
# Place names are kept in their official Portuguese spelling (standard
# practice — "Sao Paulo" is not translated in English-language text either).
BRAZIL_STATES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapa", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceara", "DF": "Distrito Federal", "ES": "Espirito Santo", "GO": "Goias",
    "MA": "Maranhao", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Para", "PB": "Paraiba", "PR": "Parana", "PE": "Pernambuco", "PI": "Piaui",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
    "RO": "Rondonia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "Sao Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}

NUMERIC_DEFAULTS = {
    "is_rural": 0.0,
    "is_public": 1.0,
    "has_water": 1.0,
    "has_electricity": 1.0,
    "electricity_absent": 0.0,
    "has_sewage": 1.0,
    "sewage_absent": 0.0,
    "has_internet": 1.0,
    "has_internet_for_students": 1.0,
    "has_library": 1.0,
    "has_computer_lab": 0.0,
    "has_sports_court": 1.0,
    "enrollment_basic_ed": 300.0,
    "enrollment_level": 200.0,
    "teacher_count_basic_ed": 15.0,
    "student_teacher_ratio": 13.3,
    "avg_class_size": 25.0,
    "overage_enrollment_share": 0.1,
    "eja_enrollment_share": 0.0,
    "is_covid_year": 0.0,
    "dropout_rate_lag1": 1.0,
    "dropout_rate_lag2": 1.0,
    "dropout_rate_3yr_avg": 1.0,
    "dropout_rate_trend": 0.0,
    "has_history": 1.0,
    "approval_rate_lag1": 90.0,
    "failure_rate_lag1": 8.0,
    "municipal_dropout_rate_lag1": 1.5,
    "state_dropout_rate_lag1": 1.5,
}

@st.cache_resource(show_spinner="Loading trained model...")
def _cached_artifacts(level: str):
    return load_artifacts(level)


@st.cache_data(show_spinner="Scoring schools...")
def _cached_predict(level: str, df: pd.DataFrame) -> pd.DataFrame:
    return predict_frame(level, df)


@st.cache_data(show_spinner="Loading training mart...")
def _cached_mart(level: str) -> pd.DataFrame | None:
    path = mart_path(level)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None


def _uncertainty_caption(art: dict) -> str:
    tm = art["metrics"].get("test_metrics") or {}
    mae = tm.get("mae")
    if mae is None:
        return ""
    return (
        f"Typical error on held-out schools is about **+/-{mae:.1f} percentage points** "
        "(hold-out MAE). Treat this as a risk *ranking* signal, not a precise forecast."
    )


def run_app(level: str, title: str) -> None:
    try:
        # Only succeeds when this page is the top-level script (standalone
        # `streamlit run app/fundamental/main.py`). When run as a page inside
        # app/main.py's st.navigation(), the entrypoint already set this and
        # a second call would raise — that's fine, just skip it.
        st.set_page_config(page_title=title, layout="wide")
    except st.errors.StreamlitAPIException:
        pass
    st.title(title)
    st.caption(LIMITATION_STATEMENT)

    try:
        art = _cached_artifacts(level)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tabs = st.tabs(
        ["School triage (batch)", "Find a school", "Single school profiler", "Model insights"]
    )

    with tabs[0]:
        st.subheader("Batch school triage")
        st.write(
            "Upload a CSV with school feature columns (same schema as the training mart). "
            "The model predicts the official-style dropout rate (%)."
        )
        mart_df = _cached_mart(level)
        if mart_df is not None:
            sample_n = st.slider("Sample size from the training mart", 100, 5000, 500, step=100)
            if st.button("Score a sample of the training mart"):
                sample = mart_df.sample(n=min(sample_n, len(mart_df)), random_state=42)
                scored = _cached_predict(level, sample)
                show_cols = [
                    c
                    for c in [
                        "year",
                        "school_id",
                        "school_name",
                        "state_code",
                        "target_dropout_rate",
                        "pred_dropout_rate",
                        "enrollment_level",
                    ]
                    if c in scored.columns
                ]
                st.dataframe(scored[show_cols].sort_values("pred_dropout_rate", ascending=False))
                st.download_button(
                    "Download scored sample",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name=f"scored_{level}_sample.csv",
                    mime="text/csv",
                )
        uploaded = st.file_uploader("Or upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            scored = predict_frame(level, df)
            st.dataframe(scored.head(100))
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name=f"scored_{level}.csv",
                mime="text/csv",
            )

    with tabs[1]:
        st.subheader("Find a school and prioritize by predicted risk")
        mart_df = _cached_mart(level)
        if mart_df is None:
            st.warning(f"Training mart not found for {level}.")
        else:
            years_available = sorted(mart_df["year"].dropna().astype(int).unique().tolist())
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                query = st.text_input("Search by school or municipality name (optional)")
            with col2:
                state_filter = st.selectbox(
                    "State (UF)", ["All"] + sorted(mart_df["state_code"].dropna().unique().tolist())
                )
            with col3:
                year_filter = st.selectbox("Year", years_available, index=len(years_available) - 1)

            filtered = mart_df.loc[mart_df["year"] == year_filter].copy()
            if state_filter != "All":
                filtered = filtered.loc[filtered["state_code"] == state_filter]
            if query:
                q = query.strip().lower()
                name_cols = [c for c in ("school_name", "municipality_name") if c in filtered.columns]
                mask = pd.Series(False, index=filtered.index)
                for c in name_cols:
                    mask = mask | filtered[c].astype(str).str.lower().str.contains(q, na=False)
                filtered = filtered.loc[mask]

            st.caption(f"{len(filtered):,} schools match this filter (year {year_filter}).")
            if len(filtered) > 0:
                scored = _cached_predict(level, filtered.head(20000))
                top_n = st.slider("How many top-risk schools to show / export", 10, 500, 50, step=10)
                ranked = scored.sort_values("pred_dropout_rate", ascending=False).head(top_n)
                show_cols = [
                    c
                    for c in [
                        "school_id",
                        "school_name",
                        "municipality_name",
                        "state_code",
                        "pred_dropout_rate",
                        "target_dropout_rate",
                        "dropout_rate_lag1",
                        "enrollment_level",
                    ]
                    if c in ranked.columns
                ]
                st.dataframe(ranked[show_cols])
                st.download_button(
                    f"Download top {top_n} highest-risk schools (CSV)",
                    ranked[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"top_{top_n}_priority_schools_{level}_{year_filter}.csv",
                    mime="text/csv",
                )

    with tabs[2]:
        st.subheader("Single school profiler")
        st.caption(_uncertainty_caption(art))
        row: dict = {}

        st.markdown("**Location and network**")
        loc_cols = st.columns(3)
        with loc_cols[0]:
            state_label = st.selectbox(
                "State (UF)",
                [f"{code} - {name}" for code, name in BRAZIL_STATES.items()],
                index=list(BRAZIL_STATES.keys()).index("SP"),
                key=f"{level}_state_code",
            )
            row["state_code"] = state_label.split(" - ")[0]
        with loc_cols[1]:
            dep_label = st.selectbox(
                "Administrative network", list(ADMIN_DEPENDENCY_OPTIONS.keys()), index=2,
                key=f"{level}_admin_dependency_type",
            )
            row["admin_dependency_type"] = ADMIN_DEPENDENCY_OPTIONS[dep_label]
        with loc_cols[2]:
            loc_label = st.selectbox(
                "Location", list(LOCATION_OPTIONS.keys()), index=0, key=f"{level}_location_type"
            )
            row["location_type"] = LOCATION_OPTIONS[loc_label]
        row["is_rural"] = 1.0 if row["location_type"] == 2 else 0.0
        row["is_public"] = 1.0 if row["admin_dependency_type"] in (1, 2, 3) else 0.0

        st.markdown("**Infrastructure**")
        infra_features = [
            "has_water", "has_electricity", "has_sewage", "has_internet",
            "has_internet_for_students", "has_library", "has_computer_lab", "has_sports_court",
        ]
        infra_cols = st.columns(4)
        for i, feat in enumerate(infra_features):
            with infra_cols[i % 4]:
                row[feat] = 1.0 if st.checkbox(
                    feat.replace("_", " ").capitalize(), value=bool(NUMERIC_DEFAULTS[feat]),
                    key=f"{level}_{feat}",
                ) else 0.0
        row["electricity_absent"] = 1.0 - row["has_electricity"]
        row["sewage_absent"] = 1.0 - row["has_sewage"]

        st.markdown("**Enrollment and staffing**")
        num_features = [
            "enrollment_basic_ed", "enrollment_level", "teacher_count_basic_ed",
            "student_teacher_ratio", "avg_class_size", "overage_enrollment_share",
            "eja_enrollment_share",
        ]
        num_cols = st.columns(4)
        for i, feat in enumerate(num_features):
            with num_cols[i % 4]:
                row[feat] = st.number_input(
                    feat.replace("_", " ").capitalize(), value=float(NUMERIC_DEFAULTS[feat]),
                    key=f"{level}_{feat}",
                )

        st.markdown(
            "**Dropout-rate history** (the school's own track record — usually the single "
            "strongest signal available)"
        )
        hist_features = [
            "dropout_rate_lag1", "dropout_rate_lag2", "dropout_rate_3yr_avg",
            "approval_rate_lag1", "failure_rate_lag1", "municipal_dropout_rate_lag1",
            "state_dropout_rate_lag1",
        ]
        hist_cols = st.columns(4)
        for i, feat in enumerate(hist_features):
            with hist_cols[i % 4]:
                row[feat] = st.number_input(
                    feat.replace("_", " ").capitalize(), value=float(NUMERIC_DEFAULTS[feat]),
                    key=f"{level}_{feat}",
                )
        row["dropout_rate_trend"] = row["dropout_rate_lag1"] - row["dropout_rate_lag2"]
        row["has_history"] = 1.0
        row["is_covid_year"] = 0.0

        if st.button("Score school", type="primary"):
            result = score_single(level, row)
            mae = (art["metrics"].get("test_metrics") or {}).get("mae")
            pred = result["pred_dropout_rate"]
            if mae:
                st.metric(
                    "Predicted dropout rate (%)",
                    f"{pred:.2f}",
                    help=f"Hold-out MAE is +/-{mae:.1f} points — read this as roughly [{max(0, pred - mae):.1f}, {pred + mae:.1f}].",
                )
            else:
                st.metric("Predicted dropout rate (%)", f"{pred:.2f}")
            st.json(result)

    with tabs[3]:
        st.subheader("Hold-out metrics & drivers")
        tm = art["metrics"].get("test_metrics", {})
        bm = art["metrics"].get("baseline_test_metrics", {})
        beats = art["metrics"].get("beats_baseline_test")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Model**")
            st.json(tm)
        with c2:
            st.write("**Trivial baseline** (always predict the training mean)")
            st.json(bm)
        if beats is not None:
            if beats:
                st.success("The selected model beats the trivial baseline on held-out schools.")
            else:
                st.error(
                    "The selected model does NOT beat the trivial baseline on held-out schools. "
                    "Treat predictions with caution — see docs/validity_and_english_revision.md."
                )
        ranking = art["metrics"].get("test_ranking_metrics")
        if ranking:
            st.write("**Triage ranking quality** (top-decile lift over random selection)")
            st.json(ranking)
        st.write(f"Selected model: **{art['metrics'].get('selected_model', 'n/a')}**")
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / level
            / "figures"
            / "global_importance_top.csv"
        )
        if csv_path.exists():
            imp = pd.read_csv(csv_path)
            st.bar_chart(imp.set_index("feature")["importance"].head(12))
            st.dataframe(imp.head(15))
        st.info(
            "Compare Fundamental vs Medio drivers using notebook "
            "`notebooks/04_compare_fundamental_vs_medio.ipynb`."
        )

    st.divider()
    st.caption(
        f"model_version={art['model_version']} | level={level} | {LIMITATION_STATEMENT}"
    )
