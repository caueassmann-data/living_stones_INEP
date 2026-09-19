"""Shared Streamlit UI for the two Brazil school dropout-risk triage apps.

Both app/fundamental/main.py and app/medio/main.py just call run_app() with
their level; this file has the actual UI so the two apps cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import load_artifacts, predict_frame, score_single
from src.prioritize import (
    ADMIN_DEPENDENCY_LABELS,
    DEFAULT_SCOPE,
    DEFAULT_TOP_N,
    MAX_TOP_N,
    MIN_TOP_N,
    PUBLIC_ADMIN_DEPENDENCY_TYPES,
    SCOPE_DISPLAY_NAME,
    TOP_N_PRESETS,
    coverage_message,
    low_signal_message,
    prioritize,
)
from src.utils import LIMITATION_STATEMENT, resolve_app_mart_path

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
    # Full mart when it exists (local development, every year available),
    # otherwise the slim committed copy a deployment ships with.
    path = resolve_app_mart_path(level)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Scoring every school for this year...")
def _cached_scored_year(level: str, year: int) -> pd.DataFrame:
    """Score one whole year once, then filter the already-scored frame.

    Keyed on (level, year) rather than on a DataFrame: @st.cache_data hashes
    its arguments, and hashing a 100k-row frame on every widget interaction is
    what made the old code cap the input at 20,000 rows before ranking — which
    silently meant a "top 50" was the top 50 of whatever happened to come first
    in mart order. Scoring the full year once is both correct and faster.
    """
    mart = _cached_mart(level)
    if mart is None:
        return pd.DataFrame()
    slice_ = mart.loc[pd.to_numeric(mart["year"], errors="coerce") == int(year)].copy()
    return predict_frame(level, slice_)


def _uncertainty_caption(art: dict) -> str:
    tm = art["metrics"].get("test_metrics") or {}
    mae = tm.get("mae")
    if mae is None:
        return ""
    return (
        f"Typical error on held-out schools is about **+/-{mae:.1f} percentage points** "
        "(hold-out MAE). Treat this as a risk *ranking* signal, not a precise forecast."
    )


def _render_prioritization_tab(level: str, art: dict) -> None:
    """Top-N prioritization inside one education network.

    This is the tool's main output for a field team. The rule itself lives in
    src/prioritize.py; everything here is selection widgets and rendering, so
    the rule can be tested (tests/test_prioritize.py) without Streamlit.

    Note the wording throughout: the user picks how many schools their team can
    follow up on, and gets that many. The mart's `high_risk` label is not shown
    here — the Foundation kept it as a historical/evaluation label, reported in
    the model-results app instead.
    """
    st.subheader("Prioritized school list (Top-N within a network)")
    st.caption(
        "A prioritization of where to look first, not a verdict on a school. "
        "Set N to the number of schools your team can realistically follow up on."
    )

    mart_df = _cached_mart(level)
    if mart_df is None:
        st.warning(f"Training mart not found for {level}.")
        return

    years_available = sorted(mart_df["year"].dropna().astype(int).unique().tolist())

    row1 = st.columns([1, 2, 2])
    with row1[0]:
        year_filter = st.selectbox(
            "Year", years_available, index=len(years_available) - 1, key=f"{level}_pri_year"
        )
    with row1[1]:
        scope = st.selectbox(
            "Prioritize within",
            list(SCOPE_DISPLAY_NAME.keys()),
            index=list(SCOPE_DISPLAY_NAME.keys()).index(DEFAULT_SCOPE),
            format_func=lambda s: SCOPE_DISPLAY_NAME[s],
            key=f"{level}_pri_scope",
            help=(
                "The pool a school competes inside. A municipal network is often very "
                "small (the median one has a handful of schools), so the default is one "
                "administrative network within one state."
            ),
        )
    with row1[2]:
        # Derived from the mart, not hard-coded, so a level with no Federal
        # schools does not offer an empty filter.
        available_types = sorted(
            pd.to_numeric(mart_df["admin_dependency_type"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        default_types = [t for t in available_types if t in PUBLIC_ADMIN_DEPENDENCY_TYPES]
        admin_types = st.multiselect(
            "Administrative network",
            available_types,
            default=default_types,
            format_func=lambda t: ADMIN_DEPENDENCY_LABELS.get(t, str(t)),
            key=f"{level}_pri_admin",
        )

    year_slice = mart_df.loc[pd.to_numeric(mart_df["year"], errors="coerce") == int(year_filter)]

    row2 = st.columns([1, 3])
    with row2[0]:
        state_options = sorted(year_slice["state_code"].dropna().unique().tolist())
        state_filter = st.selectbox(
            "State (UF)", ["All Brazil"] + state_options, key=f"{level}_pri_state"
        )
    with row2[1]:
        if state_filter == "All Brazil":
            municipality_ids: list = []
            st.multiselect(
                "Municipality",
                [],
                disabled=True,
                key=f"{level}_pri_mun_disabled",
                help="Pick a state first to choose municipalities.",
            )
        else:
            in_state = year_slice.loc[year_slice["state_code"] == state_filter]
            options = (
                in_state[["municipality_id", "municipality_name"]]
                .drop_duplicates()
                .sort_values("municipality_name")
            )
            labels = {
                int(r.municipality_id): f"{r.municipality_name}"
                for r in options.itertuples(index=False)
            }
            municipality_ids = st.multiselect(
                "Municipality (leave empty for the whole state)",
                list(labels.keys()),
                format_func=lambda m: labels.get(m, str(m)),
                key=f"{level}_pri_mun",
            )

    row3 = st.columns([2, 2])
    with row3[0]:
        n_choice = st.radio(
            "How many schools can your team follow up on?",
            [*TOP_N_PRESETS, "Custom"],
            index=TOP_N_PRESETS.index(DEFAULT_TOP_N),
            horizontal=True,
            key=f"{level}_pri_n_choice",
        )
        top_n = (
            st.number_input(
                "Custom N",
                min_value=MIN_TOP_N,
                max_value=MAX_TOP_N,
                value=DEFAULT_TOP_N,
                step=5,
                key=f"{level}_pri_n_custom",
            )
            if n_choice == "Custom"
            else int(n_choice)
        )
    with row3[1]:
        require_history = st.checkbox(
            "Only schools with at least one year of dropout history",
            value=False,
            key=f"{level}_pri_history",
            help=(
                "A school with no track record is predicted almost entirely from Census "
                "features. That matters more under Top-N than under a threshold, because "
                "such a school can land at the very top of a short list."
            ),
        )

    scored_year = _cached_scored_year(level, int(year_filter))
    if scored_year.empty:
        st.warning(f"No rows for {year_filter} in the {level} mart.")
        return

    result = prioritize(
        level,
        df=scored_year,
        year=int(year_filter),
        scope=scope,
        top_n=int(top_n),
        state_codes=[state_filter] if state_filter != "All Brazil" else None,
        municipality_ids=municipality_ids or None,
        admin_dependency_types=admin_types or None,
        require_history=require_history,
    )

    if result.pool_size == 0:
        st.warning("No schools match this selection. Try widening the filters.")
        return

    cols = st.columns(4)
    cols[0].metric("Schools in scope", f"{result.pool_size:,}")
    cols[1].metric("Networks covered", f"{result.meta['n_scopes']:,}")
    cols[2].metric("Schools prioritized", f"{result.n_selected:,}")
    pool_mean = float(result.scope_summary["mean_pred_pool"].mean())
    sel_mean = float(result.frame["pred_dropout_rate"].mean())
    cols[3].metric(
        "Mean predicted dropout in list",
        f"{sel_mean:.2f}%",
        delta=f"{sel_mean - pool_mean:+.2f} pp vs pool",
    )

    message = coverage_message(result)
    if message:
        st.info(message)
    weak = low_signal_message(result)
    if weak:
        st.warning(weak)

    st.caption(_uncertainty_caption(art))
    st.dataframe(result.frame, width="stretch", hide_index=True)
    st.download_button(
        f"Download this prioritized list ({result.n_selected} schools, CSV)",
        result.to_csv_bytes(),
        file_name=result.filename(level),
        mime="text/csv",
    )

    with st.expander("Per-network detail (pool size vs schools selected)"):
        st.dataframe(result.scope_summary, width="stretch", hide_index=True)

    with st.expander("Look up a specific school (does not change the list above)"):
        # Deliberately separate from the filters: the old version searched by
        # name BEFORE ranking, which quietly redefined what "Top 50" meant.
        query = st.text_input("School or municipality name", key=f"{level}_pri_lookup")
        if query:
            q = query.strip().lower()
            pool = scored_year
            if state_filter != "All Brazil":
                pool = pool.loc[pool["state_code"] == state_filter]
            name_cols = [c for c in ("school_name", "municipality_name") if c in pool.columns]
            mask = pd.Series(False, index=pool.index)
            for c in name_cols:
                mask = mask | pool[c].astype(str).str.lower().str.contains(q, na=False)
            hits = pool.loc[mask]
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
                if c in hits.columns
            ]
            st.caption(f"{len(hits):,} matching schools in {year_filter}.")
            st.dataframe(
                hits[show_cols].sort_values("pred_dropout_rate", ascending=False).head(200),
                width="stretch",
                hide_index=True,
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
        [
            "Prioritized school list",
            "Batch scoring (CSV)",
            "Single school profiler",
            "Model insights",
        ]
    )

    with tabs[0]:
        _render_prioritization_tab(level, art)

    with tabs[1]:
        st.subheader("Batch scoring")
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
                        "municipality_name",
                        "state_code",
                        "admin_dependency_label",
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
            st.caption(
                "Measured against the mart's `high_risk` label, which the Foundation kept as a "
                "historical/evaluation label. It is not the product rule and is not shown to "
                "the end user — the prioritized list is."
            )
            st.json(ranking)
        within = art["metrics"].get("test_ranking_within_network")
        if within:
            st.write("**Ranking quality inside a network** (how the tool is actually used)")
            st.caption(
                "Top-N within one network, restricted to networks large enough for N to be a "
                "real constraint. Lower than the national figure above, because schools inside "
                "one network are more alike — this is the number to expect in the field."
            )
            st.json(within)
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
