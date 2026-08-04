"""Shared Streamlit UI for Brazil school-level abandonment apps."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import load_artifacts, predict_frame, score_single
from src.utils import LIMITATION_STATEMENT, mart_path


def run_app(level: str, title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(LIMITATION_STATEMENT)

    try:
        art = load_artifacts(level)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tabs = st.tabs(["School triage (batch)", "Single school profiler", "Model insights"])

    with tabs[0]:
        st.subheader("Batch school triage")
        st.write(
            "Upload a CSV with school feature columns (same schema as the training mart). "
            "The model predicts the official-style abandonment rate (%)."
        )
        default_path = mart_path(level)
        if default_path.exists():
            if st.button("Score a sample of the training mart (first 500 rows)"):
                sample = pd.read_parquet(default_path).head(500)
                scored = predict_frame(level, sample)
                show_cols = [
                    c
                    for c in [
                        "year",
                        "school_id",
                        "uf",
                        "target_dropout_rate",
                        "pred_taxa_abandono",
                        "enrollment_level",
                    ]
                    if c in scored.columns
                ]
                st.dataframe(scored[show_cols].sort_values("pred_taxa_abandono", ascending=False))
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
        st.subheader("Single school profiler")
        defaults = {
            "tp_dependencia": 2,
            "tp_localizacao": 1,
            "is_rural": 0,
            "is_public": 1,
            "in_agua": 1,
            "in_energia": 1,
            "in_esgoto": 1,
            "in_internet": 1,
            "in_biblioteca": 1,
            "in_lab_info": 0,
            "in_quadra": 1,
            "qt_mat_bas": 300,
            "enrollment_level": 200,
            "qt_doc_bas": 15,
            "student_teacher_ratio": 13.3,
        }
        row = {}
        cols = st.columns(3)
        for i, feat in enumerate(art["features"]):
            with cols[i % 3]:
                row[feat] = st.number_input(
                    feat,
                    value=float(defaults.get(feat, 0.0)),
                    key=f"{level}_{feat}",
                )
        if st.button("Score school", type="primary"):
            result = score_single(level, row)
            st.metric("Predicted abandonment rate (%)", f"{result['pred_taxa_abandono']:.2f}")
            st.json(result)

    with tabs[2]:
        st.subheader("Hold-out metrics & drivers")
        st.json(art["metrics"].get("test_metrics", {}))
        st.write(f"Selected model: **{art['metrics'].get('selected_model', 'n/a')}**")
        imp_path = Path(art.get("metrics", {}).get("figures", ""))  # may not exist
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
            "Compare Fundamental vs Médio drivers using notebook "
            "`notebooks/04_compare_fundamental_vs_medio.ipynb`."
        )

    st.divider()
    st.caption(
        f"model_version={art['model_version']} | level={level} | {LIMITATION_STATEMENT}"
    )
