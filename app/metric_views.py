"""Plain-language views of the model metrics, for people who do not read JSON.

The metrics live in models/<level>/metrics.json as nested dictionaries, which
is the right format for code and the wrong one for a stakeholder meeting. This
module turns them into cards, charts, and one-sentence readings. The raw JSON
is still available, but only inside a collapsed "technical details" expander.

The reshaping is done by pure functions (`ranking_rows`, `within_network_rows`,
`error_comparison`) so it can be tested without Streamlit; the `render_*`
functions only draw.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Chosen to stay legible on both Streamlit themes (the app is shown light and
# dark), rather than the matplotlib palette used for the static PNG exports.
COLOR_MODEL = "#2f80c9"
COLOR_REFERENCE = "#8a8f98"
COLOR_WARM = "#e08a2b"

# Keys in metrics.json -> how a person would say it.
CUT_LABELS = {
    "top_10pct": "Top 10% of schools",
    "top_20pct": "Top 20% of schools",
    "top_33pct": "Top 33% of schools",
}
NETWORK_KINDS = {
    "state_network": "State network",
    "municipal_network": "Municipal network",
}


# --------------------------------------------------------------------------
# Pure reshaping (tested in tests/test_metric_views.py)
# --------------------------------------------------------------------------


def pct_lower(model: float | None, baseline: float | None) -> float | None:
    """How much smaller `model` is than `baseline`, as a fraction (0.31 = 31% lower)."""
    if model is None or baseline is None or baseline == 0:
        return None
    return (baseline - model) / baseline


def error_comparison(model: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per error metric, model next to the do-nothing baseline."""
    specs = [
        (
            "mae",
            "Typical error (points)",
            "Technical name: MAE. The average miss, in percentage points of dropout rate. "
            "Lower is better.",
            True,
        ),
        (
            "rmse",
            "Error counting big misses (points)",
            "Technical name: RMSE. Like the typical error, but a few very wrong predictions "
            "weigh more. Lower is better.",
            True,
        ),
        (
            "r2",
            "Differences between schools explained",
            "Technical name: R². How much of the school-to-school difference in dropout "
            "the model accounts for. 0 means nothing (the do-nothing model); 1 would be "
            "perfect. Higher is better.",
            False,
        ),
    ]
    rows = []
    for key, label, help_text, lower_is_better in specs:
        m, b = model.get(key), baseline.get(key)
        if m is None:
            continue
        rows.append(
            {
                "key": key,
                "label": label,
                "help": help_text,
                "model": float(m),
                "baseline": float(b) if b is not None else None,
                "lower_is_better": lower_is_better,
            }
        )
    return rows


def ranking_rows(ranking: dict[str, Any] | None) -> pd.DataFrame:
    """The national ranking metrics as one readable row per cut-off."""
    if not ranking:
        return pd.DataFrame()
    base = ranking.get("base_rate_high_risk")
    rows = []
    for key, stats in (ranking.get("by_k") or {}).items():
        rows.append(
            {
                "cut": CUT_LABELS.get(key, key),
                "schools_selected": int(stats["k_rows"]),
                "precision": float(stats["precision"]),
                "random": float(base) if base is not None else float("nan"),
                "lift": float(stats["lift"]),
                "recall": float(stats["recall"]),
            }
        )
    return pd.DataFrame(rows)


def within_network_rows(within: dict[str, Any] | None) -> pd.DataFrame:
    """Within-network metrics as one row per (kind of network, N)."""
    if not within:
        return pd.DataFrame()
    rows = []
    for kind, payload in within.items():
        for key, stats in (payload.get("by_top_n") or {}).items():
            if not stats.get("n_groups_evaluated"):
                continue
            rows.append(
                {
                    "network_kind": NETWORK_KINDS.get(kind, kind),
                    "kind_key": kind,
                    "n": int(key.replace("top_", "")),
                    "networks": int(stats["n_groups_evaluated"]),
                    "precision": float(stats["precision_at_n_macro"]),
                    "base_rate": float(stats["base_rate_macro"]),
                    "lift": float(stats["lift_macro"]),
                    "observed_in_list": stats.get("mean_observed_dropout_selected"),
                    "observed_in_network": stats.get("mean_observed_dropout_pool"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_error_comparison(
    model: dict[str, Any],
    baseline: dict[str, Any],
    *,
    baseline_caption: str = "always predicting the average school",
) -> None:
    """Cards: the model's error next to the do-nothing baseline, with a plain reading."""
    rows = error_comparison(model, baseline)
    if not rows:
        st.info("No error metrics available for this model.")
        return

    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        m, b = row["model"], row["baseline"]
        is_share = row["key"] == "r2"
        delta = None
        if b is not None:
            if is_share:
                # The do-nothing model's R2 is a hair below zero (-0.0001); show
                # it as 0% rather than "-0%".
                b_shown = 0.0 if abs(b) < 0.005 else b
                delta = f"{(m - b_shown) * 100:+.0f} pts vs do-nothing model ({b_shown:.0%})"
            else:
                delta = f"{m - b:+.2f} vs do-nothing model ({b:.2f})"
        col.metric(
            row["label"],
            f"{m:.0%}" if is_share else f"{m:.2f}",
            delta=delta,
            # Green when the model is better: lower for errors, higher for R2.
            delta_color="inverse" if row["lower_is_better"] else "normal",
            help=row["help"],
        )

    mae = next((r for r in rows if r["key"] == "mae"), None)
    if mae and mae["baseline"]:
        gain = pct_lower(mae["model"], mae["baseline"])
        if gain is not None and gain > 0:
            st.success(
                f"Compared with {baseline_caption}, the model's typical miss is "
                f"**{gain:.0%} smaller** ({mae['model']:.2f} vs {mae['baseline']:.2f} "
                "percentage points)."
            )
        elif gain is not None:
            st.error(
                f"The model's typical miss is **not smaller** than {baseline_caption} "
                f"({mae['model']:.2f} vs {mae['baseline']:.2f} points). Treat its "
                "predictions as exploratory only."
            )


def render_ranking_quality(ranking: dict[str, Any] | None, *, key: str) -> None:
    """How well sorting by predicted risk finds the schools that really are high-risk."""
    df = ranking_rows(ranking)
    if df.empty:
        return

    base = float(df["random"].iloc[0])
    first = df.iloc[0]
    st.write(
        f"Sort every school by predicted risk and look only at the **{first['cut'].lower()}**: "
        f"**{first['precision']:.0%}** of them turn out to be truly high-risk, versus "
        f"**{base:.0%}** if you picked schools at random — about "
        f"**{first['lift']:.1f}× better**, while catching {first['recall']:.0%} of all "
        "high-risk schools."
    )

    long = pd.concat(
        [
            df[["cut", "precision"]].rename(columns={"precision": "share"}).assign(
                series="Ranked by the model"
            ),
            df[["cut", "random"]].rename(columns={"random": "share"}).assign(
                series="Picked at random"
            ),
        ]
    )
    order = df["cut"].tolist()
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("cut:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("series:N"),
            y=alt.Y(
                "share:Q",
                title="Share that are truly high-risk",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Ranked by the model", "Picked at random"],
                    range=[COLOR_MODEL, COLOR_REFERENCE],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("cut:N", title="Cut-off"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("share:Q", title="High-risk share", format=".1%"),
            ],
        )
    )
    labels = bars.mark_text(dy=-7, fontSize=12).encode(text=alt.Text("share:Q", format=".0%"))
    st.altair_chart(bars + labels, width="stretch", key=f"{key}_ranking_chart")

    table = df.rename(
        columns={
            "cut": "Cut-off",
            "schools_selected": "Schools",
            "precision": "Truly high-risk (model)",
            "random": "Truly high-risk (random)",
            "lift": "vs. random",
            "recall": "High-risk schools found",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Schools": st.column_config.NumberColumn(format="localized"),
            "Truly high-risk (model)": st.column_config.NumberColumn(format="percent"),
            "Truly high-risk (random)": st.column_config.NumberColumn(format="percent"),
            "vs. random": st.column_config.NumberColumn(
                format="%.2f×", help="How many times better than picking schools at random."
            ),
            "High-risk schools found": st.column_config.NumberColumn(
                format="percent",
                help="Share of ALL high-risk schools that fall inside this cut-off.",
            ),
        },
    )
    st.caption(
        "“High-risk” here is the historical label in the data (roughly the top third "
        "by observed dropout, or 5% and above). It is used only to grade the ranking; "
        "the prioritized list is what teams actually work from."
    )


def render_within_network(within: dict[str, Any] | None, *, key: str) -> None:
    """Ranking quality inside one network, which is how the tool is really used."""
    df = within_network_rows(within)
    if df.empty:
        return

    st.write(
        "The tool is used **inside one network**: a team takes the top N schools of the "
        "network it works in. Schools inside one network are more alike than schools across "
        "Brazil, so this is a tougher test — and the number to expect in the field."
    )

    state = df[(df["kind_key"] == "state_network") & (df["observed_in_list"].notna())]
    if not state.empty:
        pick = state[state["n"] == 50]
        row = pick.iloc[0] if not pick.empty else state.iloc[0]
        ratio = (
            row["observed_in_list"] / row["observed_in_network"]
            if row["observed_in_network"]
            else None
        )
        tail = f" — about **{ratio:.1f}×** the network average" if ratio else ""
        st.write(
            f"In a state network, the top **{int(row['n'])}** schools average "
            f"**{row['observed_in_list']:.1f}%** actual dropout, against "
            f"**{row['observed_in_network']:.1f}%** for the whole network{tail}."
        )

        long = pd.concat(
            [
                state.assign(
                    series="Prioritized schools", value=state["observed_in_list"]
                ),
                state.assign(
                    series="Whole network", value=state["observed_in_network"]
                ),
            ]
        )
        long["n_label"] = "Top " + long["n"].astype(str)
        order = [f"Top {n}" for n in sorted(state["n"].unique())]
        bars = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("n_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("series:N"),
                y=alt.Y("value:Q", title="Actual dropout rate (%)"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Prioritized schools", "Whole network"],
                        range=[COLOR_WARM, COLOR_REFERENCE],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("n_label:N", title="List size"),
                    alt.Tooltip("series:N", title=""),
                    alt.Tooltip("value:Q", title="Dropout (%)", format=".2f"),
                ],
            )
        )
        labels = bars.mark_text(dy=-7, fontSize=12).encode(text=alt.Text("value:Q", format=".1f"))
        st.altair_chart(bars + labels, width="stretch", key=f"{key}_within_chart")

    table = df.drop(columns=["kind_key", "base_rate"]).rename(
        columns={
            "network_kind": "Network",
            "n": "List size",
            "networks": "Networks tested",
            "precision": "Truly high-risk",
            "lift": "vs. random",
            "observed_in_list": "Dropout, listed (%)",
            "observed_in_network": "Dropout, network (%)",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Truly high-risk": st.column_config.NumberColumn(
                format="percent", help="Share of the listed schools that are truly high-risk."
            ),
            "vs. random": st.column_config.NumberColumn(
                format="%.2f×", help="How many times better than picking schools at random."
            ),
            "Dropout, listed (%)": st.column_config.NumberColumn(
                format="%.2f", help="Average actual dropout of the prioritized schools."
            ),
            "Dropout, network (%)": st.column_config.NumberColumn(
                format="%.2f", help="Average actual dropout of every school in the network."
            ),
        },
    )
    st.caption(
        "A **state network** is one administrative network (municipal, state, federal) "
        "within one state; a **municipal network** is the same within one municipality. "
        "Only networks with more than N schools are tested: in a smaller one the "
        "“top N” is simply the whole network, so there is nothing to measure. "
        "Municipal networks are usually that small (the median has 3 schools), which is "
        "why the app defaults to the state network."
    )


def render_drivers(importance: pd.DataFrame, *, key: str, top: int = 10) -> None:
    """The features the model leans on most, under readable names."""
    from app.model_viz import label_feature

    if importance.empty:
        return
    df = importance.sort_values("importance", ascending=False).head(top).copy()
    df["label"] = df["feature"].map(label_feature)
    df["share"] = df["importance"] / importance["importance"].sum()

    top_row = df.iloc[0]
    st.write(
        f"The single biggest factor is **{top_row['label'].lower()}**, which accounts for "
        f"**{top_row['share']:.0%}** of what the model relies on."
    )
    order = df["label"].tolist()
    chart = (
        alt.Chart(df)
        .mark_bar(color=COLOR_MODEL)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("share:Q", title="Share of the model's attention", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("label:N", title="Factor"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=30 * len(df))
    )
    # Explicit mid-grey: the default text colour is near-black and disappears
    # on the dark theme.
    labels = chart.mark_text(align="left", dx=4, fontSize=12, color=COLOR_REFERENCE).encode(
        text=alt.Text("share:Q", format=".0%")
    )
    st.altair_chart(chart + labels, width="stretch", key=f"{key}_drivers_chart")


MODEL_NAMES = {
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "ridge": "Ridge regression",
    "dummy_mean": "Always guess the average",
    "dummy_median": "Always guess the median",
}


def model_comparison_rows(leaderboard: list[dict[str, Any]], selected: str | None) -> pd.DataFrame:
    """Cross-validation results with readable names and a role per row."""
    rows = []
    for r in leaderboard or []:
        name = r["model"]
        if r.get("is_baseline"):
            role = "Do-nothing baseline"
        elif name == selected:
            role = "Chosen model"
        else:
            role = "Other candidate"
        rows.append(
            {
                "model": MODEL_NAMES.get(name, name),
                "role": role,
                "error": float(r["cv_mae_mean"]),
                "spread": float(r.get("cv_mae_std", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("error").reset_index(drop=True) if rows else pd.DataFrame()


def render_model_comparison(
    leaderboard: list[dict[str, Any]], selected: str | None, *, key: str
) -> None:
    """Why this model was chosen: every candidate's error next to the baselines."""
    df = model_comparison_rows(leaderboard, selected)
    if df.empty:
        return

    chosen = df[df["role"] == "Chosen model"]
    baselines = df[df["role"] == "Do-nothing baseline"]
    if not chosen.empty and not baselines.empty:
        best_base = baselines.iloc[0]
        c = chosen.iloc[0]
        gain = pct_lower(c["error"], best_base["error"])
        st.write(
            f"**{c['model']}** had the smallest typical error ({c['error']:.2f} points). "
            f"The best do-nothing option, *{best_base['model'].lower()}*, scored "
            f"{best_base['error']:.2f}"
            + (f", so the model is **{gain:.0%} better**." if gain and gain > 0 else ".")
            + " The gap is modest: dropout depends on much more than a school's records show."
        )

    order = df["model"].tolist()
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("error:Q", title="Typical error, in percentage points (lower is better)"),
            color=alt.Color(
                "role:N",
                title=None,
                scale=alt.Scale(
                    domain=["Chosen model", "Other candidate", "Do-nothing baseline"],
                    range=[COLOR_MODEL, "#9db8d4", COLOR_WARM],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("error:Q", title="Typical error", format=".3f"),
                alt.Tooltip("spread:Q", title="Variation between folds", format=".3f"),
            ],
        )
        .properties(height=44 * len(df))
    )
    labels = bars.mark_text(align="left", dx=4, fontSize=12, color=COLOR_REFERENCE).encode(
        text=alt.Text("error:Q", format=".2f")
    )
    st.altair_chart(bars + labels, width="stretch", key=f"{key}_cv_chart")


def render_state_comparison(table: pd.DataFrame, *, key: str) -> None:
    """Predicted vs actual average dropout by state, as bars plus a clean table."""
    if table is None or table.empty:
        return
    df = table.copy()
    long = pd.concat(
        [
            df.assign(series="Predicted by the model", value=df["mean_pred"]),
            df.assign(series="Actual", value=df["mean_actual"]),
        ]
    )
    order = df.sort_values("mean_actual", ascending=False)["state_code"].tolist()
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("state_code:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("series:N"),
            y=alt.Y("value:Q", title="Average dropout rate (%)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Predicted by the model", "Actual"],
                    range=[COLOR_MODEL, COLOR_REFERENCE],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("state_code:N", title="State"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("value:Q", title="Dropout (%)", format=".2f"),
                alt.Tooltip("n:Q", title="Schools in sample"),
            ],
        )
    )
    st.altair_chart(bars, width="stretch", key=f"{key}_state_chart")

    shown = df.rename(
        columns={
            "state_code": "State",
            "n": "Schools in sample",
            "mean_pred": "Predicted dropout (%)",
            "mean_actual": "Actual dropout (%)",
        }
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "Schools in sample": st.column_config.NumberColumn(format="%d"),
            "Predicted dropout (%)": st.column_config.NumberColumn(format="%.2f"),
            "Actual dropout (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_technical_details(title: str, payload: Any) -> None:
    """The raw JSON, kept for analysts but out of the way."""
    if not payload:
        return
    with st.expander(f"Technical details: {title} (raw JSON)"):
        st.json(payload)
