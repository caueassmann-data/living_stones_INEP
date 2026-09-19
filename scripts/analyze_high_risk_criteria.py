"""Milestone 1 analysis: current high-risk flag vs candidate operational rules.

Reads the existing marts (no retraining). Scores the latest year with the
trained pipelines so operational shortlists can be compared to the mart label.
Prints a JSON payload consumed by the findings note.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_school_mart
from src.inference import predict_frame

OUT_PATH = PROJECT_ROOT / "docs" / "milestone1_high_risk_stats.json"
EXAMPLE_STATES = ("BA", "PE", "SP", "AM")
LATEST_YEAR = 2025


def _share(n: int, d: int) -> float:
    return float(n / d) if d else float("nan")


def _counts(series: pd.Series) -> dict:
    vc = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def observed_profile(df: pd.DataFrame, level: str) -> dict:
    n_rows = int(len(df))
    n_schools = int(df["school_id"].nunique())
    n_high = int((df["high_risk"] == 1).sum())
    n_schools_ever_high = int(df.loc[df["high_risk"] == 1, "school_id"].nunique())
    by_year = []
    for year, g in df.groupby("year"):
        n = int(len(g))
        h = int((g["high_risk"] == 1).sum())
        by_year.append(
            {
                "year": int(year),
                "rows": n,
                "schools": int(g["school_id"].nunique()),
                "high_risk_rows": h,
                "high_risk_share": round(_share(h, n), 4),
                "mean_dropout": round(float(g["target_dropout_rate"].mean()), 3),
                "p66_dropout": round(float(g["target_dropout_rate"].quantile(0.66)), 3),
                "share_ge_5": round(float((g["target_dropout_rate"] >= 5).mean()), 4),
                "share_ge_10": round(float((g["target_dropout_rate"] >= 10).mean()), 4),
                "share_zero": round(float((g["target_dropout_rate"] == 0).mean()), 4),
            }
        )
    latest = df[df["year"] == LATEST_YEAR].copy()
    by_state = (
        latest.groupby("state_code")
        .agg(
            schools=("school_id", "nunique"),
            high_risk=("high_risk", "sum"),
            mean_dropout=("target_dropout_rate", "mean"),
        )
        .reset_index()
        .sort_values("high_risk", ascending=False)
    )
    by_state["high_risk_share"] = (by_state["high_risk"] / by_state["schools"]).round(4)
    example = {}
    for uf in EXAMPLE_STATES:
        sl = latest[latest["state_code"] == uf]
        example[uf] = {
            "schools": int(sl["school_id"].nunique()),
            "high_risk": int((sl["high_risk"] == 1).sum()),
            "high_risk_share": round(float((sl["high_risk"] == 1).mean()), 4) if len(sl) else None,
        }
    return {
        "level": level,
        "rows": n_rows,
        "schools": n_schools,
        "high_risk_rows": n_high,
        "high_risk_share": round(_share(n_high, n_rows), 4),
        "schools_ever_high_risk": n_schools_ever_high,
        "risk_band": _counts(df["risk_band"]),
        "q66_all_years": round(float(df["target_dropout_rate"].quantile(0.66)), 3),
        "mean_dropout": round(float(df["target_dropout_rate"].mean()), 3),
        "share_zero_dropout": round(float((df["target_dropout_rate"] == 0).mean()), 4),
        "by_year": by_year,
        "latest_year": LATEST_YEAR,
        "latest_schools": int(latest["school_id"].nunique()),
        "latest_high_risk": int((latest["high_risk"] == 1).sum()),
        "latest_high_risk_share": round(float((latest["high_risk"] == 1).mean()), 4),
        "latest_by_state_top10": by_state.head(10).round(3).to_dict(orient="records"),
        "example_states_latest": example,
    }


def _overlap(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool)
    b = b.astype(bool)
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    union = both + only_a + only_b
    return {
        "n_a": int(a.sum()),
        "n_b": int(b.sum()),
        "intersection": both,
        "jaccard": round(_share(both, union), 4),
        "share_of_a_in_b": round(_share(both, int(a.sum())), 4),
        "share_of_b_in_a": round(_share(both, int(b.sum())), 4),
    }


def candidate_rules(latest: pd.DataFrame) -> dict:
    n = len(latest)
    rate = latest["target_dropout_rate"]
    pred = latest["pred_dropout_rate"]
    observed_rank = rate.rank(method="min", ascending=False)
    pred_rank = pred.rank(method="min", ascending=False)
    pred_rank_in_state = pred.groupby(latest["state_code"]).rank(method="min", ascending=False)
    obs_rank_in_state = rate.groupby(latest["state_code"]).rank(method="min", ascending=False)
    n_states = int(latest["state_code"].nunique())

    rules = {
        "current_high_risk_label": latest["high_risk"].eq(1),
        "observed_ge_5pct": rate >= 5,
        "observed_ge_10pct": rate >= 10,
        "observed_top10pct_national": observed_rank <= max(int(round(0.10 * n)), 1),
        "observed_top5pct_national": observed_rank <= max(int(round(0.05 * n)), 1),
        "observed_top10pct_within_state": obs_rank_in_state <= np.maximum(
            (latest.groupby("state_code")["school_id"].transform("size") * 0.10).round(), 1
        ),
        "observed_top50_national": observed_rank <= 50,
        "observed_top50_per_state": obs_rank_in_state <= 50,
        "pred_ge_5pct": pred >= 5,
        "pred_top10pct_national": pred_rank <= max(int(round(0.10 * n)), 1),
        "pred_top5pct_national": pred_rank <= max(int(round(0.05 * n)), 1),
        "pred_top10pct_within_state": pred_rank_in_state <= np.maximum(
            (latest.groupby("state_code")["school_id"].transform("size") * 0.10).round(), 1
        ),
        "pred_top50_national": pred_rank <= 50,
        "pred_top50_per_state": pred_rank_in_state <= 50,
        "pred_top20_per_state": pred_rank_in_state <= 20,
    }

    sizes = {name: {"n": int(mask.sum()), "share": round(_share(int(mask.sum()), n), 4)} for name, mask in rules.items()}
    sizes["n_schools_latest"] = n
    sizes["n_states"] = n_states
    sizes["pred_top50_per_state"]["approx_list_size_if_full_states"] = 50 * n_states
    sizes["pred_top20_per_state"]["approx_list_size_if_full_states"] = 20 * n_states

    comparisons = {
        "current_vs_pred_top10pct_national": _overlap(rules["current_high_risk_label"], rules["pred_top10pct_national"]),
        "current_vs_pred_top50_per_state": _overlap(rules["current_high_risk_label"], rules["pred_top50_per_state"]),
        "current_vs_observed_ge_5pct": _overlap(rules["current_high_risk_label"], rules["observed_ge_5pct"]),
        "pred_top50_national_vs_observed_top50_national": _overlap(
            rules["pred_top50_national"], rules["observed_top50_national"]
        ),
        "pred_top10pct_national_vs_observed_top10pct_national": _overlap(
            rules["pred_top10pct_national"], rules["observed_top10pct_national"]
        ),
        "pred_top50_per_state_vs_observed_top50_per_state": _overlap(
            rules["pred_top50_per_state"], rules["observed_top50_per_state"]
        ),
        "pred_ge_5pct_vs_observed_ge_5pct": _overlap(rules["pred_ge_5pct"], rules["observed_ge_5pct"]),
    }

    example_states = {}
    for uf in EXAMPLE_STATES:
        sl = latest[latest["state_code"] == uf]
        if sl.empty:
            continue
        pred_top50 = sl.nlargest(50, "pred_dropout_rate")
        obs_top50 = sl.nlargest(50, "target_dropout_rate")
        overlap_ids = set(pred_top50["school_id"]) & set(obs_top50["school_id"])
        example_states[uf] = {
            "schools": int(len(sl)),
            "current_high_risk": int((sl["high_risk"] == 1).sum()),
            "observed_ge_5": int((sl["target_dropout_rate"] >= 5).sum()),
            "pred_ge_5": int((sl["pred_dropout_rate"] >= 5).sum()),
            "pred_top50_mean_pred": round(float(pred_top50["pred_dropout_rate"].mean()), 2),
            "pred_top50_mean_observed": round(float(pred_top50["target_dropout_rate"].mean()), 2),
            "pred_top50_vs_observed_top50_overlap": int(len(overlap_ids)),
            "current_high_risk_in_pred_top50": int(pred_top50["high_risk"].sum()),
        }

    return {"sizes": sizes, "comparisons": comparisons, "example_states": example_states}


def analyze_level(level: str) -> dict:
    df = load_school_mart(level)
    profile = observed_profile(df, level)
    latest = df[df["year"] == LATEST_YEAR].copy()
    scored = predict_frame(level, latest)
    latest = latest.copy()
    latest["pred_dropout_rate"] = scored["pred_dropout_rate"].to_numpy()
    candidates = candidate_rules(latest)
    return {"profile": profile, "candidates": candidates}


def main() -> None:
    payload = {
        "current_rule": {
            "high_if": "rank_percentile > 0.66 OR observed dropout_rate >= 5.0",
            "floor_tie": "minimum observed rate (usually 0%) is forced to low / not high_risk",
            "bands": "low <=0.33, moderate 0.33-0.66, high >0.66, with >=5% override to high",
            "applies_within": "education level (Fundamental and Medio separately), all years pooled",
            "source": "src/etl/build_school_risk_marts.py::assign_risk_bands",
        },
        "levels": {},
    }
    for level in ("fundamental", "medio"):
        print(f"Analyzing {level}...", flush=True)
        payload["levels"][level] = analyze_level(level)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
