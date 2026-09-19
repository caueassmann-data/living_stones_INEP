"""Milestone 1, tasks 6-7: evidence for Top-N prioritization within a network.

Produces the numbers the Foundation needs to size a rollout after choosing
Top-N prioritization over a fixed dropout threshold:

  * how big education networks actually are, per granularity (this is the
    decisive one — N = 50 inside a municipal network is not a constraint when
    the median such network holds a handful of schools);
  * how well the ranking discriminates inside a network, at N = 20/50/100;
  * how many schools a national rollout would put on worklists;
  * how stable a list is (tie-break, and year-over-year churn).

Separate from scripts/analyze_high_risk_criteria.py on purpose: that script's
output, docs/milestone1_high_risk_stats.json, is the frozen evidence behind an
already-delivered stakeholder note, and regenerating it with new keys would
rewrite numbers that note cites. This writes its own file.

Reads existing marts and trained pipelines. No retraining.

    python scripts/analyze_network_prioritization.py
    python scripts/analyze_network_prioritization.py --state PE --year 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_school_mart
from src.evaluate import evaluate_ranking_within_group
from src.inference import predict_frame
from src.prioritize import (
    PUBLIC_ADMIN_DEPENDENCY_TYPES,
    TOP_N_PRESETS,
    add_scope_key,
    prioritize,
)
from src.utils import EDUCATION_LEVELS, write_json

OUT_PATH = PROJECT_ROOT / "docs" / "milestone1_network_prioritization_stats.json"
DEFAULT_YEAR = 2025
DEFAULT_EXAMPLE_STATE = "BA"
GRANULARITIES = ("state", "state_network", "municipality", "network")


def _scored_year(level: str, year: int) -> pd.DataFrame:
    df = load_school_mart(level)
    slice_ = df.loc[pd.to_numeric(df["year"], errors="coerce") == year].copy()
    return predict_frame(level, slice_)


def scope_size_distribution(scored: pd.DataFrame) -> dict:
    """How many schools sit in one pool, per granularity.

    This is the number that decides whether Top-N is a real constraint. A
    granularity where most pools are smaller than N cannot prioritize.
    """
    out: dict[str, dict] = {}
    public = scored.loc[scored["admin_dependency_type"].isin(PUBLIC_ADMIN_DEPENDENCY_TYPES)]
    for scope in GRANULARITIES:
        sizes = add_scope_key(public, scope).groupby("scope_key").size()
        entry = {
            "n_scopes": int(len(sizes)),
            "schools_total": int(sizes.sum()),
            "median_size": float(sizes.median()),
            "p75_size": float(sizes.quantile(0.75)),
            "p90_size": float(sizes.quantile(0.90)),
            "max_size": int(sizes.max()),
        }
        for n in TOP_N_PRESETS:
            eligible = sizes[sizes > n]
            entry[f"scopes_larger_than_{n}"] = int(len(eligible))
            entry[f"share_scopes_larger_than_{n}"] = round(float((sizes > n).mean()), 4)
            entry[f"schools_in_scopes_larger_than_{n}"] = int(eligible.sum())
            # The budgeting number: how many schools land on a worklist if
            # every eligible network runs this N.
            entry[f"rollout_list_size_at_{n}"] = int(len(eligible) * n)
        out[scope] = entry
    return out


def topn_quality(scored: pd.DataFrame, scope: str) -> dict:
    """Ranking quality inside a network, and what the selected schools look like."""
    public = scored.loc[scored["admin_dependency_type"].isin(PUBLIC_ADMIN_DEPENDENCY_TYPES)]
    keyed = add_scope_key(public, scope)
    metrics = evaluate_ranking_within_group(
        keyed["pred_dropout_rate"].to_numpy(),
        keyed["high_risk"].fillna(0).to_numpy(),
        keyed["scope_key"],
        top_ns=TOP_N_PRESETS,
        observed_rate=keyed["target_dropout_rate"].to_numpy(),
    )
    # Continuity with the Milestone 1 note: how much of a prioritized list the
    # old high_risk flag already contained.
    for n in TOP_N_PRESETS:
        key = f"top_{n}"
        if metrics["by_top_n"][key].get("n_groups_evaluated", 0) == 0:
            continue
        result = prioritize(
            "unused",
            df=keyed,
            scope=scope,
            top_n=n,
            admin_dependency_types=PUBLIC_ADMIN_DEPENDENCY_TYPES,
            include_backend_labels=True,
        )
        metrics["by_top_n"][key]["share_of_list_already_high_risk"] = round(
            float(result.frame["high_risk"].mean()), 4
        )
        metrics["by_top_n"][key]["list_size_nationwide"] = int(result.n_selected)
    return metrics


def tie_break_stability(scored: pd.DataFrame, scope: str, top_n: int) -> dict:
    """How many rows the deterministic tie-break changes.

    About 1.9% of predictions are exact duplicates, so without a tie-break the
    same query can return different schools between runs. This quantifies how
    much of a list is decided by that tie-break rather than by the model.
    """
    base = prioritize(
        "unused",
        df=scored,
        scope=scope,
        top_n=top_n,
        admin_dependency_types=PUBLIC_ADMIN_DEPENDENCY_TYPES,
    )
    shuffled = prioritize(
        "unused",
        df=scored.sample(frac=1.0, random_state=1234),
        scope=scope,
        top_n=top_n,
        admin_dependency_types=PUBLIC_ADMIN_DEPENDENCY_TYPES,
    )
    a = set(base.frame["school_id"])
    b = set(shuffled.frame["school_id"])
    return {
        "scope": scope,
        "top_n": top_n,
        "list_size": int(len(a)),
        "schools_differing_after_row_shuffle": int(len(a ^ b) // 2),
        "identical": a == b,
    }


def year_over_year_persistence(level: str, scope: str, top_n: int, year: int) -> dict:
    """Share of this year's list that was also on last year's list.

    Field teams ask this first: if the worklist turns over completely every
    year, it is describing noise rather than schools that need sustained
    follow-up.
    """
    this_year = _scored_year(level, year)
    last_year = _scored_year(level, year - 1)
    if last_year.empty:
        return {"scope": scope, "top_n": top_n, "available": False}

    now = prioritize(
        "unused",
        df=this_year,
        scope=scope,
        top_n=top_n,
        admin_dependency_types=PUBLIC_ADMIN_DEPENDENCY_TYPES,
    )
    before = prioritize(
        "unused",
        df=last_year,
        scope=scope,
        top_n=top_n,
        admin_dependency_types=PUBLIC_ADMIN_DEPENDENCY_TYPES,
    )
    a = set(now.frame["school_id"])
    b = set(before.frame["school_id"])
    return {
        "scope": scope,
        "top_n": top_n,
        "available": True,
        "years": [year - 1, year],
        "list_size": int(len(a)),
        "carried_over": int(len(a & b)),
        "share_carried_over": round(float(len(a & b) / len(a)), 4) if a else float("nan"),
    }


def worked_example(scored: pd.DataFrame, state: str, top_n: int) -> dict:
    """One concrete network, the way a field team would see it."""
    out: dict = {"state_code": state, "top_n": top_n}

    state_net = prioritize(
        "unused",
        df=scored,
        scope="state_network",
        state_codes=[state],
        admin_dependency_types=[3],
        top_n=top_n,
        include_backend_labels=True,
    )
    pool = scored.loc[(scored["state_code"] == state) & (scored["admin_dependency_type"] == 3)]
    out["state_municipal_network"] = {
        "pool_size": int(len(pool)),
        "selected": state_net.n_selected,
        "mean_observed_dropout_selected": round(
            float(state_net.frame["target_dropout_rate"].mean()), 3
        ),
        "mean_observed_dropout_pool": round(float(pool["target_dropout_rate"].mean()), 3),
        "share_of_list_already_high_risk": round(float(state_net.frame["high_risk"].mean()), 4)
        if state_net.n_selected
        else float("nan"),
        "schools_currently_high_risk_in_pool": int(pool["high_risk"].sum()),
    }

    if len(pool):
        biggest = pool.groupby(["municipality_id", "municipality_name"]).size().idxmax()
        muni_id, muni_name = biggest
        muni = prioritize(
            "unused",
            df=scored,
            scope="network",
            state_codes=[state],
            municipality_ids=[muni_id],
            admin_dependency_types=[3],
            top_n=top_n,
        )
        muni_pool = pool.loc[pool["municipality_id"] == muni_id]
        out["largest_municipal_network"] = {
            "municipality_name": str(muni_name),
            "municipality_id": int(muni_id),
            "pool_size": int(len(muni_pool)),
            "selected": muni.n_selected,
            "fully_covered": bool(muni.fully_covered_scopes),
            "mean_observed_dropout_selected": round(
                float(muni.frame["target_dropout_rate"].mean()), 3
            )
            if muni.n_selected
            else float("nan"),
            "mean_observed_dropout_pool": round(float(muni_pool["target_dropout_rate"].mean()), 3),
        }
    return out


def _print_markdown(payload: dict) -> None:
    """A table to paste straight into the stakeholder note."""
    print("\n### Network size by granularity (public schools, latest year)\n")
    print("| Level | Granularity | Networks | Median size | Networks > 50 | Rollout list at N=50 |")
    print("|---|---|---:|---:|---:|---:|")
    for level, data in payload["levels"].items():
        for scope, entry in data["scope_size_distribution"].items():
            print(
                f"| {level} | {scope} | {entry['n_scopes']:,} | {entry['median_size']:.0f} | "
                f"{entry['scopes_larger_than_50']:,} | {entry['rollout_list_size_at_50']:,} |"
            )

    print("\n### Ranking quality inside a state network (networks larger than N)\n")
    print("> In-sample: most of these schools were seen during training. The held-out")
    print("> equivalent is in models/<level>/metrics.json -> test_ranking_within_network.\n")
    print("| Level | N | Networks | Precision@N | Base rate | Lift | Observed dropout in list | In pool |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for level, data in payload["levels"].items():
        for key, stats in data["topn_quality_state_network"]["by_top_n"].items():
            if not stats.get("n_groups_evaluated"):
                continue
            print(
                f"| {level} | {key.replace('top_', '')} | {stats['n_groups_evaluated']:,} | "
                f"{stats['precision_at_n_macro']:.3f} | {stats['base_rate_macro']:.3f} | "
                f"{stats['lift_macro']:.2f} | "
                f"{stats.get('mean_observed_dropout_selected', float('nan')):.2f}% | "
                f"{stats.get('mean_observed_dropout_pool', float('nan')):.2f}% |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--state",
        default=DEFAULT_EXAMPLE_STATE,
        help=(
            "State used for the worked example. Not hard-coded on purpose: the Foundation "
            "said the pilot geography follows the committed stakeholder, not the data."
        ),
    )
    parser.add_argument("--levels", nargs="*", default=list(EDUCATION_LEVELS))
    args = parser.parse_args()

    payload: dict = {
        "year": args.year,
        "example_state": args.state,
        "decision": {
            "rule": "top_n_within_network",
            "default_top_n": 50,
            "presets": list(TOP_N_PRESETS),
            "n_owner": "field implementation team",
            "high_risk_label": "kept in the backend as a historical/evaluation label",
        },
        "caveat": (
            "Quality figures here are IN-SAMPLE: the models were trained on a grouped split "
            "covering about 80% of schools, so most schools scored below were seen during "
            "training. They describe how a list looks on known data, which is the right "
            "frame for sizing a rollout, and they OVERSTATE accuracy on new schools. For "
            "the unbiased number use models/<level>/metrics.json -> "
            "test_ranking_within_network, which is computed on held-out schools only."
        ),
        "levels": {},
    }

    for level in args.levels:
        scored = _scored_year(level, args.year)
        payload["levels"][level] = {
            "schools_scored": int(len(scored)),
            "scope_size_distribution": scope_size_distribution(scored),
            "topn_quality_state_network": topn_quality(scored, "state_network"),
            "topn_quality_municipal_network": topn_quality(scored, "network"),
            "tie_break_stability": tie_break_stability(scored, "state_network", 50),
            "year_over_year_persistence": year_over_year_persistence(
                level, "state_network", 50, args.year
            ),
            "worked_example": worked_example(scored, args.state, 50),
        }

    write_json(payload, OUT_PATH)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])
    print(f"\nWrote {OUT_PATH}")
    _print_markdown(payload)


if __name__ == "__main__":
    main()
