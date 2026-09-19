"""Capacity-constrained Top-N prioritization of schools within a network.

This module holds the product's *decision rule*, deliberately separate from
src/inference.py (which is the model-artifact boundary: load, predict, score).
The rule below is a business policy chosen by the Foundation and will change
again; the model plumbing should not have to move when it does.

The rule, as decided by the Foundation (Milestone 1, tasks 6-7):

  1. Pick a scope — an education network the field team actually works in.
  2. Rank the schools in that scope by predicted dropout rate.
  3. Take the top N, where N is how many schools the team can follow up on
     (default 50, set by whoever runs the field implementation).

Why Top-N and not a dropout threshold: a fixed cutoff such as ">= 5%" means
very different things in Ensino Fundamental (mean 0.5%) and Ensino Medio, so
it cannot be one product rule. A count matches how an intervention is actually
resourced. See docs/milestone1_high_risk_criteria.md.

What this module does NOT do: it does not touch `high_risk` / `risk_band` in
the mart (src/etl/build_school_risk_marts.py `assign_risk_bands`). That label
stays exactly as it is, as a historical/evaluation label used by ranking
metrics. It is not shown to the end user and is excluded from exports here
unless explicitly requested.

Scope granularity matters more than it looks. Measured on the 2025 marts, the
median Brazilian *municipal* network holds 3 schools and only 214 of 9,392 have
50 or more, so "top 50 within a municipal network" returns the whole network
almost everywhere. That is why DEFAULT_SCOPE is "state_network" (one
administrative network within one state) rather than "network", and why every
result carries `scope_fully_covered` instead of silently passing a whole pool
off as a prioritized list.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.inference import predict_frame
from src.utils import MODEL_VERSION

# Bumped when the prioritization rule itself changes (not when the model is
# retrained). Stamped onto every exported row so a CSV found months later can
# be traced back to the rule that produced it.
PRIORITIZATION_VERSION = "1.0.0"

DEFAULT_TOP_N = 50
TOP_N_PRESETS = (20, 50, 100)
MIN_TOP_N = 5
MAX_TOP_N = 500

# Scope = the pool a school competes inside. Ranking is always *within* a
# scope, never across scopes.
SCOPE_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "national": (),
    "state": ("state_code",),
    "state_network": ("state_code", "admin_dependency_type"),
    "municipality": ("state_code", "municipality_id"),
    "network": ("state_code", "municipality_id", "admin_dependency_type"),
    # "selection" pools every municipality the user picked into ONE scope —
    # the answer for a team working across a consortium of municipalities.
    "selection": (),
}
SCOPES = tuple(SCOPE_KEY_COLUMNS)
DEFAULT_SCOPE = "state_network"

SCOPE_DISPLAY_NAME = {
    "national": "Whole country (one national list)",
    "state": "State (UF) — all networks together",
    "state_network": "State network (UF x administrative network)",
    "municipality": "Municipality — all networks together",
    "network": "Municipal network (municipality x administrative network)",
    "selection": "Selected municipalities pooled as one list (consortium)",
}

ADMIN_DEPENDENCY_LABELS = {1: "Federal", 2: "State", 3: "Municipal", 4: "Private"}
PUBLIC_ADMIN_DEPENDENCY_TYPES = (1, 2, 3)

# Ordering used to cut the list at N. The first key is the decision; the rest
# exist only to make the cut reproducible. About 1.9% of 2025 predictions are
# exact duplicates, so without an explicit tie-break the same query returns a
# different set of schools between runs — indefensible to a field team holding
# last month's export.
TIE_BREAK: tuple[tuple[str, bool], ...] = (
    ("pred_dropout_rate", False),  # (column, ascending)
    ("dropout_rate_lag1", False),
    ("enrollment_level", False),
    ("school_id", True),
)

# What a field team gets. `high_risk` / `risk_band` are deliberately absent.
EXPORT_COLUMNS = [
    "priority_rank",
    "scope_label",
    "scope_size",
    "scope_fully_covered",
    "school_id",
    "school_name",
    "municipality_name",
    "state_code",
    "admin_dependency_label",
    "pred_dropout_rate",
    "students_at_risk_estimate",
    "pred_percentile_in_scope",
    "dropout_rate_lag1",
    "target_dropout_rate",
    "enrollment_level",
    "has_history",
    "year",
]
BACKEND_LABEL_COLUMNS = ["high_risk", "risk_band"]

# Stamped on every exported row so a downloaded CSV is self-describing.
META_COLUMNS = ["top_n", "scope", "model_version", "prioritization_version", "generated_at_utc"]


@dataclass
class PrioritizationResult:
    """A prioritized school list plus the scope facts needed to read it honestly."""

    frame: pd.DataFrame
    scope_summary: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return int(len(self.frame))

    @property
    def pool_size(self) -> int:
        return int(self.meta.get("pool_size", 0))

    @property
    def fully_covered_scopes(self) -> int:
        """Scopes where N was >= the scope size, so no prioritization happened."""
        if self.scope_summary.empty:
            return 0
        return int(self.scope_summary["scope_fully_covered"].sum())

    def to_csv_bytes(self) -> bytes:
        return self.frame.to_csv(index=False).encode("utf-8")

    def filename(self, level: str) -> str:
        scope_slug = _slug(self.meta.get("scope_slug", "selection"))
        return (
            f"priority_top{self.meta.get('top_n', DEFAULT_TOP_N)}_{level}_"
            f"{self.meta.get('year', 'all')}_{self.meta.get('scope', DEFAULT_SCOPE)}_"
            f"{scope_slug}.csv"
        )


def _slug(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(value))
    return "-".join(part for part in text.split("-") if part)[:60] or "all"


def scope_key_columns(scope: str) -> list[str]:
    """Columns that together identify one prioritization pool."""
    if scope not in SCOPE_KEY_COLUMNS:
        raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")
    return list(SCOPE_KEY_COLUMNS[scope])


def _admin_label(value: Any) -> str:
    try:
        return ADMIN_DEPENDENCY_LABELS.get(int(value), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def add_scope_key(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Attach `scope_key` (grouping key) and `scope_label` (human-readable)."""
    cols = scope_key_columns(scope)
    out = df.copy()

    if not cols:
        # "national" and "selection" are both a single pool; they differ only
        # in how the caller filtered the frame beforehand.
        out["scope_key"] = "all"
        out["scope_label"] = (
            "Whole country" if scope == "national" else "Selected municipalities"
        )
        return out

    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f"Scope {scope!r} needs column(s) {missing} which are not in the frame")

    out["scope_key"] = out[cols].astype(str).agg("|".join, axis=1)

    parts: list[pd.Series] = []
    if "state_code" in cols:
        parts.append(out["state_code"].astype(str))
    if "municipality_id" in cols:
        name = (
            out["municipality_name"].astype(str)
            if "municipality_name" in out.columns
            else out["municipality_id"].astype(str)
        )
        parts.append(name)
    if "admin_dependency_type" in cols:
        parts.append(out["admin_dependency_type"].map(_admin_label))
    out["scope_label"] = parts[0]
    for extra in parts[1:]:
        out["scope_label"] = out["scope_label"] + " / " + extra
    return out


def rank_within_scope(scored: pd.DataFrame, *, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Rank by predicted dropout inside each `scope_key` and keep the top N.

    Uses a stable sort plus `cumcount()` rather than `Series.rank()`: with tied
    predictions, `rank(method="min")` assigns several schools the same rank and
    a `rank <= N` filter then returns MORE than N rows. A team that asked for 50
    schools must get 50.
    """
    if "scope_key" not in scored.columns:
        raise KeyError("rank_within_scope() expects a frame from add_scope_key()")

    sort_cols = [c for c, _ in TIE_BREAK if c in scored.columns]
    ascending = [asc for c, asc in TIE_BREAK if c in scored.columns]
    ordered = scored.sort_values(sort_cols, ascending=ascending, kind="stable")

    grouped = ordered.groupby("scope_key", sort=False)
    ordered["priority_rank"] = grouped.cumcount() + 1
    ordered["scope_size"] = grouped["scope_key"].transform("size")
    # Percentile of the school inside its own pool (a read-out, never the
    # selection rule — a percentage cut is exactly what the Foundation rejected
    # because it is not comparable between Fundamental and Medio).
    ordered["pred_percentile_in_scope"] = (
        1.0 - (ordered["priority_rank"] - 1) / ordered["scope_size"]
    ).round(4)
    ordered["scope_fully_covered"] = ordered["scope_size"] <= top_n
    return ordered


def _apply_filters(
    df: pd.DataFrame,
    *,
    year: int | None,
    state_codes: Sequence[str] | None,
    municipality_ids: Sequence[Any] | None,
    admin_dependency_types: Sequence[int] | None,
    require_history: bool,
) -> pd.DataFrame:
    out = df
    if year is not None:
        out = out.loc[pd.to_numeric(out["year"], errors="coerce") == int(year)]
    if state_codes:
        out = out.loc[out["state_code"].isin(list(state_codes))]
    if municipality_ids:
        wanted = pd.to_numeric(pd.Series(list(municipality_ids)), errors="coerce")
        out = out.loc[pd.to_numeric(out["municipality_id"], errors="coerce").isin(wanted)]
    if admin_dependency_types:
        wanted_types = [int(t) for t in admin_dependency_types]
        out = out.loc[
            pd.to_numeric(out["admin_dependency_type"], errors="coerce").isin(wanted_types)
        ]
    if require_history and "has_history" in out.columns:
        out = out.loc[pd.to_numeric(out["has_history"], errors="coerce") == 1]
    return out.copy()


def _scope_summary(ranked: pd.DataFrame, selected: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One row per scope: how big the pool was and how much of it was selected."""
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "scope_key",
                "scope_label",
                "scope_size",
                "n_selected",
                "scope_fully_covered",
                "mean_pred_selected",
                "mean_pred_pool",
            ]
        )
    pool = (
        ranked.groupby("scope_key", sort=False)
        .agg(
            scope_label=("scope_label", "first"),
            scope_size=("scope_key", "size"),
            mean_pred_pool=("pred_dropout_rate", "mean"),
        )
        .reset_index()
    )
    chosen = (
        selected.groupby("scope_key", sort=False)
        .agg(
            n_selected=("scope_key", "size"),
            mean_pred_selected=("pred_dropout_rate", "mean"),
        )
        .reset_index()
    )
    summary = pool.merge(chosen, on="scope_key", how="left")
    summary["n_selected"] = summary["n_selected"].fillna(0).astype(int)
    summary["scope_fully_covered"] = summary["scope_size"] <= top_n
    return summary.sort_values("scope_size", ascending=False).reset_index(drop=True)


def prioritize(
    level: str,
    *,
    df: pd.DataFrame | None = None,
    year: int | None = None,
    scope: str = DEFAULT_SCOPE,
    top_n: int = DEFAULT_TOP_N,
    state_codes: Sequence[str] | None = None,
    municipality_ids: Sequence[Any] | None = None,
    admin_dependency_types: Sequence[int] | None = None,
    require_history: bool = False,
    include_backend_labels: bool = False,
    predictor: Callable[[str, pd.DataFrame], pd.DataFrame] = predict_frame,
) -> PrioritizationResult:
    """Build a capacity-constrained Top-N list of schools within each scope.

    `df` may be a raw mart slice or a frame that already carries
    `pred_dropout_rate` (the Streamlit app scores a whole year once and reuses
    it across filter changes). When it is None the level's mart is loaded.

    `predictor` is injectable so tests can exercise the rule without a trained
    model on disk.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if scope not in SCOPE_KEY_COLUMNS:
        raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")

    if df is None:
        from src.data_loader import load_school_mart

        df = load_school_mart(level)

    filtered = _apply_filters(
        df,
        year=year,
        state_codes=state_codes,
        municipality_ids=municipality_ids,
        admin_dependency_types=admin_dependency_types,
        require_history=require_history,
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta: dict[str, Any] = {
        "level": level,
        "year": year,
        "scope": scope,
        "scope_display": SCOPE_DISPLAY_NAME[scope],
        "top_n": int(top_n),
        "pool_size": int(len(filtered)),
        "state_codes": list(state_codes) if state_codes else [],
        "municipality_ids": list(municipality_ids) if municipality_ids else [],
        "admin_dependency_types": list(admin_dependency_types) if admin_dependency_types else [],
        "require_history": bool(require_history),
        "model_version": MODEL_VERSION,
        "prioritization_version": PRIORITIZATION_VERSION,
        "generated_at_utc": generated_at,
        "scope_slug": _scope_slug(scope, state_codes, municipality_ids, admin_dependency_types),
    }

    if filtered.empty:
        meta.update({"n_scopes": 0, "n_selected": 0, "n_scopes_fully_covered": 0})
        return PrioritizationResult(
            frame=pd.DataFrame(columns=EXPORT_COLUMNS + META_COLUMNS),
            scope_summary=_scope_summary(filtered, filtered, top_n),
            meta=meta,
        )

    scored = filtered if "pred_dropout_rate" in filtered.columns else predictor(level, filtered)
    scored = add_scope_key(scored, scope)
    ranked = rank_within_scope(scored, top_n=top_n)
    selected = ranked.loc[ranked["priority_rank"] <= top_n].copy()

    if "admin_dependency_label" not in selected.columns:
        selected["admin_dependency_label"] = (
            selected["admin_dependency_type"].map(_admin_label)
            if "admin_dependency_type" in selected.columns
            else "Unknown"
        )
    # A 6% predicted rate in a 15-student school and in a 900-student school
    # are not the same intervention; capacity planning needs both numbers.
    selected["students_at_risk_estimate"] = (
        pd.to_numeric(selected["pred_dropout_rate"], errors="coerce")
        / 100.0
        * pd.to_numeric(selected.get("enrollment_level"), errors="coerce")
    ).round(1)

    for col, value in (
        ("top_n", int(top_n)),
        ("scope", scope),
        ("model_version", MODEL_VERSION),
        ("prioritization_version", PRIORITIZATION_VERSION),
        ("generated_at_utc", generated_at),
    ):
        selected[col] = value

    keep = [c for c in EXPORT_COLUMNS if c in selected.columns]
    if include_backend_labels:
        keep += [c for c in BACKEND_LABEL_COLUMNS if c in selected.columns]
    keep += META_COLUMNS
    frame = selected[keep].sort_values(
        ["scope_label", "priority_rank"], kind="stable"
    ).reset_index(drop=True)

    summary = _scope_summary(ranked, selected, top_n)
    meta.update(
        {
            "n_scopes": int(len(summary)),
            "n_selected": int(len(frame)),
            "n_scopes_fully_covered": int(summary["scope_fully_covered"].sum()),
            "mean_pred_selected": float(frame["pred_dropout_rate"].mean()),
            "mean_pred_pool": float(ranked["pred_dropout_rate"].mean()),
        }
    )
    return PrioritizationResult(frame=frame, scope_summary=summary, meta=meta)


def _scope_slug(
    scope: str,
    state_codes: Sequence[str] | None,
    municipality_ids: Sequence[Any] | None,
    admin_dependency_types: Sequence[int] | None,
) -> str:
    parts: list[str] = []
    if state_codes:
        parts.append("-".join(str(s) for s in list(state_codes)[:3]))
    if municipality_ids:
        parts.append(f"{len(list(municipality_ids))}mun")
    if admin_dependency_types:
        parts.append("-".join(_admin_label(t) for t in admin_dependency_types))
    return _slug("_".join(parts) if parts else scope)


def coverage_message(result: PrioritizationResult) -> str | None:
    """Plain-language warning when N was not actually a constraint.

    Returned instead of raised because a small network is a legitimate answer —
    it just must not be presented as if prioritization had happened.
    """
    covered = result.fully_covered_scopes
    if covered == 0:
        return None
    top_n = result.meta.get("top_n", DEFAULT_TOP_N)
    n_scopes = result.meta.get("n_scopes", 0)
    if n_scopes == 1:
        size = int(result.scope_summary["scope_size"].iloc[0])
        return (
            f"This network has {size} schools, fewer than N={top_n}, so every school in it "
            "is listed. The list is ordered, but no prioritization is happening. Widen the "
            "scope (add municipalities, or switch to the state network) or lower N."
        )
    return (
        f"{covered} of {n_scopes} networks in this selection have {top_n} schools or fewer, "
        "so their whole network is listed rather than a prioritized subset. Column "
        "`scope_fully_covered` marks them."
    )


# Below this predicted dropout, a "highest-risk" school is not meaningfully at
# risk. Set at 1 percentage point: hold-out MAE is around +/-1.1 points, so
# differences under it are inside the model's own error.
LOW_SIGNAL_PRED_THRESHOLD = 1.0


def low_signal_message(result: PrioritizationResult) -> str | None:
    """Warn when even the top of the list is predicted to be near zero.

    Some networks genuinely have almost no dropout — Salvador's municipal
    Fundamental network averages 0.05% observed — and there the ranking is
    sorting noise: its top 50 has *lower* observed dropout than the network as
    a whole. A Top-N list always returns N schools, so nothing in the output
    itself reveals that there was nothing to prioritize. This says it.
    """
    if result.n_selected == 0:
        return None
    mean_pred = result.meta.get("mean_pred_selected")
    if mean_pred is None or mean_pred >= LOW_SIGNAL_PRED_THRESHOLD:
        return None
    return (
        f"Schools in this list are predicted at only {mean_pred:.2f}% dropout on average — "
        "below the model's own margin of error. This network has very little dropout to "
        "prioritize, so treat the ordering as weak evidence and compare against a network "
        "where the rate is higher."
    )
