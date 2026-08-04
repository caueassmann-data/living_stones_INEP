"""Build the Brazil school-risk marts (Ensino Fundamental and Ensino Medio).

WHAT THIS FILE DOES, IN PLAIN TERMS
------------------------------------
For each education level (fundamental / medio) it builds one table where
every row is "this school, in this year" and the columns are:

1. School Census features observed that year (infrastructure, enrollment,
   staffing, location, administrative network) — see `_load_census_features`.
2. History features built from *previous* years' official attainment rates
   for that same school (its own dropout-rate trend, plus the surrounding
   municipality's and state's recent average) — see `_add_history_features`.
   These did not exist before v0.3.0: the model previously had no way to
   learn "this school has historically had elevated dropout," which is the
   single strongest signal available for this kind of problem.
3. The target: that year's official INEP dropout rate for the level.

WHY SAME-YEAR CENSUS FEATURES ARE NOT LEAKAGE
------------------------------------------------
It might look suspicious that Census features from year Y are used to
predict the dropout rate of year Y (rather than Y+1). It is not leakage:
INEP defines "taxa de abandono" as the share of students who stopped
attending *after* the School Census reference date of that same year (see
docs/data_card.md). The Census snapshot is taken before the outcome window
it is being used to predict. What *would* be leakage — and was the actual
gap in the previous version of this pipeline — is using a school's *own
current-year* attainment numbers as a predictor. This file never does that;
all attainment-derived features are shifted to strictly earlier years before
being joined in (see `_add_history_features`).

Two marts are written, one per level, under
`latam_education_data/marts/school_risk_br_{level}/`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.etl.config import MARTS_ROOT, RAW_ROOT, STAGING_ROOT, discover_brazil_years
from src.etl.utils import write_json, write_parquet
from src.utils import COVID_YEARS

LEVELS = {
    "fundamental": {
        "dropout_col": "dropout_rate_fundamental",
        "approval_col": "approval_rate_fundamental",
        "failure_col": "failure_rate_fundamental",
        "enrollment_cols": ["enrollment_fundamental", "enrollment_basic_ed"],
        "class_count_col": "class_count_fundamental",
        "mart_name": "school_risk_br_fundamental",
        "min_enrollment": 20,
    },
    "medio": {
        "dropout_col": "dropout_rate_medio",
        "approval_col": "approval_rate_medio",
        "failure_col": "failure_rate_medio",
        "enrollment_cols": ["enrollment_medio", "enrollment_basic_ed"],
        "class_count_col": "class_count_medio",
        "mart_name": "school_risk_br_medio",
        "min_enrollment": 20,
    },
}


def assign_risk_bands(dropout_rate: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Assign a 3-level risk_band and a binary high_risk flag from dropout rates.

    Many schools have exactly 0% dropout, so a plain tertile split on the raw
    rate can put the 33rd/66th percentile cutpoints at 0 and misclassify a
    large block of identical values. Rank-percentile within the level avoids
    that collision; the >=5% absolute override additionally guarantees any
    school with a clearly elevated rate is flagged "high" even in a year/level
    where the rank-based cutpoint happens to sit below 5%. `high_risk` (the
    binary flag consumed by src/evaluate.py::evaluate_ranking) uses the exact
    same >=5% absolute threshold as risk_band=="high" — an earlier version of
    this function used `>= max(q66, 5.0)` for high_risk only, which silently
    diverged from risk_band whenever the 66th-percentile rate itself exceeded
    5% (caught by tests/test_risk_bands.py).

    Returns (risk_band, high_risk, q66) where q66 is the raw 66th-percentile
    dropout rate, recorded in the mart manifest for transparency only — it is
    not used as a decision threshold.
    """
    ranks = dropout_rate.rank(method="average", pct=True)
    risk_band = pd.Series("low", index=dropout_rate.index, dtype="object")
    risk_band.loc[ranks > 0.33] = "moderate"
    risk_band.loc[ranks > 0.66] = "high"
    risk_band.loc[dropout_rate >= 5.0] = "high"
    q66 = float(dropout_rate.quantile(0.66))
    high_risk = ((ranks > 0.66) | (dropout_rate >= 5.0)).astype(int)

    # A large tied block at the distribution's floor (commonly many schools
    # at exactly 0% dropout) can have an *average* rank-percentile above the
    # 0.33/0.66 cutoffs even though every member of the tie is, by
    # definition, not elevated relative to anything (pandas' "average" tie
    # method assigns the midpoint of the tied block's occupied rank range,
    # so once a tied floor block exceeds ~66% of the population its midpoint
    # crosses 0.33). Pin the floor value to "low" / not-high-risk — unless
    # the floor itself is already >=5% (every school in the slice is
    # elevated), in which case the absolute override should stand.
    floor_value = dropout_rate.min()
    if floor_value < 5.0:
        floor_mask = dropout_rate == floor_value
        risk_band.loc[floor_mask] = "low"
        high_risk = high_risk.where(~floor_mask, 0)
    return risk_band, high_risk, q66


# The on-disk raw/staging folders keep INEP's original "brasil" directory
# name (see scripts/stage_inep_rendimento.py for why); only the Python
# identifiers below use the English spelling "Brazil".


def _brazil_school_census_dir(year: int) -> Path | None:
    candidates = [
        RAW_ROOT / "national" / "brasil" / f"microdados_censo_escolar_{year}" / "dados",
        Path(__file__).resolve().parents[2]
        / "latam_education_data"
        / "02_national"
        / "brasil"
        / f"microdados_censo_escolar_{year}"
        / "dados",
    ]
    for base in candidates:
        if base.exists():
            return base
    return None


def _find_brazil_school_csv(year: int) -> Path | None:
    base = _brazil_school_census_dir(year)
    if base is None:
        return None
    candidates = list(base.glob("microdados_ed_basica*.csv")) + list(
        base.glob("microdados_ed_basica*.CSV")
    )
    if candidates:
        return candidates[0]
    escola = list(base.glob("Tabela_Escola*.csv"))
    return escola[0] if escola else None


def _find_brazil_aux_csv(year: int, patterns: list[str]) -> Path | None:
    """Find auxiliary Census tables (e.g. Matricula / Docente) when split from Escola.

    Starting in 2025, INEP began publishing enrollment and teacher counts in
    separate CSVs instead of columns on the main school table. This looks
    for those split tables so counts are not silently lost for recent years.
    """
    base = _brazil_school_census_dir(year)
    if base is None:
        return None
    for pattern in patterns:
        hits = list(base.glob(pattern))
        if hits:
            return hits[0]
    return None


def _brazil_available_columns(path: Path) -> set[str]:
    con = duckdb.connect()
    cols = con.execute(
        f"""
        DESCRIBE SELECT * FROM read_csv_auto(
            '{path.as_posix()}',
            delim=';',
            header=True,
            ignore_errors=True,
            sample_size=1000
        )
        """
    ).df()["column_name"].tolist()
    con.close()
    return set(cols)


def _load_school_side_counts(year: int, con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    """Load QT_MAT_* / QT_DOC_* / QT_TUR_* from split 2025+ tables when Escola lacks them."""
    mat_path = _find_brazil_aux_csv(year, ["Tabela_Matricula*.csv", "Tabela_Matricula*.CSV"])
    doc_path = _find_brazil_aux_csv(year, ["Tabela_Docente*.csv", "Tabela_Docente*.CSV"])
    tur_path = _find_brazil_aux_csv(year, ["Tabela_Turma*.csv", "Tabela_Turma*.CSV"])
    if mat_path is None and doc_path is None and tur_path is None:
        return None

    frames: list[pd.DataFrame] = []
    for path, wanted in (
        (mat_path, ("CO_ENTIDADE", "QT_MAT_BAS", "QT_MAT_FUND", "QT_MAT_MED", "QT_MAT_EJA")),
        (doc_path, ("CO_ENTIDADE", "QT_DOC_BAS", "QT_DOC_FUND", "QT_DOC_MED")),
        (tur_path, ("CO_ENTIDADE", "QT_TUR_FUND", "QT_TUR_MED")),
    ):
        if path is None:
            continue
        available = _brazil_available_columns(path)
        keep = [c for c in wanted if c in available]
        if "CO_ENTIDADE" in keep and len(keep) > 1:
            sql_cols = ", ".join(keep)
            frames.append(
                con.execute(
                    f"""
                    SELECT {sql_cols}
                    FROM read_csv_auto(
                        '{path.as_posix()}',
                        delim=';',
                        header=True,
                        ignore_errors=True,
                        sample_size=-1
                    )
                    """
                ).df()
            )
    if not frames:
        return None
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="CO_ENTIDADE", how="outer")
    return out


def _load_attainment_history(years: list[int]) -> pd.DataFrame:
    """Load all staged INEP attainment-rate years (approval/failure/dropout, both levels).

    This is intentionally loaded for the *full* requested year range up
    front, independent of the min-enrollment filtering applied later to the
    modeling mart, so that lag/history features are computed against the
    school's real attainment history rather than a filtered subset of it.
    """
    frames = []
    for year in years:
        path = STAGING_ROOT / "brasil" / "taxas_rendimento" / f"tx_rend_escolas_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing staged attainment-rate file {path}. "
                "Run: python scripts/stage_inep_rendimento.py"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _add_history_features(attainment: pd.DataFrame, level: str) -> pd.DataFrame:
    """Add lagged/historical attainment features, using only years < the row's own year.

    For each school, computes:
      - dropout_rate_lag1 / lag2: the dropout rate 1 and 2 years earlier.
      - dropout_rate_3yr_avg: trailing average of up to the 3 prior years.
      - dropout_rate_trend: lag1 - lag2 (positive = dropout rate rising).
      - has_history: 1 if a prior-year rate exists for this school, else 0.
        (Distinguishes "school new to the panel" from "had 0% dropout last
        year" — both would otherwise look like a missing/zero value.)
      - approval_rate_lag1 / failure_rate_lag1: the school's own prior-year
        promotion and grade-repetition rates. A rising failure rate is a
        well-established leading indicator of dropout in the following year.
      - municipal_dropout_rate_lag1 / state_dropout_rate_lag1: the average
        dropout rate across all schools in the same municipality / state the
        prior year — a spatial-lag signal that captures local shocks (e.g. a
        regional economic downturn) a single school's own history cannot.
    """
    meta = LEVELS[level]
    dropout_col, approval_col, failure_col = (
        meta["dropout_col"],
        meta["approval_col"],
        meta["failure_col"],
    )
    cols = ["year", "school_id", "municipality_id", "state_code", dropout_col]
    for c in (approval_col, failure_col):
        if c not in attainment.columns:
            attainment[c] = np.nan
    cols += [approval_col, failure_col]
    panel = attainment[cols].dropna(subset=["year", "school_id"]).copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel.sort_values(["school_id", "year"])

    by_school = panel.groupby("school_id", group_keys=False)
    panel["dropout_rate_lag1"] = by_school[dropout_col].shift(1)
    panel["dropout_rate_lag2"] = by_school[dropout_col].shift(2)
    panel["dropout_rate_3yr_avg"] = by_school[dropout_col].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    panel["dropout_rate_trend"] = panel["dropout_rate_lag1"] - panel["dropout_rate_lag2"]
    panel["has_history"] = panel["dropout_rate_lag1"].notna().astype(int)
    panel["approval_rate_lag1"] = by_school[approval_col].shift(1)
    panel["failure_rate_lag1"] = by_school[failure_col].shift(1)

    # Spatial lag: mean dropout rate across the municipality / state the
    # prior year. Grouping by (region, year) then shifting per region
    # mirrors the per-school shift above, just at a coarser grain.
    muni_year = (
        panel.groupby(["municipality_id", "year"], as_index=False)[dropout_col]
        .mean()
        .sort_values(["municipality_id", "year"])
    )
    muni_year["municipal_dropout_rate_lag1"] = muni_year.groupby("municipality_id")[
        dropout_col
    ].shift(1)

    state_year = (
        panel.groupby(["state_code", "year"], as_index=False)[dropout_col]
        .mean()
        .sort_values(["state_code", "year"])
    )
    state_year["state_dropout_rate_lag1"] = state_year.groupby("state_code")[dropout_col].shift(1)

    panel = panel.merge(
        muni_year[["municipality_id", "year", "municipal_dropout_rate_lag1"]],
        on=["municipality_id", "year"],
        how="left",
    )
    panel = panel.merge(
        state_year[["state_code", "year", "state_dropout_rate_lag1"]],
        on=["state_code", "year"],
        how="left",
    )

    history_cols = [
        "school_id",
        "year",
        "dropout_rate_lag1",
        "dropout_rate_lag2",
        "dropout_rate_3yr_avg",
        "dropout_rate_trend",
        "has_history",
        "approval_rate_lag1",
        "failure_rate_lag1",
        "municipal_dropout_rate_lag1",
        "state_dropout_rate_lag1",
    ]
    return panel[history_cols]


def _load_census_features(years: list[int]) -> pd.DataFrame:
    """Load School Census features for the given years, with a stable English schema.

    INEP changes its raw column set slightly almost every year (new
    indicators added, old ones split into side tables); this function reads
    whatever is available for each year and always emits the same output
    columns, filling with NaN for anything a given year's file does not have.
    Missing values are handled later by the modeling pipeline's imputer.
    """
    wanted = [
        "NU_ANO_CENSO",
        "CO_ENTIDADE",
        "SG_UF",
        "CO_MUNICIPIO",
        "TP_DEPENDENCIA",
        "TP_LOCALIZACAO",
        "TP_SITUACAO_FUNCIONAMENTO",
        "IN_AGUA_POTAVEL",
        "IN_AGUA_REDE_PUBLICA",
        "IN_ENERGIA_REDE_PUBLICA",
        "IN_ENERGIA_INEXISTENTE",
        "IN_ESGOTO_REDE_PUBLICA",
        "IN_ESGOTO_INEXISTENTE",
        "IN_INTERNET",
        "IN_INTERNET_ALUNOS",
        "IN_BANDA_LARGA",
        "IN_BIBLIOTECA",
        "IN_BIBLIOTECA_SALA_LEITURA",
        "IN_LABORATORIO_INFORMATICA",
        "IN_QUADRA_ESPORTES",
        "QT_MAT_BAS",
        "QT_MAT_FUND",
        "QT_MAT_MED",
        "QT_MAT_EJA",
        "QT_MAT_BAS_15_17",
        "QT_MAT_BAS_18_MAIS",
        "QT_DOC_BAS",
        "QT_DOC_FUND",
        "QT_DOC_MED",
        "QT_TUR_FUND",
        "QT_TUR_MED",
    ]
    frames: list[pd.DataFrame] = []
    con = duckdb.connect()
    for year in years:
        path = _find_brazil_school_csv(year)
        if path is None:
            continue
        available = _brazil_available_columns(path)
        cols = [c for c in wanted if c in available]
        if "CO_ENTIDADE" not in cols and "CO_ESCOLA" in available:
            cols.append("CO_ESCOLA")
        if not cols:
            continue
        col_sql = ", ".join(cols)
        df = con.execute(
            f"""
            SELECT {col_sql}
            FROM read_csv_auto(
                '{path.as_posix()}',
                delim=';',
                header=True,
                ignore_errors=True,
                sample_size=-1
            )
            """
        ).df()
        # From 2025 onward INEP often splits enrollment/teacher/class counts
        # into side tables instead of columns on the main school table.
        need_side = not {"QT_MAT_FUND", "QT_MAT_MED", "QT_MAT_BAS"} & set(df.columns)
        if need_side:
            side = _load_school_side_counts(year, con)
            if side is not None and "CO_ENTIDADE" in side.columns:
                key = "CO_ENTIDADE" if "CO_ENTIDADE" in df.columns else "CO_ESCOLA"
                df = df.merge(side, left_on=key, right_on="CO_ENTIDADE", how="left")

        out = pd.DataFrame()
        out["year"] = (
            pd.to_numeric(df["NU_ANO_CENSO"], errors="coerce") if "NU_ANO_CENSO" in df else year
        )
        out["school_id"] = pd.to_numeric(
            df["CO_ENTIDADE"] if "CO_ENTIDADE" in df else df["CO_ESCOLA"], errors="coerce"
        )
        out["state_code"] = df["SG_UF"] if "SG_UF" in df else None
        out["municipality_id"] = (
            pd.to_numeric(df["CO_MUNICIPIO"], errors="coerce") if "CO_MUNICIPIO" in df else pd.NA
        )
        out["admin_dependency_type"] = (
            pd.to_numeric(df["TP_DEPENDENCIA"], errors="coerce")
            if "TP_DEPENDENCIA" in df
            else pd.NA
        )
        out["location_type"] = (
            pd.to_numeric(df["TP_LOCALIZACAO"], errors="coerce")
            if "TP_LOCALIZACAO" in df
            else pd.NA
        )
        out["operating_status_type"] = (
            pd.to_numeric(df["TP_SITUACAO_FUNCIONAMENTO"], errors="coerce")
            if "TP_SITUACAO_FUNCIONAMENTO" in df
            else pd.NA
        )

        def num(col: str) -> pd.Series:
            return (
                pd.to_numeric(df[col], errors="coerce")
                if col in df
                else pd.Series([pd.NA] * len(df))
            )

        water = num("IN_AGUA_POTAVEL")
        if water.isna().all():
            water = num("IN_AGUA_REDE_PUBLICA")
        out["has_water"] = water
        out["has_electricity"] = num("IN_ENERGIA_REDE_PUBLICA")
        out["electricity_absent"] = num("IN_ENERGIA_INEXISTENTE")
        out["has_sewage"] = num("IN_ESGOTO_REDE_PUBLICA")
        out["sewage_absent"] = num("IN_ESGOTO_INEXISTENTE")
        internet = num("IN_INTERNET")
        if internet.isna().all():
            internet = num("IN_BANDA_LARGA")
        out["has_internet"] = internet
        out["has_internet_for_students"] = num("IN_INTERNET_ALUNOS")
        library = num("IN_BIBLIOTECA")
        if library.isna().all():
            library = num("IN_BIBLIOTECA_SALA_LEITURA")
        out["has_library"] = library
        out["has_computer_lab"] = num("IN_LABORATORIO_INFORMATICA")
        out["has_sports_court"] = num("IN_QUADRA_ESPORTES")
        out["enrollment_basic_ed"] = num("QT_MAT_BAS")
        out["enrollment_fundamental"] = num("QT_MAT_FUND")
        out["enrollment_medio"] = num("QT_MAT_MED")
        out["enrollment_eja"] = num("QT_MAT_EJA")
        out["overage_enrollment_15_17"] = num("QT_MAT_BAS_15_17")
        out["overage_enrollment_18_plus"] = num("QT_MAT_BAS_18_MAIS")
        out["teacher_count_basic_ed"] = num("QT_DOC_BAS")
        out["teacher_count_fundamental"] = num("QT_DOC_FUND")
        out["teacher_count_medio"] = num("QT_DOC_MED")
        out["class_count_fundamental"] = num("QT_TUR_FUND")
        out["class_count_medio"] = num("QT_TUR_MED")
        out["is_rural"] = out["location_type"].fillna(1).astype(float).eq(2.0).astype(int)
        out["is_public"] = (
            out["admin_dependency_type"].fillna(4).astype(float).isin([1, 2, 3]).astype(int)
        )
        if out["operating_status_type"].notna().any():
            out = out[out["operating_status_type"].fillna(1).astype(float).eq(1.0)].copy()
        frames.append(out)
    con.close()
    if not frames:
        raise RuntimeError("No Brazil school Census CSVs could be loaded")
    return pd.concat(frames, ignore_index=True)


def build_school_risk_br_level(level: str, years: list[int] | None = None) -> Path:
    """Build the school x year mart for one education level with history features."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level}; expected {list(LEVELS)}")
    if years is None:
        years = discover_brazil_years(min_year=2016)
    meta = LEVELS[level]
    census = _load_census_features(years)
    attainment = _load_attainment_history(years)
    history = _add_history_features(attainment, level)

    attainment_keep = [
        "year",
        "school_id",
        "school_name",
        "municipality_name",
        "admin_dependency_label",
        "location_label",
        "region",
        meta["dropout_col"],
    ]
    attainment_keep = [c for c in attainment_keep if c in attainment.columns]
    attainment_slim = attainment[attainment_keep].copy()
    attainment_slim = attainment_slim.rename(columns={meta["dropout_col"]: "target_dropout_rate"})

    merged = census.merge(attainment_slim, on=["year", "school_id"], how="inner", suffixes=("", "_attainment"))
    merged = merged.merge(history, on=["year", "school_id"], how="left")
    n_census = len(census)
    n_attainment = len(attainment_slim)
    n_join = len(merged)

    # Keep schools with a non-null official dropout rate for this level.
    merged = merged.loc[merged["target_dropout_rate"].notna()].copy()

    # Prefer level-specific enrollment when available, else fall back to
    # total basic-education enrollment.
    enroll = None
    for col in meta["enrollment_cols"]:
        if col in merged.columns and merged[col].notna().any():
            enroll = merged[col]
            break
    if enroll is None:
        enroll = pd.Series([pd.NA] * len(merged), index=merged.index)
    merged["enrollment_level"] = pd.to_numeric(enroll, errors="coerce")
    merged = merged.loc[merged["enrollment_level"].fillna(0) >= meta["min_enrollment"]].copy()

    teachers = pd.to_numeric(merged.get("teacher_count_basic_ed"), errors="coerce")
    merged["student_teacher_ratio"] = merged["enrollment_level"] / teachers.replace({0: pd.NA})

    class_count = pd.to_numeric(merged.get(meta["class_count_col"]), errors="coerce")
    merged["avg_class_size"] = merged["enrollment_level"] / class_count.replace({0: pd.NA})

    basic_ed_enroll = pd.to_numeric(merged.get("enrollment_basic_ed"), errors="coerce")
    overage = pd.to_numeric(merged.get("overage_enrollment_15_17"), errors="coerce").fillna(
        0
    ) + pd.to_numeric(merged.get("overage_enrollment_18_plus"), errors="coerce").fillna(0)
    merged["overage_enrollment_share"] = overage / basic_ed_enroll.replace({0: pd.NA})
    eja = pd.to_numeric(merged.get("enrollment_eja"), errors="coerce")
    merged["eja_enrollment_share"] = eja / basic_ed_enroll.replace({0: pd.NA})

    merged["is_covid_year"] = merged["year"].astype(int).isin(COVID_YEARS).astype(int)
    merged["target_definition"] = "inep_official_dropout_rate"
    merged["education_level"] = level
    merged["country_iso3"] = "BRA"

    risk_band, high_risk, q66 = assign_risk_bands(merged["target_dropout_rate"])
    merged["risk_band"] = risk_band
    merged["high_risk"] = high_risk
    merged["quality_flag"] = "ok"
    merged.loc[merged["target_dropout_rate"] > 100, "quality_flag"] = "out_of_range"
    merged.loc[merged["target_dropout_rate"] < 0, "quality_flag"] = "out_of_range"

    out_dir = MARTS_ROOT / meta["mart_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{meta['mart_name']}.parquet"
    write_parquet(merged, out_path)
    write_json(
        {
            "rows": int(len(merged)),
            "years": sorted(merged["year"].dropna().astype(int).unique().tolist()),
            "schools": int(merged["school_id"].nunique()),
            "education_level": level,
            "target": "target_dropout_rate",
            "target_definition": "inep_official_dropout_rate",
            "target_source_column": meta["dropout_col"],
            "join_stats": {
                "census_rows": int(n_census),
                "attainment_rows": int(n_attainment),
                "inner_join_rows": int(n_join),
                "final_rows_with_target_and_min_enrollment": int(len(merged)),
                "join_match_rate_vs_census": round(n_join / max(n_census, 1), 4),
            },
            "history_feature_coverage": {
                "share_rows_with_prior_year_history": float(merged["has_history"].mean()),
            },
            "risk_rule": {"rank_high_gt": 0.66, "absolute_high_pct": 5.0, "q66": q66},
            "mean_dropout_rate": float(merged["target_dropout_rate"].mean()),
            "public_share": float(merged["is_public"].mean()) if "is_public" in merged else None,
            "covid_years_flagged": sorted(COVID_YEARS),
            "parquet": str(out_path),
        },
        out_dir / "manifest.json",
    )
    return out_path


def build_school_risk_br(years: list[int] | None = None) -> dict[str, Path]:
    """Build both Fundamental and Medio marts."""
    return {
        "fundamental": build_school_risk_br_level("fundamental", years=years),
        "medio": build_school_risk_br_level("medio", years=years),
    }
