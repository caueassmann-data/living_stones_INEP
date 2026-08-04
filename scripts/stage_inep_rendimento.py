"""Convert INEP Taxas de Rendimento Escolar Excel files to staging Parquet.

INEP ("Instituto Nacional de Estudos e Pesquisas Educacionais Anisio Teixeira")
is Brazil's national education-statistics agency. Every year it publishes
"Taxas de Rendimento Escolar" ("School Attainment Rates"): for every school,
the official percentage of students who were promoted, held back, or dropped
out that year.

This script reads INEP's raw spreadsheets (one per year, each with a
different header layout depending on the year — INEP does not keep a stable
schema) and writes one clean Parquet file per year with a single, English,
year-over-year-consistent schema. Downstream code (src/etl/build_school_risk_marts.py)
only ever reads the staged Parquet files, never the raw spreadsheets directly.

Source column codes (from the raw INEP header row) and what they mean:
    1_CAT_FUN / 1_CAT_MED  -> approval rate     ("Taxa de Aprovacao")
    2_CAT_FUN / 2_CAT_MED  -> failure rate      ("Taxa de Reprovacao", i.e. held back a grade)
    3_CAT_FUN / 3_CAT_MED  -> dropout rate      ("Taxa de Abandono")
FUN = Ensino Fundamental (grades 1-9, roughly ages 6-14).
MED = Ensino Medio (grades 10-12, Brazil's upper-secondary / high-school stage).

INEP files published before 2021 use a different, older set of codes
(tap_/tre_/tab_ instead of 1_CAT_/2_CAT_/3_CAT_) for the same three rates.
`LEGACY_SOURCE_TO_MODERN` maps the old codes onto the modern ones before any
other processing happens, so every year after that point is handled
identically regardless of which raw layout it came from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# NOTE: the on-disk folder keeps INEP's original Portuguese directory names
# ("brasil", "taxas_rendimento") so this script keeps working against data
# already downloaded by scripts/download_inep_rendimento.py without forcing
# a re-download. Everything *derived* from these files (columns, labels,
# code identifiers) is renamed to English below.
RAW = ROOT / "latam_education_data" / "02_national" / "brasil" / "taxas_rendimento"
STAGING = ROOT / "latam_education_data" / "staging" / "brasil" / "taxas_rendimento"
STAGING.mkdir(parents=True, exist_ok=True)

# Modern (2021+) INEP source column codes we keep from the raw file.
SOURCE_CODES = [
    "NU_ANO_CENSO",
    "NO_REGIAO",
    "SG_UF",
    "CO_MUNICIPIO",
    "NO_MUNICIPIO",
    "CO_ENTIDADE",
    "NO_ENTIDADE",
    "NO_CATEGORIA",
    "NO_DEPENDENCIA",
    "1_CAT_FUN",  # approval rate, Fundamental
    "1_CAT_FUN_AI",  # approval rate, Fundamental early years (grades 1-5)
    "1_CAT_FUN_AF",  # approval rate, Fundamental final years (grades 6-9)
    "1_CAT_MED",  # approval rate, Medio
    "2_CAT_FUN",  # failure/held-back rate, Fundamental
    "2_CAT_FUN_AI",
    "2_CAT_FUN_AF",
    "2_CAT_MED",  # failure/held-back rate, Medio
    "3_CAT_FUN",  # dropout rate, Fundamental (the modeling target)
    "3_CAT_FUN_AI",
    "3_CAT_FUN_AF",
    "3_CAT_MED",  # dropout rate, Medio (the modeling target)
    "3_CAT_MED_01",
    "3_CAT_MED_02",
    "3_CAT_MED_03",
    "3_CAT_MED_04",
]

# Pre-2021 INEP files use tap_/tre_/tab_ prefixes (aprovacao/reprovacao/abandono)
# and shorter geography column names. Map them onto the modern 1_CAT_/2_CAT_/
# 3_CAT_ schema before anything else runs.
LEGACY_SOURCE_TO_MODERN = {
    "Ano": "NU_ANO_CENSO",
    "TIPOLOCA": "NO_CATEGORIA",
    "Dependad": "NO_DEPENDENCIA",
    "tap_FUN": "1_CAT_FUN",
    "tap_F14": "1_CAT_FUN_AI",
    "tap_F58": "1_CAT_FUN_AF",
    "tap_MED": "1_CAT_MED",
    "tre_FUN": "2_CAT_FUN",
    "tre_F14": "2_CAT_FUN_AI",
    "tre_F58": "2_CAT_FUN_AF",
    "tre_MED": "2_CAT_MED",
    "tab_FUN": "3_CAT_FUN",
    "tab_F14": "3_CAT_FUN_AI",
    "tab_F58": "3_CAT_FUN_AF",
    "tab_MED": "3_CAT_MED",
    "tab_M01": "3_CAT_MED_01",
    "tab_M02": "3_CAT_MED_02",
    "tab_M03": "3_CAT_MED_03",
    "tab_M04": "3_CAT_MED_04",
}

# Final English column names written to the staged Parquet files.
RENAME_TO_ENGLISH = {
    "NU_ANO_CENSO": "year",
    "NO_REGIAO": "region",
    "SG_UF": "state_code",  # Brazilian state abbreviation, e.g. "SP", "RJ" ("UF" = Unidade Federativa)
    "CO_MUNICIPIO": "municipality_id",
    "NO_MUNICIPIO": "municipality_name",
    "CO_ENTIDADE": "school_id",
    "NO_ENTIDADE": "school_name",
    "NO_CATEGORIA": "location_label",
    "NO_DEPENDENCIA": "admin_dependency_label",
    "1_CAT_FUN": "approval_rate_fundamental",
    "1_CAT_FUN_AI": "approval_rate_fundamental_early_years",
    "1_CAT_FUN_AF": "approval_rate_fundamental_final_years",
    "1_CAT_MED": "approval_rate_medio",
    "2_CAT_FUN": "failure_rate_fundamental",
    "2_CAT_FUN_AI": "failure_rate_fundamental_early_years",
    "2_CAT_FUN_AF": "failure_rate_fundamental_final_years",
    "2_CAT_MED": "failure_rate_medio",
    "3_CAT_FUN": "dropout_rate_fundamental",
    "3_CAT_FUN_AI": "dropout_rate_fundamental_early_years",
    "3_CAT_FUN_AF": "dropout_rate_fundamental_final_years",
    "3_CAT_MED": "dropout_rate_medio",
    "3_CAT_MED_01": "dropout_rate_medio_grade1",
    "3_CAT_MED_02": "dropout_rate_medio_grade2",
    "3_CAT_MED_03": "dropout_rate_medio_grade3",
    "3_CAT_MED_04": "dropout_rate_medio_grade4",
}

# INEP's raw category labels are Portuguese words; translate the label
# *values* (not just the column names) so nothing Portuguese survives into
# the modeling data or the apps.
LOCATION_LABEL_TRANSLATION = {"Urbana": "Urban", "Rural": "Rural"}
DEPENDENCY_LABEL_TRANSLATION = {
    "Federal": "Federal",
    "Estadual": "State",
    "Municipal": "Municipal",
    "Privada": "Private",
}


def _to_num(s: pd.Series) -> pd.Series:
    """INEP marks missing rates as '--' or '-'; coerce those (and blanks) to NaN."""
    return pd.to_numeric(s.replace({"--": pd.NA, "-": pd.NA, "": pd.NA}), errors="coerce")


def _find_xlsx(year: int) -> Path | None:
    base = RAW / f"tx_rend_escolas_{year}"
    if not base.exists():
        return None
    hits = sorted(base.rglob("*.xlsx")) + sorted(base.rglob("*.xls"))
    # Prefer files that look like the school-level table (INEP zips sometimes
    # bundle municipality/state rollups alongside the school-level file).
    preferred = [p for p in hits if "tx_rend" in p.name.lower() or "escolas" in p.name.lower()]
    return (preferred or hits)[0] if hits else None


def _detect_header_row(xlsx: Path, sheet: str | int = 0) -> int:
    """Find the row that contains the column-code header (CO_ENTIDADE, etc.).

    INEP prepends 5-8 rows of title/merged-cell text before the actual header,
    and the exact number of rows has changed across releases, so we scan for
    the row rather than hard-coding an offset.
    """
    peek = pd.read_excel(xlsx, sheet_name=sheet, header=None, nrows=25)
    for i, row in peek.iterrows():
        vals = {str(v).strip() for v in row.tolist() if pd.notna(v)}
        has_rate_cols = bool(vals & {"1_CAT_FUN", "3_CAT_FUN", "3_CAT_MED", "tap_FUN", "tab_FUN", "tab_MED"})
        if "CO_ENTIDADE" in vals and has_rate_cols:
            return int(i)
        if "CO_ENTIDADE" in vals:
            return int(i)
    # Fallback used by recent INEP releases when detection above fails.
    return 8


def _pick_sheet(xlsx: Path) -> str | int:
    xl = pd.ExcelFile(xlsx)
    for name in xl.sheet_names:
        if "ESCOLA" in name.upper():
            return name
    return xl.sheet_names[0]


def _normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy (pre-2021) INEP codes onto the modern 1/2/3_CAT_* codes."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        src: dst
        for src, dst in LEGACY_SOURCE_TO_MODERN.items()
        if src in df.columns and dst not in df.columns
    }
    if rename:
        df = df.rename(columns=rename)
    return df


def convert_year(year: int) -> Path | None:
    """Read one year's raw INEP spreadsheet and write a clean English Parquet file."""
    xlsx = _find_xlsx(year)
    if xlsx is None:
        print(f"SKIP {year}: no xlsx under {RAW / f'tx_rend_escolas_{year}'}")
        return None
    sheet = _pick_sheet(xlsx)
    header_row = _detect_header_row(xlsx, sheet)
    print(f"{year}: {xlsx.name} sheet={sheet} header_row={header_row}")
    df = _normalize_source_columns(pd.read_excel(xlsx, sheet_name=sheet, header=header_row))
    cols = [c for c in SOURCE_CODES if c in df.columns]
    if "CO_ENTIDADE" not in cols:
        raise ValueError(f"{year}: CO_ENTIDADE not found in columns: {list(df.columns)[:20]}")
    if "3_CAT_FUN" not in cols and "3_CAT_MED" not in cols:
        raise ValueError(f"{year}: dropout-rate columns missing: {list(df.columns)[:30]}")

    out = df[cols].copy()
    out = out.rename(columns={k: v for k, v in RENAME_TO_ENGLISH.items() if k in out.columns})

    rate_columns = [
        c for c in out.columns if c.startswith(("approval_rate", "failure_rate", "dropout_rate"))
    ]
    for c in rate_columns:
        out[c] = _to_num(out[c])

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    else:
        out["year"] = year
    out["school_id"] = pd.to_numeric(out["school_id"], errors="coerce").astype("Int64")
    if "municipality_id" in out.columns:
        out["municipality_id"] = pd.to_numeric(out["municipality_id"], errors="coerce").astype("Int64")

    # Older Excel sheets mix types in label columns; force string for Parquet.
    for c in ("region", "state_code", "municipality_name", "school_name", "location_label", "admin_dependency_label"):
        if c in out.columns:
            out[c] = out[c].astype("string")

    if "location_label" in out.columns:
        out["location_label"] = out["location_label"].map(LOCATION_LABEL_TRANSLATION).fillna(
            out["location_label"]
        )
    if "admin_dependency_label" in out.columns:
        out["admin_dependency_label"] = out["admin_dependency_label"].map(
            DEPENDENCY_LABEL_TRANSLATION
        ).fillna(out["admin_dependency_label"])

    # Drop title/junk rows that don't carry a school id.
    out = out.loc[out["school_id"].notna()].copy()
    dest = STAGING / f"tx_rend_escolas_{year}.parquet"
    out.to_parquet(dest, index=False)
    print(f"  wrote {dest.name} rows={len(out)}")
    return dest


def discover_years() -> list[int]:
    years = set()
    for p in RAW.glob("tx_rend_escolas_*.zip"):
        m = re.search(r"(\d{4})", p.stem)
        if m:
            years.add(int(m.group(1)))
    for p in RAW.glob("tx_rend_escolas_*"):
        if p.is_dir():
            m = re.search(r"(\d{4})", p.name)
            if m:
                years.add(int(m.group(1)))
    return sorted(y for y in years if y >= 2016)


def validation_summary(paths: list[Path]) -> dict:
    """Write a small JSON summary used by tests and the data card to sanity-check staging."""
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    summary = {
        "source": "INEP Taxas de Rendimento Escolar (official)",
        "definition_dropout_rate": (
            "Percentage of students who stopped attending school after the "
            "School Census reference date during the school year (student "
            "movement status = left attending). Computed by INEP from the "
            "School Census 'Situacao do Aluno' module."
        ),
        "grain": "school x year (school_id = CO_ENTIDADE)",
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "rows": int(len(df)),
        "schools": int(df["school_id"].nunique()),
        "coverage": {
            "fundamental_non_null_pct": float(df["dropout_rate_fundamental"].notna().mean())
            if "dropout_rate_fundamental" in df
            else None,
            "medio_non_null_pct": float(df["dropout_rate_medio"].notna().mean())
            if "dropout_rate_medio" in df
            else None,
            "fundamental_mean": float(df["dropout_rate_fundamental"].mean(skipna=True))
            if "dropout_rate_fundamental" in df
            else None,
            "medio_mean": float(df["dropout_rate_medio"].mean(skipna=True))
            if "dropout_rate_medio" in df
            else None,
            "approval_rate_fundamental_non_null_pct": float(
                df["approval_rate_fundamental"].notna().mean()
            )
            if "approval_rate_fundamental" in df
            else None,
            "failure_rate_fundamental_non_null_pct": float(
                df["failure_rate_fundamental"].notna().mean()
            )
            if "failure_rate_fundamental" in df
            else None,
        },
        "student_level_labeled_dropout_available": False,
        "join_key_to_census": ["school_id=CO_ENTIDADE", "year=NU_ANO_CENSO"],
    }
    out = ROOT / "latam_education_data" / "dq" / "rendimento_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("validation ->", out)
    return summary


def main() -> None:
    years = discover_years()
    print("years to stage:", years)
    for year in years:
        try:
            convert_year(year)
        except Exception as exc:
            print(f"ERROR {year}: {exc}")
    paths = sorted(STAGING.glob("tx_rend_escolas_*.parquet"))
    if not paths:
        raise SystemExit("No rendimento years staged")
    print(json.dumps(validation_summary(paths), indent=2)[:2000])


if __name__ == "__main__":
    main()
