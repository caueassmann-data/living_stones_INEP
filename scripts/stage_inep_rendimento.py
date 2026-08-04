"""Convert INEP Taxas de Rendimento Excel files to staging Parquet (adaptive header)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "latam_education_data" / "02_national" / "brasil" / "taxas_rendimento"
STAGING = ROOT / "latam_education_data" / "staging" / "brasil" / "taxas_rendimento"
STAGING.mkdir(parents=True, exist_ok=True)

KEEP_CODES = [
    "NU_ANO_CENSO",
    "NO_REGIAO",
    "SG_UF",
    "CO_MUNICIPIO",
    "NO_MUNICIPIO",
    "CO_ENTIDADE",
    "NO_ENTIDADE",
    "NO_CATEGORIA",
    "NO_DEPENDENCIA",
    "3_CAT_FUN",
    "3_CAT_FUN_AI",
    "3_CAT_FUN_AF",
    "3_CAT_MED",
    "3_CAT_MED_01",
    "3_CAT_MED_02",
    "3_CAT_MED_03",
    "3_CAT_MED_04",
]

# Pre-2021 INEP files use tap_/tre_/tab_ prefixes (aprovação/reprovação/abandono)
# and shorter geo labels. Map them onto the modern 3_CAT_* schema first.
LEGACY_TO_MODERN = {
    "Ano": "NU_ANO_CENSO",
    "TIPOLOCA": "NO_CATEGORIA",
    "Dependad": "NO_DEPENDENCIA",
    "tab_FUN": "3_CAT_FUN",
    "tab_F14": "3_CAT_FUN_AI",  # anos iniciais
    "tab_F58": "3_CAT_FUN_AF",  # anos finais
    "tab_MED": "3_CAT_MED",
    "tab_M01": "3_CAT_MED_01",
    "tab_M02": "3_CAT_MED_02",
    "tab_M03": "3_CAT_MED_03",
    "tab_M04": "3_CAT_MED_04",
}

RENAME = {
    "NU_ANO_CENSO": "year",
    "NO_REGIAO": "region",
    "SG_UF": "uf",
    "CO_MUNICIPIO": "municipio_id",
    "NO_MUNICIPIO": "municipio_name",
    "CO_ENTIDADE": "school_id",
    "NO_ENTIDADE": "school_name",
    "NO_CATEGORIA": "localizacao",
    "NO_DEPENDENCIA": "dependencia",
    "3_CAT_FUN": "taxa_abandono_fund",
    "3_CAT_FUN_AI": "taxa_abandono_fund_ai",
    "3_CAT_FUN_AF": "taxa_abandono_fund_af",
    "3_CAT_MED": "taxa_abandono_med",
    "3_CAT_MED_01": "taxa_abandono_med_1",
    "3_CAT_MED_02": "taxa_abandono_med_2",
    "3_CAT_MED_03": "taxa_abandono_med_3",
    "3_CAT_MED_04": "taxa_abandono_med_4",
}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"--": pd.NA, "-": pd.NA, "": pd.NA}), errors="coerce")


def _find_xlsx(year: int) -> Path | None:
    base = RAW / f"tx_rend_escolas_{year}"
    if not base.exists():
        return None
    hits = sorted(base.rglob("*.xlsx")) + sorted(base.rglob("*.xls"))
    # Prefer files that look like the school table
    preferred = [p for p in hits if "tx_rend" in p.name.lower() or "escolas" in p.name.lower()]
    return (preferred or hits)[0] if hits else None


def _detect_header_row(xlsx: Path, sheet: str | int = 0) -> int:
    """Find the row that contains CO_ENTIDADE (column codes header)."""
    peek = pd.read_excel(xlsx, sheet_name=sheet, header=None, nrows=25)
    for i, row in peek.iterrows():
        vals = {str(v).strip() for v in row.tolist() if pd.notna(v)}
        has_abandono = bool(vals & {"3_CAT_FUN", "3_CAT_MED", "tab_FUN", "tab_MED"})
        if "CO_ENTIDADE" in vals and has_abandono:
            return int(i)
        if "CO_ENTIDADE" in vals:
            return int(i)
    # Fallback used by recent INEP releases
    return 8


def _pick_sheet(xlsx: Path) -> str | int:
    xl = pd.ExcelFile(xlsx)
    for name in xl.sheet_names:
        if "ESCOLA" in name.upper():
            return name
    return xl.sheet_names[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy INEP codes onto the modern 3_CAT_* column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {src: dst for src, dst in LEGACY_TO_MODERN.items() if src in df.columns and dst not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def convert_year(year: int) -> Path | None:
    xlsx = _find_xlsx(year)
    if xlsx is None:
        print(f"SKIP {year}: no xlsx under {RAW / f'tx_rend_escolas_{year}'}")
        return None
    sheet = _pick_sheet(xlsx)
    header_row = _detect_header_row(xlsx, sheet)
    print(f"{year}: {xlsx.name} sheet={sheet} header_row={header_row}")
    df = _normalize_columns(pd.read_excel(xlsx, sheet_name=sheet, header=header_row))
    cols = [c for c in KEEP_CODES if c in df.columns]
    if "CO_ENTIDADE" not in cols:
        raise ValueError(f"{year}: CO_ENTIDADE not found in columns: {list(df.columns)[:20]}")
    if "3_CAT_FUN" not in cols and "3_CAT_MED" not in cols:
        raise ValueError(f"{year}: abandonment columns missing: {list(df.columns)[:30]}")

    out = df[cols].copy()
    out = out.rename(columns={k: v for k, v in RENAME.items() if k in out.columns})
    for c in out.columns:
        if c.startswith("taxa_abandono"):
            out[c] = _to_num(out[c])
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    else:
        out["year"] = year
    out["school_id"] = pd.to_numeric(out["school_id"], errors="coerce").astype("Int64")
    if "municipio_id" in out.columns:
        out["municipio_id"] = pd.to_numeric(out["municipio_id"], errors="coerce").astype("Int64")
    # Older Excel sheets mix types in label columns; force string for parquet.
    for c in ("region", "uf", "municipio_name", "school_name", "localizacao", "dependencia"):
        if c in out.columns:
            out[c] = out[c].astype("string")

    # Drop title/junk rows without school id
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
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    summary = {
        "source": "INEP Taxas de Rendimento Escolar (official)",
        "definition_abandono": (
            "Percentage of students who stopped attending school after the School Census "
            "reference date during the school year (movement = left attending). "
            "Computed from Censo Escolar Situação do Aluno module."
        ),
        "grain": "school x year (CO_ENTIDADE / school_id)",
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "rows": int(len(df)),
        "schools": int(df["school_id"].nunique()),
        "coverage": {
            "fundamental_non_null_pct": float(df["taxa_abandono_fund"].notna().mean())
            if "taxa_abandono_fund" in df
            else None,
            "medio_non_null_pct": float(df["taxa_abandono_med"].notna().mean())
            if "taxa_abandono_med" in df
            else None,
            "fundamental_mean": float(df["taxa_abandono_fund"].mean(skipna=True))
            if "taxa_abandono_fund" in df
            else None,
            "medio_mean": float(df["taxa_abandono_med"].mean(skipna=True))
            if "taxa_abandono_med" in df
            else None,
        },
        "student_level_labeled_dropout_available": False,
        "join_key_to_censo": ["school_id=CO_ENTIDADE", "year=NU_ANO_CENSO"],
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
