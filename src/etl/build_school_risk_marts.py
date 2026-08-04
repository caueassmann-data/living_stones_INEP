"""Brazil school-risk marts: Censo features + official INEP abandonment rates."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.etl.config import MARTS_ROOT, RAW_ROOT, STAGING_ROOT, discover_brasil_years
from src.etl.utils import write_json, write_parquet

LEVELS = {
    "fundamental": {
        "target_col": "taxa_abandono_fund",
        "enrollment_cols": ["QT_MAT_FUND", "QT_MAT_BAS"],
        "mart_name": "school_risk_br_fundamental",
        "min_enrollment": 20,
    },
    "medio": {
        "target_col": "taxa_abandono_med",
        "enrollment_cols": ["QT_MAT_MED", "QT_MAT_BAS"],
        "mart_name": "school_risk_br_medio",
        "min_enrollment": 20,
    },
}


def _brasil_dados_dir(year: int) -> Path | None:
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


def _find_brasil_school_csv(year: int) -> Path | None:
    base = _brasil_dados_dir(year)
    if base is None:
        return None
    candidates = list(base.glob("microdados_ed_basica*.csv")) + list(
        base.glob("microdados_ed_basica*.CSV")
    )
    if candidates:
        return candidates[0]
    escola = list(base.glob("Tabela_Escola*.csv"))
    return escola[0] if escola else None


def _find_brasil_aux_csv(year: int, patterns: list[str]) -> Path | None:
    """Find auxiliary Censo tables (e.g. Matricula / Docente) when split from Escola."""
    base = _brasil_dados_dir(year)
    if base is None:
        return None
    for pattern in patterns:
        hits = list(base.glob(pattern))
        if hits:
            return hits[0]
    return None


def _brasil_available_columns(path: Path) -> set[str]:
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
    """Load QT_MAT_* / QT_DOC_* from split 2025+ tables when Escola lacks them."""
    mat_path = _find_brasil_aux_csv(
        year, ["Tabela_Matricula*.csv", "Tabela_Matricula*.CSV"]
    )
    doc_path = _find_brasil_aux_csv(
        year, ["Tabela_Docente*.csv", "Tabela_Docente*.CSV"]
    )
    if mat_path is None and doc_path is None:
        return None

    frames: list[pd.DataFrame] = []
    if mat_path is not None:
        mat_cols = _brasil_available_columns(mat_path)
        keep = [c for c in ("CO_ENTIDADE", "QT_MAT_BAS", "QT_MAT_FUND", "QT_MAT_MED") if c in mat_cols]
        if "CO_ENTIDADE" in keep and len(keep) > 1:
            sql_cols = ", ".join(keep)
            frames.append(
                con.execute(
                    f"""
                    SELECT {sql_cols}
                    FROM read_csv_auto(
                        '{mat_path.as_posix()}',
                        delim=';',
                        header=True,
                        ignore_errors=True,
                        sample_size=-1
                    )
                    """
                ).df()
            )
    if doc_path is not None:
        doc_cols = _brasil_available_columns(doc_path)
        keep = [c for c in ("CO_ENTIDADE", "QT_DOC_BAS", "QT_DOC_FUND", "QT_DOC_MED") if c in doc_cols]
        if "CO_ENTIDADE" in keep and len(keep) > 1:
            sql_cols = ", ".join(keep)
            frames.append(
                con.execute(
                    f"""
                    SELECT {sql_cols}
                    FROM read_csv_auto(
                        '{doc_path.as_posix()}',
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


def _load_rendimento(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = STAGING_ROOT / "brasil" / "taxas_rendimento" / f"tx_rend_escolas_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing staged rendimento {path}. Run: python scripts/stage_inep_rendimento.py"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _load_censo_features(years: list[int]) -> pd.DataFrame:
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
        "IN_BANDA_LARGA",
        "IN_BIBLIOTECA",
        "IN_BIBLIOTECA_SALA_LEITURA",
        "IN_LABORATORIO_INFORMATICA",
        "IN_QUADRA_ESPORTES",
        "QT_MAT_BAS",
        "QT_MAT_FUND",
        "QT_MAT_MED",
        "QT_DOC_BAS",
        "QT_DOC_FUND",
        "QT_DOC_MED",
    ]
    frames: list[pd.DataFrame] = []
    con = duckdb.connect()
    for year in years:
        path = _find_brasil_school_csv(year)
        if path is None:
            continue
        available = _brasil_available_columns(path)
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
        # From 2025 onward INEP often splits enrollment/teacher counts into side tables.
        need_side = not {"QT_MAT_FUND", "QT_MAT_MED", "QT_MAT_BAS"} & set(df.columns)
        if need_side:
            side = _load_school_side_counts(year, con)
            if side is not None and "CO_ENTIDADE" in side.columns:
                key = "CO_ENTIDADE" if "CO_ENTIDADE" in df.columns else "CO_ESCOLA"
                df = df.merge(side, left_on=key, right_on="CO_ENTIDADE", how="left")
        out = pd.DataFrame()
        out["year"] = (
            pd.to_numeric(df["NU_ANO_CENSO"], errors="coerce")
            if "NU_ANO_CENSO" in df
            else year
        )
        out["school_id"] = pd.to_numeric(
            df["CO_ENTIDADE"] if "CO_ENTIDADE" in df else df["CO_ESCOLA"],
            errors="coerce",
        )
        out["uf"] = df["SG_UF"] if "SG_UF" in df else None
        out["municipio_id"] = (
            pd.to_numeric(df["CO_MUNICIPIO"], errors="coerce")
            if "CO_MUNICIPIO" in df
            else pd.NA
        )
        out["tp_dependencia"] = (
            pd.to_numeric(df["TP_DEPENDENCIA"], errors="coerce")
            if "TP_DEPENDENCIA" in df
            else pd.NA
        )
        out["tp_localizacao"] = (
            pd.to_numeric(df["TP_LOCALIZACAO"], errors="coerce")
            if "TP_LOCALIZACAO" in df
            else pd.NA
        )
        out["tp_situacao"] = (
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

        agua = num("IN_AGUA_POTAVEL")
        if agua.isna().all():
            agua = num("IN_AGUA_REDE_PUBLICA")
        out["in_agua"] = agua
        out["in_energia"] = num("IN_ENERGIA_REDE_PUBLICA")
        out["in_energia_inexistente"] = num("IN_ENERGIA_INEXISTENTE")
        out["in_esgoto"] = num("IN_ESGOTO_REDE_PUBLICA")
        out["in_esgoto_inexistente"] = num("IN_ESGOTO_INEXISTENTE")
        internet = num("IN_INTERNET")
        if internet.isna().all():
            internet = num("IN_BANDA_LARGA")
        out["in_internet"] = internet
        bib = num("IN_BIBLIOTECA")
        if bib.isna().all():
            bib = num("IN_BIBLIOTECA_SALA_LEITURA")
        out["in_biblioteca"] = bib
        out["in_lab_info"] = num("IN_LABORATORIO_INFORMATICA")
        out["in_quadra"] = num("IN_QUADRA_ESPORTES")
        out["qt_mat_bas"] = num("QT_MAT_BAS")
        out["qt_mat_fund"] = num("QT_MAT_FUND")
        out["qt_mat_med"] = num("QT_MAT_MED")
        out["qt_doc_bas"] = num("QT_DOC_BAS")
        out["qt_doc_fund"] = num("QT_DOC_FUND")
        out["qt_doc_med"] = num("QT_DOC_MED")
        out["is_rural"] = out["tp_localizacao"].fillna(1).astype(float).eq(2.0).astype(int)
        out["is_public"] = out["tp_dependencia"].fillna(4).astype(float).isin([1, 2, 3]).astype(int)
        if out["tp_situacao"].notna().any():
            out = out[out["tp_situacao"].fillna(1).astype(float).eq(1.0)].copy()
        frames.append(out)
    con.close()
    if not frames:
        raise RuntimeError("No Brasil school CSVs could be loaded")
    return pd.concat(frames, ignore_index=True)


def build_school_risk_br_level(level: str, years: list[int] | None = None) -> Path:
    """Build school×year mart for Fundamental or Médio with official abandonment target."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level}; expected {list(LEVELS)}")
    if years is None:
        years = discover_brasil_years(min_year=2016)
    meta = LEVELS[level]
    censo = _load_censo_features(years)
    rend = _load_rendimento(years)
    rend_keep = [
        "year",
        "school_id",
        "school_name",
        "dependencia",
        "localizacao",
        meta["target_col"],
    ]
    rend = rend[rend_keep].copy()
    rend = rend.rename(columns={meta["target_col"]: "taxa_abandono"})

    merged = censo.merge(rend, on=["year", "school_id"], how="inner", suffixes=("", "_rend"))
    n_censo = len(censo)
    n_rend = len(rend)
    n_join = len(merged)

    # Keep schools with a non-null official abandonment rate for this level
    merged = merged.loc[merged["taxa_abandono"].notna()].copy()

    # Prefer level-specific enrollment when available
    enroll = None
    for col in meta["enrollment_cols"]:
        key = col.lower() if col.startswith("QT_") else col
        # map QT_MAT_FUND -> qt_mat_fund
        snake = col.lower()
        if snake in merged.columns:
            enroll = merged[snake]
            break
        alt = {
            "QT_MAT_FUND": "qt_mat_fund",
            "QT_MAT_MED": "qt_mat_med",
            "QT_MAT_BAS": "qt_mat_bas",
        }.get(col)
        if alt and alt in merged.columns:
            enroll = merged[alt]
            break
    if enroll is None:
        enroll = pd.Series([pd.NA] * len(merged))
    merged["enrollment_level"] = pd.to_numeric(enroll, errors="coerce")
    merged = merged.loc[
        merged["enrollment_level"].fillna(0) >= meta["min_enrollment"]
    ].copy()

    merged["target_dropout_rate"] = merged["taxa_abandono"].astype(float)
    merged["target_definition"] = "inep_taxa_abandono_official"
    merged["education_level"] = level
    merged["country_iso3"] = "BRA"
    # Risk bands: many schools have 0% abandonment, so tertile cutpoints can collide.
    # Use rank percentiles within the level; fall back to absolute thresholds.
    ranks = merged["target_dropout_rate"].rank(method="average", pct=True)
    merged["risk_band"] = pd.Series("low", index=merged.index, dtype="object")
    merged.loc[ranks > 0.33, "risk_band"] = "moderate"
    merged.loc[ranks > 0.66, "risk_band"] = "high"
    # Also flag absolute elevated rates (useful when mass of zeros dominates ranks)
    merged.loc[merged["target_dropout_rate"] >= 5.0, "risk_band"] = "high"
    q66 = float(merged["target_dropout_rate"].quantile(0.66))
    merged["high_risk"] = (
        (ranks > 0.66) | (merged["target_dropout_rate"] >= max(q66, 5.0))
    ).astype(int)
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
            "target_definition": "inep_taxa_abandono_official",
            "target_source_column": meta["target_col"],
            "join_stats": {
                "censo_rows": int(n_censo),
                "rendimento_rows": int(n_rend),
                "inner_join_rows": int(n_join),
                "final_rows_with_target_and_min_enrollment": int(len(merged)),
                "join_match_rate_vs_censo": round(n_join / max(n_censo, 1), 4),
            },
            "risk_rule": {
                "rank_high_gt": 0.66,
                "absolute_high_pct": 5.0,
                "q66": q66,
            },
            "mean_taxa_abandono": float(merged["target_dropout_rate"].mean()),
            "public_share": float(merged["is_public"].mean()) if "is_public" in merged else None,
            "parquet": str(out_path),
        },
        out_dir / "manifest.json",
    )
    return out_path


def build_school_risk_br(years: list[int] | None = None) -> dict[str, Path]:
    """Build both Fundamental and Médio marts."""
    return {
        "fundamental": build_school_risk_br_level("fundamental", years=years),
        "medio": build_school_risk_br_level("medio", years=years),
    }
