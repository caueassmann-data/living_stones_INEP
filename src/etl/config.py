from __future__ import annotations

import re
from pathlib import Path

# Avoid Path.resolve(): on OneDrive/Desktop it can hang notebook kernels.
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "latam_education_data"
RAW_ROOT = DATA_ROOT / "raw"
STAGING_ROOT = DATA_ROOT / "staging"
MARTS_ROOT = DATA_ROOT / "marts"
DQ_ROOT = DATA_ROOT / "dq"
META_ROOT = DATA_ROOT / "meta"

# Fallback only; prefer discover_brasil_years().
BRASIL_YEARS_DEFAULT: list[int] | None = None


def _censo_years() -> set[int]:
    years: set[int] = set()
    roots = [
        RAW_ROOT / "national" / "brasil",
        DATA_ROOT / "02_national" / "brasil",
    ]
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("microdados_censo_escolar_*"):
            m = re.search(r"(20\d{2})", p.name)
            if m:
                years.add(int(m.group(1)))
    return years


def _rendimento_staged_years() -> set[int]:
    years: set[int] = set()
    staging = STAGING_ROOT / "brasil" / "taxas_rendimento"
    if not staging.exists():
        return years
    for p in staging.glob("tx_rend_escolas_*.parquet"):
        m = re.search(r"(20\d{2})", p.name)
        if m:
            years.add(int(m.group(1)))
    return years


def discover_brasil_years(*, min_year: int = 2016) -> list[int]:
    """Intersection of staged rendimento years and available Censo years."""
    years = sorted(
        y
        for y in (_censo_years() & _rendimento_staged_years())
        if y >= min_year
    )
    if not years:
        raise RuntimeError(
            "No overlapping Brasil years found. "
            "Run: python scripts/download_inep_rendimento.py && "
            "python scripts/stage_inep_rendimento.py"
        )
    return years


def ensure_dirs() -> None:
    for p in [
        STAGING_ROOT / "brasil" / "taxas_rendimento",
        MARTS_ROOT / "school_risk_br_fundamental",
        MARTS_ROOT / "school_risk_br_medio",
        DQ_ROOT,
        META_ROOT,
    ]:
        p.mkdir(parents=True, exist_ok=True)
