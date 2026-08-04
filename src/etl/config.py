"""Paths and year-discovery helpers for the Brazil school-risk ETL."""

from __future__ import annotations

import re
from pathlib import Path

# See src/utils.py for the note on why Path.resolve() is avoided project-wide
# (OneDrive/Desktop sync locks have been observed to hang notebook kernels).
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "latam_education_data"
RAW_ROOT = DATA_ROOT / "raw"
STAGING_ROOT = DATA_ROOT / "staging"
MARTS_ROOT = DATA_ROOT / "marts"
DQ_ROOT = DATA_ROOT / "dq"
META_ROOT = DATA_ROOT / "meta"


def _census_years() -> set[int]:
    """Years for which raw School Census extracts are present on disk."""
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


def _attainment_staged_years() -> set[int]:
    """Years for which staged INEP attainment-rate Parquet files are present."""
    years: set[int] = set()
    staging = STAGING_ROOT / "brasil" / "taxas_rendimento"
    if not staging.exists():
        return years
    for p in staging.glob("tx_rend_escolas_*.parquet"):
        m = re.search(r"(20\d{2})", p.name)
        if m:
            years.add(int(m.group(1)))
    return years


def discover_brazil_years(*, min_year: int = 2016) -> list[int]:
    """Intersection of staged attainment years and available Census years.

    Both sources are required to build a mart row (Census features + the
    official dropout-rate target), so only years where both exist are usable.
    """
    years = sorted(y for y in (_census_years() & _attainment_staged_years()) if y >= min_year)
    if not years:
        raise RuntimeError(
            "No overlapping Brazil years found. "
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
