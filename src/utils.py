"""Project paths, constants, and the shared feature schema.

Read this file first if you are new to the project: every other module
imports its column names, paths, and the official limitation statement from
here, so this is the single source of truth for "what a school-year row
looks like."

Scope: Brazil only, basic education only (Ensino Fundamental and Ensino
Medio), school-level (not individual students). See LIMITATION_STATEMENT
below for the exact wording used in every app and doc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# NOTE on Path.resolve(): deliberately avoided project-wide. On this machine
# the repo lives under Desktop, which OneDrive syncs; Path.resolve() (and
# Path.exists() on some drives) has been observed to hang Jupyter kernels
# under OneDrive sync locks. Plain Path composition below is intentional,
# not an oversight.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "latam_education_data"
MARTS_ROOT = DATA_ROOT / "marts"
# Slim, committed copies of the marts holding only the most recent years, so a
# deployed app can run from a plain `git clone`. The full marts are gigabytes
# of build output and stay out of version control — see build_deploy_artifacts.
APP_MARTS_ROOT = DATA_ROOT / "marts_app"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

RANDOM_STATE = 42
MODEL_VERSION = "0.3.0"

LIMITATION_STATEMENT = (
    "This early-warning prototype predicts school-level dropout rates for Brazilian "
    "basic education (Ensino Fundamental and Ensino Medio) using the official INEP "
    "school-attainment rates joined to School Census features, including each "
    "school's own dropout history. It does not score individual students and is "
    "not a multi-country LATAM model."
)

DATASET_DISPLAY_NAME = "INEP School Census + School Attainment Rates (Brazil)"

# "fundamental" and "medio" are kept as-is rather than translated: they name
# Brazil's own two stages of basic education and do not map cleanly onto a
# single English school-system term.
#   Ensino Fundamental ("fundamental"): grades 1-9, roughly ages 6-14.
#   Ensino Medio ("medio"): grades 10-12, Brazil's upper-secondary / high-school stage.
EDUCATION_LEVELS = ("fundamental", "medio")

LEVEL_DISPLAY_NAME = {
    "fundamental": "Ensino Fundamental (grades 1-9, ages ~6-14)",
    "medio": "Ensino Medio (grades 10-12, Brazil's upper-secondary stage)",
}

# School years in which INEP suspended its normal grade-progression rules
# because of COVID-19 school closures. Attainment rates in these years follow
# different administrative rules (e.g. blanket promotion in many networks),
# so they are flagged as a feature (is_covid_year) rather than silently
# treated as a normal year.
COVID_YEARS = frozenset({2020, 2021})

# --- Feature schema -------------------------------------------------------
# Every column below is available *before* the target is known: either it is
# a Census feature captured on Census day (which precedes the "taxa de
# abandono" outcome window — see docs/data_card.md), or it is a lagged /
# historical statistic built only from years strictly before the row's own
# year (see src/etl/build_school_risk_marts.py "history features"). None of
# it depends on the current year's own attainment outcome.

# Nominal (unordered) codes: modeled with one-hot encoding, never scaled.
CATEGORICAL_FEATURE_COLUMNS = [
    "admin_dependency_type",  # INEP TP_DEPENDENCIA: 1 Federal, 2 State, 3 Municipal, 4 Private
    "location_type",  # INEP TP_LOCALIZACAO: 1 Urban, 2 Rural
    "state_code",  # Brazilian state abbreviation, e.g. "SP", "RJ" ("UF" = Unidade Federativa)
]

# Continuous / binary features: imputed (median) and scaled.
NUMERIC_FEATURE_COLUMNS = [
    "is_rural",
    "is_public",
    "has_water",
    "has_electricity",
    "electricity_absent",
    "has_sewage",
    "sewage_absent",
    "has_internet",
    "has_internet_for_students",
    "has_library",
    "has_computer_lab",
    "has_sports_court",
    "enrollment_basic_ed",
    "enrollment_level",
    "teacher_count_basic_ed",
    "student_teacher_ratio",
    "avg_class_size",
    "overage_enrollment_share",
    "eja_enrollment_share",
    "is_covid_year",
    # History features (the fix for the "no autocorrelation used" gap):
    # the school's own dropout-rate trajectory, and its local (municipality /
    # state) neighborhood, using only years strictly before the row's year.
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

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def mart_path(level: str) -> Path:
    """The full mart: every year, built locally by the ETL. Never committed.

    Training and evaluation must always use this one — never the slim deploy
    copy below, which would silently train on a two-year window.
    """
    name = f"school_risk_br_{level}"
    return MARTS_ROOT / name / f"{name}.parquet"


def app_mart_path(level: str) -> Path:
    """The slim, committed mart used when the full one is not on disk."""
    return APP_MARTS_ROOT / f"school_risk_br_{level}_recent.parquet"


def resolve_app_mart_path(level: str) -> Path:
    """Mart for the Streamlit app: the full one locally, the slim one deployed.

    The app derives its year picker from whatever it loads, so it adapts to
    either without a code change. Only read-only app code may call this; see
    mart_path() for why training must not.
    """
    full = mart_path(level)
    try:
        if full.exists():
            return full
    except OSError:
        # Path.exists() has been observed to hang or raise under OneDrive sync
        # locks on this project's Desktop checkout (see the note above).
        pass
    return app_mart_path(level)


def model_dir(level: str) -> Path:
    return MODELS_DIR / level


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
