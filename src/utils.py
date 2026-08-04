"""Project paths, constants, and limitation statement (Brazil school-level)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Avoid Path.resolve(): on OneDrive/Desktop it can hang notebook kernels.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "latam_education_data"
MARTS_ROOT = DATA_ROOT / "marts"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

RANDOM_STATE = 42
MODEL_VERSION = "0.2.0"

LIMITATION_STATEMENT = (
    "This early-warning prototype predicts school-level abandonment rates for Brazilian "
    "basic education (Ensino Fundamental and Ensino Médio) using official INEP Taxas de "
    "Rendimento joined to Censo Escolar school features. It does not score individual "
    "students and is not a multi-country LATAM model."
)

DATASET_DISPLAY_NAME = "INEP Censo Escolar + Taxas de Rendimento (Brazil)"

EDUCATION_LEVELS = ("fundamental", "medio")

FEATURE_COLUMNS = [
    "tp_dependencia",
    "tp_localizacao",
    "is_rural",
    "is_public",
    "in_agua",
    "in_energia",
    "in_esgoto",
    "in_internet",
    "in_biblioteca",
    "in_lab_info",
    "in_quadra",
    "qt_mat_bas",
    "enrollment_level",
    "qt_doc_bas",
    "student_teacher_ratio",
]


def mart_path(level: str) -> Path:
    name = f"school_risk_br_{level}"
    return MARTS_ROOT / name / f"{name}.parquet"


def model_dir(level: str) -> Path:
    return MODELS_DIR / level


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
