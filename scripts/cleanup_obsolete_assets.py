"""Remove obsolete Kaggle / multi-country LATAM product assets after Brazil pivot."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DELETE_PATHS = [
    ROOT / "data" / "raw" / "dropoutKaggle.csv",
    ROOT / "latam_education_data" / "03_reference",
    ROOT / "latam_education_data" / "marts" / "school_risk_br",
    ROOT / "latam_education_data" / "marts" / "school_risk_co",
    ROOT / "latam_education_data" / "marts" / "indicators_latam",
    ROOT / "notebooks" / "01_eda_track_b_prototype.ipynb",
    ROOT / "notebooks" / "02_eda_track_a_latam.ipynb",
    ROOT / "app" / "main.py",
    ROOT / "scripts" / "patch_nb02_bootstrap.py",
    ROOT / "tests" / "test_data_and_features.py",
    ROOT / "scripts" / "smoke_inference.py",
]

# HE model artifacts at models root (keep models/fundamental and models/medio)
MODELS_ROOT_FILES = [
    "pipeline.joblib",
    "feature_list.json",
    "metrics.json",
    "model_card_meta.json",
    "shap_background.parquet",
]

NATIONAL_NON_BR = [
    ROOT / "latam_education_data" / "02_national" / "argentina",
    ROOT / "latam_education_data" / "02_national" / "colombia",
    ROOT / "latam_education_data" / "02_national" / "uruguay",
]

REGIONAL = ROOT / "latam_education_data" / "01_regional"
STAGING_NON_BR = ROOT / "latam_education_data" / "staging"


def rm(path: Path) -> None:
    if not path.exists():
        print("skip missing", path)
        return
    if path.is_file():
        path.unlink()
        print("deleted file", path)
    else:
        shutil.rmtree(path, ignore_errors=True)
        print("deleted dir", path)


def main() -> None:
    for p in DELETE_PATHS:
        rm(p)
    models = ROOT / "models"
    if models.exists():
        for name in MODELS_ROOT_FILES:
            rm(models / name)
        figs = models / "figures"
        if figs.exists():
            rm(figs)
    for p in NATIONAL_NON_BR:
        rm(p)
    if REGIONAL.exists():
        rm(REGIONAL)
    if STAGING_NON_BR.exists():
        for child in STAGING_NON_BR.iterdir():
            if child.name != "brasil":
                rm(child)
    # obsolete ETL modules that only served LATAM panel
    for name in [
        "normalize_sources.py",
        "build_indicators_mart.py",
        "dq_checks.py",
        "ingest_validate.py",
    ]:
        rm(ROOT / "src" / "etl" / name)
    print("cleanup done")


if __name__ == "__main__":
    main()
