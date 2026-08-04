# Scope revision — response to team-lead feedback

**Project:** Brazil School Dropout Risk (Fundamental & Médio)  
**Publisher:** Living Stone Foundation — Applied Data Lab  
**Date:** 2026-07-29

## Why we revised the scope

Feedback asked us to:

1. Align the **target population** with the **training data**
2. Narrow from multi-country LATAM to **one country**
3. Validate variables, dropout definition, and prediction grain
4. Let modeling / XAI / Streamlit follow from validated data (not the reverse)

## Validated findings

### Population vs data alignment

| Previous prototype | Revised scope |
|---|---|
| Kaggle higher-education students (non-LATAM) | Brazilian **basic education** schools |
| Student-level binary dropout | **School × year** official abandonment rate |
| Multi-country LATAM indicators as parallel track | Brazil-only product core |

### Country and education levels

- **Country:** Brazil (richest open data already in the repo: Censo Escolar 2016–2025 + INEP rendimento staged for **2018–2025**; 2016–2017 rendimento unavailable on the download path used).
- **Levels:** **Ensino Fundamental** and **Ensino Médio** as **two parallel products** (2 models, 2 Streamlit apps).
  - Rationale: abandonment means and correlates differ (current mart means ≈ **1.1%** Fundamental vs ≈ **3.3%** Médio). Pooling would mix regimes.
  - This is still a **narrowed** scope vs multi-country LATAM; it is intentionally dual-level inside one country.

### Dropout definition (official)

INEP **taxa de abandono** from Taxas de Rendimento Escolar:

> Percentage of students who stopped attending after the School Census reference date during the school year (Situação do Aluno).

Source columns: `3_CAT_FUN` (Fundamental), `3_CAT_MED` (Médio).  
Join key to Censo: `CO_ENTIDADE` + year.

Artifact: `latam_education_data/dq/rendimento_validation.json`

### Prediction grain

**School-level**, not student-level.

Open INEP downloads used here provide school aggregates for abandonment. They do **not** provide a labeled student panel for Fundamental/Médio dropout. Claiming student-level prediction would be scientifically dishonest with the current open data.

### Variables available (MVP feature families)

From Censo Escolar (examples): administrative dependence, urban/rural, water/energy/sewage/internet, library/lab/sports court, enrollment and teacher counts, student–teacher ratio.

Target from rendimento: continuous abandonment rate (%).

### Modeling / XAI / Streamlit (chosen after validation)

- Task: **regression** of school abandonment rate (MAE/RMSE/R²).
- Candidates evaluated fairly: Ridge, RandomForest, XGBoost → winner stored per level in `models/*/metrics.json`.
- **Observed winners (v0.2.0 sample train):** Fundamental → **XGBoost** (test MAE ≈ 1.15); Médio → **RandomForest** (test MAE ≈ 3.08).
- **Driver contrast (supports two models):** Fundamental importance skewed toward infrastructure/location (`in_energia`, `is_rural`, `in_internet`); Médio skewed toward scale/staffing (`qt_mat_bas`, `student_teacher_ratio`, `enrollment_level`).
- XAI: native importances / |coefficients| (permutation fallback).
- Apps: `app/fundamental/main.py` and `app/medio/main.py`.

## What we removed

- Kaggle higher-ed training path as product core
- Multi-country LATAM indicator warehouse as product core
- Structural vulnerability **proxy** sold as abandonment
- Colombia school mart from the active ETL path

## Ask for approval to proceed

We propose continuing with this Brazil dual-level school early-warning MVP, with clear limitation statements in README, apps, and model card.
