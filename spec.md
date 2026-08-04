# TECHNICAL SPECIFICATION
## Brazil School Dropout Risk (Fundamental & Médio)
### Early warning of school-level abandonment in Brazilian basic education

| Field | Value |
|---|---|
| **Publisher** | Living Stone Foundation — Applied Data Lab |
| **Document** | `spec.md` |
| **Status** | Revised scope after team-lead feedback |
| **Version** | `2.0.0` |
| **Language** | English (canonical) |
| **Random seed** | `RANDOM_STATE = 42` |
| **Data** | INEP Censo Escolar + Taxas de Rendimento (Brazil) |

---

## 1. Mission

Build a **proactive, school-level** decision-support prototype that:

1. Uses **official** Brazilian abandonment rates (`taxa de abandono`) as the modeling target.
2. Trains **separate** models for **Ensino Fundamental** and **Ensino Médio** (drivers differ).
3. Explains predictors with model-appropriate XAI.
4. Serves triage via **two Streamlit apps**.

## 2. Scope boundaries (non-negotiable)

| In scope | Out of scope |
|---|---|
| Brazil only | Multi-country LATAM warehouse as product core |
| Basic education (Fundamental + Médio) | Higher-education Kaggle student dataset |
| School × year predictions | Student-level dropout scoring (labels not available in open INEP extracts used here) |
| Official INEP abandonment rates | Structural vulnerability proxies sold as abandonment |

## 3. Dual product tracks

| Track | Mart | Model dir | App |
|---|---|---|---|
| Fundamental | `marts/school_risk_br_fundamental/` | `models/fundamental/` | `app/fundamental/main.py` |
| Médio | `marts/school_risk_br_medio/` | `models/medio/` | `app/medio/main.py` |

Hard rule: do not pool Fundamental and Médio into one training table for the primary models.

## 4. Target definition

**Official INEP taxa de abandono** from Taxas de Rendimento Escolar:

> Percentage of students who stopped attending after the School Census reference date during the school year (Situação do Aluno movement = left attending).

- Fundamental target column: `3_CAT_FUN`
- Médio target column: `3_CAT_MED`
- Join keys: `CO_ENTIDADE` (`school_id`) + `year`

## 5. Modeling policy

Algorithm family is **selected from held-out / CV performance** (Ridge, RandomForest, XGBoost regressors). Do not hard-code a winner in advance.

Primary metrics: **MAE**, **RMSE**, **R²** on held-out schools/years rows.

XAI: native importances / absolute coefficients, with permutation importance as fallback.

## 6. Required limitation statement

> This early-warning prototype predicts school-level abandonment rates for Brazilian basic education (Ensino Fundamental and Ensino Médio) using official INEP Taxas de Rendimento joined to Censo Escolar school features. It does not score individual students and is not a multi-country LATAM model.

## 7. Definition of Done (MVP)

- [x] Download & stage INEP rendimento 2018–2025 (2016–2017 unavailable on INEP URL)
- [x] Dual marts with official target (proxy retired)
- [x] Two trained models with metrics + importance artifacts
- [x] Two Streamlit entrypoints
- [x] Docs rewritten; obsolete Kaggle/LATAM claims removed
- [x] `docs/scope_revision.md` for team lead
