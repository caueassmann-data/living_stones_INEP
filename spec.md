# TECHNICAL SPECIFICATION
## Brazil School Dropout Risk (Fundamental & Medio)
### Early warning of school-level dropout in Brazilian basic education

| Field | Value |
|---|---|
| **Publisher** | Living Stone Foundation — Applied Data Lab |
| **Document** | `spec.md` |
| **Status** | Revised after a validity review found the models did not beat a trivial baseline — see `docs/validity_and_english_revision.md` |
| **Version** | `3.0.0` |
| **Language** | English (canonical: code, data columns, and documentation) |
| **Random seed** | `RANDOM_STATE = 42` |
| **Data** | INEP School Census + School Attainment Rates (Brazil) |

---

## 1. Mission

Build a **proactive, school-level** decision-support prototype that:

1. Uses **official** Brazilian dropout rates (`taxa de abandono`) as the modeling target.
2. Trains **separate** models for **Ensino Fundamental** and **Ensino Medio** (drivers differ).
3. Uses each school's own dropout-rate history as a predictor, not just its
   current-year Census snapshot — the single strongest signal available for
   this problem, and the main gap closed in v3.0.0.
4. Explains predictors with model-appropriate XAI, and reports triage-ranking
   quality (not just point-error metrics) because the product is used to
   prioritize, not to forecast exact rates.
5. Never reports a "winning" model without first checking it beats a trivial
   baseline (always-predict-the-mean), on both a held-out-schools split and a
   held-out-future-years split.
6. Serves triage via **two Streamlit apps** (plus a unified navigation
   entrypoint, `app/main.py`).

## 2. Scope boundaries (non-negotiable)

| In scope | Out of scope |
|---|---|
| Brazil only | Multi-country LATAM warehouse as product core |
| Basic education (Fundamental + Medio) | Higher-education Kaggle student dataset |
| School x year predictions | Student-level dropout scoring (labels not available in the open INEP extracts used here) |
| Official INEP dropout rates | Structural vulnerability proxies sold as dropout |

## 3. Dual product tracks

| Track | Mart | Model dir | App |
|---|---|---|---|
| Fundamental | `marts/school_risk_br_fundamental/` | `models/fundamental/` | `app/fundamental/main.py` |
| Medio | `marts/school_risk_br_medio/` | `models/medio/` | `app/medio/main.py` |

Hard rule: do not pool Fundamental and Medio into one training table for the primary models.

## 4. Target definition

**Official INEP dropout rate** ("taxa de abandono") from the School Attainment
Rates ("Taxas de Rendimento Escolar"):

> Percentage of students who stopped attending after the School Census
> reference date during the school year (student movement status = left
> attending).

- Fundamental source column: `3_CAT_FUN` -> staged as `dropout_rate_fundamental`
- Medio source column: `3_CAT_MED` -> staged as `dropout_rate_medio`
- Join keys: `school_id` (`CO_ENTIDADE`) + `year`

Same-year Census features are **not** leakage against this target: the Census
snapshot is captured before the reference date the dropout-rate window starts
from. See `docs/data_card.md` for the full argument. What must never be used
is a school's own *current-year* attainment numbers as a predictor — only
strictly-prior-year attainment history is used (see section 5).

## 5. Modeling policy

Algorithm family is **selected from grouped, held-out CV performance**
(candidates: `dummy_mean`, `dummy_median`, Ridge, RandomForest, XGBoost). Do
not hard-code a winner in advance. The two dummy baselines are never eligible
to be selected as the winner — they exist solely as a floor every real
candidate must clear (`beats_baseline_cv`, `beats_baseline_test` in
`models/{level}/metrics.json`).

Splitting:
- Primary: `GroupShuffleSplit` / `GroupKFold` by `school_id` (a school's rows
  never straddle both sides of a split).
- Secondary, out-of-time: train on years <= 2023, test on 2024-2025.

Primary metrics: **MAE**, **RMSE**, **R2** on held-out rows, on both splits
above. Secondary metrics: precision@k / recall@k / lift / average precision
against the mart's `high_risk` label (triage-ranking quality).

XAI: native importances / absolute coefficients, with permutation importance
as fallback.

## 6. Required limitation statement

> This early-warning prototype predicts school-level dropout rates for
> Brazilian basic education (Ensino Fundamental and Ensino Medio) using the
> official INEP school-attainment rates joined to School Census features,
> including each school's own dropout history. It does not score individual
> students and is not a multi-country LATAM model.

## 7. Definition of Done (v3.0.0)

- [x] Download & stage INEP attainment rates 2018-2025 (2016-2017 unavailable
      on the INEP download path used)
- [x] Dual marts with the official target and full English column schema
      (see `docs/validity_and_english_revision.md`)
- [x] Trivial-baseline check wired into training and surfaced in every app
- [x] Group- and temporal-holdout evaluation (previous version used a
      row-level random split, which let the same school leak across train/test)
- [x] History features (lagged dropout/approval/failure rates, municipal and
      state spatial lags) — previously absent entirely
- [x] Two trained models with metrics, ranking metrics, and importance artifacts
- [x] Two Streamlit entrypoints plus a unified navigation entrypoint
- [x] Docs rewritten in English throughout (code, columns, comments, docs)
- [x] `docs/scope_revision.md` (2026-07-29 scope pivot) and
      `docs/validity_and_english_revision.md` (this revision) for the team lead
