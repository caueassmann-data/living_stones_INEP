# TECHNICAL SPECIFICATION
## Brazil School Dropout Risk (Fundamental & Medio)
### Early warning of school-level dropout in Brazilian basic education

| Field | Value |
|---|---|
| **Publisher** | Living Stone Foundation — Applied Data Lab |
| **Document** | `spec.md` |
| **Status** | Revised after a validity review found the models did not beat a trivial baseline — see `docs/validity_and_english_revision.md` |
| **Version** | `3.1.0` |
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
against the mart's `high_risk` label (national triage-ranking quality), plus
precision@N **within each education network** (`test_ranking_within_network`),
which is the metric that matches the deployed decision rule below. Networks no
larger than N are excluded from it: their precision@N equals their base rate by
construction.

XAI: native importances / absolute coefficients, with permutation importance
as fallback.

## 5b. Decision rule (product)

The rule the product applies is **capacity-constrained Top-N within one
education network** (`src/prioritize.py`), decided by the Foundation in
Milestone 1 and recorded in `docs/milestone1_high_risk_criteria.md` §6:

1. Filter to one scope — a network the field team actually works in.
2. Rank by `pred_dropout_rate`, with a deterministic tie-break.
3. Take the top N, where **N is follow-up capacity**, set by the field
   implementation team. Default 50; presets 20 / 50 / 100; adjustable.

Non-negotiables:
- **No threshold on the dropout rate** as the product rule. A fixed cut is not
  comparable between Fundamental and Medio, which is why it was rejected.
- **N is a count, never a percentage.** A percentile cut reintroduces the same
  cross-level incomparability.
- **`high_risk` / `assign_risk_bands` stay exactly as they are** — a historical
  and evaluation label, never shown to end users.
- **Scope granularity is selectable, and the default is the state network.**
  The median municipal network holds 3 schools, so N = 50 there is not a
  constraint; when the chosen scope is smaller than N, the product must say so
  rather than present the whole network as prioritized.

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

## 8. Definition of Done (v3.1.0 — Milestone 1 tasks 6-7)

- [x] Top-N decision rule implemented as a reusable, tested module
      (`src/prioritize.py`, `tests/test_prioritize.py`), separate from the
      model artifacts so the rule can change without touching inference
- [x] Scope hierarchy (state network / municipality / municipal network /
      pooled municipalities) with the state network as default, and an explicit
      message whenever the scope is smaller than N
- [x] N adjustable by the end user, default 50, presets 20 / 50 / 100
- [x] `high_risk` removed from the end-user surface and unchanged in the ETL
      (`tests/test_risk_bands.py` passes unmodified)
- [x] Full-selection scoring — no silent row cap before ranking
- [x] Deterministic tie-break, verified by test and on the real 2025 marts
- [x] Within-network ranking metric in `models/{level}/metrics.json`
- [x] Network-level evidence for the Foundation
      (`scripts/analyze_network_prioritization.py`)
- [ ] Pilot network chosen with a committed stakeholder, and N sized against
      their actual capacity (blocked on the Foundation)
