# Data Card — Brazil School Dropout Marts

## Sources

1. **INEP School Census** ("Censo Escolar") — school microdata: infrastructure,
   administrative network, location, enrollment/teacher counts.
2. **INEP School Attainment Rates** ("Taxas de Rendimento Escolar") — official
   approval, failure (grade repetition), and **dropout** rates per school.

INEP ("Instituto Nacional de Estudos e Pesquisas Educacionais Anisio Teixeira")
is Brazil's national education-statistics agency; both sources are published
by them and are the same official data Brazil's own education authorities use.

Years in the current marts: **2018-2025** (all years >= 2016 where both the
staged attainment-rate file and Census microdata are available and join
successfully; 2016-2017 attainment rates were not available on the INEP
download path used).

## Grain

`school_id` (`CO_ENTIDADE`) x `year` x `education_level`

## Target

| Level | Source column | Mart column |
|---|---|---|
| Fundamental | `3_CAT_FUN` | `target_dropout_rate` |
| Medio | `3_CAT_MED` | `target_dropout_rate` |

`target_definition = inep_official_dropout_rate`

### Definition (plain language)

Share of students who **stopped attending** during the school year, after the
Census reference date. Reported by each school via the "Situacao do Aluno"
("Student Status") module of the School Census. **Not** a student-level
longitudinal ID panel — INEP does not publish one for this population.

### Why Census features from the same year are not leakage

INEP's own definition places the dropout-rate outcome window *after* the
Census reference date. A school's Census snapshot (infrastructure, enrollment,
staffing — captured on Census day) therefore genuinely precedes the outcome it
is used to predict. What would be leakage is using a school's *own current-year*
attainment numbers (its own dropout/approval/failure rate for that same year)
as a predictor — this pipeline never does that. All attainment-derived
features are historical: built only from years strictly earlier than the row's
own year (see "History features" below and
`src/etl/build_school_risk_marts.py::_add_history_features`).

## History features (added in v0.3.0)

Earlier versions of this pipeline used only same-year Census features and
never gave the model access to a school's own dropout-rate track record — the
single strongest predictor available for this kind of problem. Every mart row
now also carries, computed from strictly earlier years only:

- `dropout_rate_lag1` / `dropout_rate_lag2` — the school's own dropout rate 1
  and 2 years earlier.
- `dropout_rate_3yr_avg` — trailing 3-year average.
- `dropout_rate_trend` — `lag1 - lag2` (positive = rate rising).
- `has_history` — 0 for a school's first year in the panel (distinguishes
  "no track record yet" from "had 0% dropout last year").
- `approval_rate_lag1` / `failure_rate_lag1` — the school's own prior-year
  promotion and grade-repetition rates (a rising failure rate is a
  well-established leading indicator of dropout the following year).
- `municipal_dropout_rate_lag1` / `state_dropout_rate_lag1` — the average
  dropout rate across the school's municipality / state the prior year, a
  spatial-lag signal for local shocks a single school's own history cannot
  capture.

Coverage: roughly 86% of rows in both marts have at least one prior year of
history (see `history_feature_coverage` in each mart's `manifest.json`); the
remaining ~14% are schools' first appearance in the 2018-2025 window.

## Other feature families

From the School Census: administrative network (federal/state/municipal/
private), urban/rural location, water/electricity/sewage/internet access
(including internet specifically for students), library/computer-lab/sports-
court presence, enrollment and teacher counts, student-teacher ratio, average
class size, share of over-age students (a proxy for grade-age distortion, an
established dropout correlate in the Brazilian education literature), and
share of adult-education (EJA) enrollment.

An `is_covid_year` flag marks 2020-2021, when INEP suspended normal
grade-progression rules because of pandemic school closures — attainment
rates in those years follow different administrative rules and should not be
read the same way as a normal year.

## Artifacts

- `latam_education_data/marts/school_risk_br_fundamental/school_risk_br_fundamental.parquet`
- `latam_education_data/marts/school_risk_br_medio/school_risk_br_medio.parquet`
- Validation: `latam_education_data/dq/rendimento_validation.json`
- Each mart's `manifest.json` (join stats, history-feature coverage, risk-band
  thresholds) sits alongside its Parquet file.

## What is unavailable

- Open **student-level** labeled dropout data for Brazilian Fundamental/Medio
  in this repository.
- Treating infrastructure proxies as dropout (retired in v2.0.0).

## Known limitations

1. Join coverage is incomplete versus the full Census universe (see each
   mart's `manifest.json` -> `join_stats`).
2. A large share of schools report 0% dropout in a given year (roughly
   65% for Fundamental, 46% for Medio), which skews the target distribution;
   see `docs/validity_and_english_revision.md` for how this affects
   evaluation and risk-band assignment.
3. Public and private schools differ sharply; interpret pooled results with
   that in mind (see `docs/model_card.md` -> "Equity notes").
4. Predictors are mostly structural/administrative — not classroom pedagogy
   or household income microdata.
5. 2020-2021 attainment rates reflect COVID-era administrative rules, not
   ordinary-year dynamics (flagged via `is_covid_year`).
