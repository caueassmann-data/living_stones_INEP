# Data Card — Brazil School Abandonment Marts

## Sources

1. **INEP Censo Escolar** (school microdata) — features (infrastructure, dependency, location, enrollment/teachers).
2. **INEP Taxas de Rendimento Escolar** — official approval / failure / **abandonment** rates.

Years in current marts: **2018–2025** (all years ≥ 2016 where both staged INEP rendimento and Censo school microdata join successfully; 2016–2017 rendimento was not available on the INEP download path used).

## Grain

`school_id` (`CO_ENTIDADE`) × `year` × `education_level`

## Targets

| Level | Source column | Mart column |
|---|---|---|
| Fundamental | `3_CAT_FUN` | `target_dropout_rate` |
| Médio | `3_CAT_MED` | `target_dropout_rate` |

`target_definition = inep_taxa_abandono_official`

### Definition (plain language)

Share of students who **left attending** during the school year after the census reference date. School-reported via Situação do Aluno. **Not** a student longitudinal ID panel.

## Artifacts

- `latam_education_data/marts/school_risk_br_fundamental/school_risk_br_fundamental.parquet`
- `latam_education_data/marts/school_risk_br_medio/school_risk_br_medio.parquet`
- Validation: `latam_education_data/dq/rendimento_validation.json`

## What is unavailable

- Open **student-level** labeled dropout for Brazilian Fundamental/Médio in this repository.
- Treating infrastructure proxies as abandonment (retired).

## Known limitations

1. Join coverage is incomplete vs full Censo universe (see mart manifests `join_stats`).
2. Many schools have 0% abandonment → skewed target.
3. Private vs public schools differ sharply; interpret stratified results carefully.
4. Predictors are mostly structural/administrative — not classroom pedagogy or household income microdata.
