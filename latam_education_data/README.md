# Brazil education data layout

Primary national data for the Brazil School Dropout Risk project.

The folder names below (`02_national`, `brasil`, `microdados_censo_escolar_*`,
`taxas_rendimento`) keep INEP's original directory naming so this layout stays
compatible with data already downloaded by `scripts/download_inep_rendimento.py`.
Everything read *from* these files (column names, labels) is translated to
English during staging — see `scripts/stage_inep_rendimento.py`.

```text
02_national/brasil/
  microdados_censo_escolar_YYYY/   # INEP School Census (features)
  taxas_rendimento/                # INEP School Attainment Rates (targets)
```

Staging: `latam_education_data/staging/brasil/taxas_rendimento/`  
Marts: `latam_education_data/marts/school_risk_br_{fundamental,medio}/`

Rebuild:

```bash
python scripts/download_inep_rendimento.py
python scripts/stage_inep_rendimento.py
python -m src.etl.run_etl
```
