# Brazil education data layout

Primary national data for the Brazil School Dropout Risk project.

```text
02_national/brasil/
  microdados_censo_escolar_YYYY/   # INEP Censo Escolar (features)
  taxas_rendimento/                # INEP Taxas de Rendimento (targets)
```

Staging: `latam_education_data/staging/brasil/taxas_rendimento/`  
Marts: `latam_education_data/marts/school_risk_br_{fundamental,medio}/`

Rebuild:

```bash
python scripts/download_inep_rendimento.py
python scripts/stage_inep_rendimento.py
python -m src.etl.run_etl
```
