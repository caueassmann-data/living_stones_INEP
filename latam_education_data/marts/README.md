# Marts — Brazil school risk

## Level-specific marts

| Mart | Level | Target |
|---|---|---|
| `school_risk_br_fundamental/` | Ensino Fundamental | official INEP `taxa_abandono` (`3_CAT_FUN`) |
| `school_risk_br_medio/` | Ensino Médio | official INEP `taxa_abandono` (`3_CAT_MED`) |

Each folder contains:

- `*.parquet` — analysis/training table
- `manifest.json` — row counts, join stats, risk rules

## Rebuild

```bash
python scripts/stage_inep_rendimento.py
# Default years = intersection of staged rendimento ∩ Censo (currently 2018–2025)
python -m src.etl.run_etl
```

## Query example

```python
import pandas as pd
fund = pd.read_parquet("latam_education_data/marts/school_risk_br_fundamental/school_risk_br_fundamental.parquet")
print(fund[["year","school_id","uf","target_dropout_rate","enrollment_level"]].head())
```
