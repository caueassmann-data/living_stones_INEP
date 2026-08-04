# Marts — Brazil school risk

## Level-specific marts

| Mart | Level | Target |
|---|---|---|
| `school_risk_br_fundamental/` | Ensino Fundamental | official INEP dropout rate (`3_CAT_FUN`) |
| `school_risk_br_medio/` | Ensino Medio | official INEP dropout rate (`3_CAT_MED`) |

Each folder contains:

- `*.parquet` — analysis/training table (English column schema — see
  `docs/data_card.md`)
- `manifest.json` — row counts, join stats, history-feature coverage, risk-band thresholds

## Rebuild

```bash
python scripts/stage_inep_rendimento.py
# Default years = intersection of staged attainment-rate years and Census years (currently 2018-2025)
python -m src.etl.run_etl
```

## Query example

```python
import pandas as pd
fund = pd.read_parquet("latam_education_data/marts/school_risk_br_fundamental/school_risk_br_fundamental.parquet")
print(fund[["year", "school_id", "state_code", "target_dropout_rate", "dropout_rate_lag1", "enrollment_level"]].head())
```
