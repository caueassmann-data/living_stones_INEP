# Brazil School Dropout Risk (Fundamental & Médio)

Living Stone Foundation — Applied Data Lab

School-level early-warning **prototype** for Brazilian basic education abandonment rates:

- **Ensino Fundamental** — model + Streamlit app
- **Ensino Médio** — model + Streamlit app

Population and labels are aligned: **Brazil**, **basic education**, **official INEP taxa de abandono**, grain **school × year**.

## Important limitation

This prototype predicts **school-level** abandonment rates using INEP Taxas de Rendimento joined to Censo Escolar features. It does **not** score individual students and is **not** a multi-country LATAM model.

## Quickstart (venv recommended)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Rebuild data marts (after downloading rendimento)

```bash
python scripts/download_inep_rendimento.py
python scripts/stage_inep_rendimento.py
# Default years = intersection of staged rendimento ∩ Censo (currently 2018–2025)
python -m src.etl.run_etl
```

### Train both level-specific models

```bash
python -m src.train --level both
```

### Streamlit apps

```bash
# Closing visual — model metrics, drivers, UF risk (recommended demo)
python -m streamlit run app/model_results/main.py

# School triage apps (score schools)
python -m streamlit run app/fundamental/main.py
python -m streamlit run app/medio/main.py
```

Export static PNGs for Slack / reports (also under `models/*/figures/`):

```bash
python scripts/export_model_figures.py
```

Figures written:
- `metrics_summary.png`
- `cv_leaderboard.png`
- `feature_importance.png`

### Tests

```bash
pytest -q
```

## EDA notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_eda_brazil_data_validation.ipynb` | Target definition, coverage, join validation |
| `notebooks/02_eda_fundamental.ipynb` | Fundamental EDA |
| `notebooks/03_eda_medio.ipynb` | Médio EDA |
| `notebooks/04_compare_fundamental_vs_medio.ipynb` | Why two models (driver comparison) |

Regenerate:

```bash
python scripts/build_eda_notebooks.py
```

## Project docs

- [`spec.md`](spec.md) — authoritative specification
- [`docs/scope_revision.md`](docs/scope_revision.md) — response to team-lead feedback
- [`docs/data_card.md`](docs/data_card.md)
- [`docs/model_card.md`](docs/model_card.md)
- [`latam_education_data/02_national/brasil/taxas_rendimento/README.md`](latam_education_data/02_national/brasil/taxas_rendimento/README.md)
