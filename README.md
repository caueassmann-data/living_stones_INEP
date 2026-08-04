# Brazil School Dropout Risk (Fundamental & Medio)

Living Stone Foundation — Applied Data Lab

School-level early-warning **prototype** for Brazilian basic education
dropout rates:

- **Ensino Fundamental** (grades 1-9, ages ~6-14) — model + Streamlit app
- **Ensino Medio** (grades 10-12, Brazil's upper-secondary stage) — model + Streamlit app

Population and labels are aligned: **Brazil**, **basic education**, **official
INEP dropout rate ("taxa de abandono")**, grain **school x year**. Every mart
row also carries the school's own dropout-rate history (see "What changed in
v0.3.0" below) — the model is not limited to a single-year snapshot.

## Important limitation

This prototype predicts **school-level** dropout rates using official INEP
School Attainment Rates joined to School Census features, plus each school's
own dropout history. It does **not** score individual students and is **not**
a multi-country LATAM model.

## What changed in v0.3.0 (read this first)

A validity review of the previous version (documented in full in
[`docs/validity_and_english_revision.md`](docs/validity_and_english_revision.md))
found that both trained models **did not beat a trivial "always predict the
training-set mean" baseline** on the metric used to select them. The root
causes and fixes:

1. **No baseline was ever checked.** Fixed: `dummy_mean` / `dummy_median` are
   now permanent entries in the model leaderboard, and a model that does not
   beat them is flagged loudly (`beats_baseline_cv`, `beats_baseline_test` in
   `models/{level}/metrics.json`, plus a warning banner in every app).
2. **Train/test split was random by row, not by school.** With ~6-8 years of
   data per school, the same school could appear on both sides of a split.
   Fixed: `GroupShuffleSplit` / `GroupKFold` by `school_id`, plus a second,
   stricter out-of-time holdout (train on years <= 2023, test on 2024-2025).
3. **No use of each school's own dropout-rate history.** The single strongest
   available predictor for this kind of problem was simply absent. Fixed:
   lagged dropout/approval/failure rates and municipality/state spatial lags
   — see `docs/data_card.md` -> "History features."
4. **No ranking-quality metric**, even though the product is a triage tool.
   Fixed: precision@k / recall@k / lift / average precision, reported next to
   the regression metrics.

The entire codebase, data schema, and documentation were also translated to
English (columns, identifiers, comments, docs) since this is an international
project — see `docs/validity_and_english_revision.md` for the full rename map.

## Quickstart (venv recommended)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Rebuild data marts (after downloading attainment rates)

```bash
python scripts/download_inep_rendimento.py
python scripts/stage_inep_rendimento.py
# Default years = intersection of staged attainment-rate years and Census years (currently 2018-2025)
python -m src.etl.run_etl
```

### Train both level-specific models

```bash
python -m src.train --level both
```

By default this trains on the full mart (earlier versions silently capped
training at 80,000 rows, discarding ~90% of the Fundamental mart). Pass
`--max-rows N` to sample for a faster local run.

### Streamlit apps

```bash
# Recommended: everything in one app with a sidebar to switch views
python -m streamlit run app/main.py

# Or run any view standalone:
python -m streamlit run app/model_results/main.py   # model metrics, drivers, state (UF) risk
python -m streamlit run app/fundamental/main.py      # school triage — Fundamental
python -m streamlit run app/medio/main.py            # school triage — Medio
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

Most tests use small synthetic fixtures and run without the multi-GB marts
built; a few (in `tests/test_brazil_marts.py`) skip automatically if the marts
have not been built yet.

## EDA notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_eda_brazil_data_validation.ipynb` | Target definition, coverage, join validation |
| `notebooks/02_eda_fundamental.ipynb` | Fundamental EDA |
| `notebooks/03_eda_medio.ipynb` | Medio EDA |
| `notebooks/04_compare_fundamental_vs_medio.ipynb` | Why two models (driver comparison) |

Regenerate:

```bash
python scripts/build_eda_notebooks.py
```

## Project docs

- [`spec.md`](spec.md) — authoritative specification
- [`docs/validity_and_english_revision.md`](docs/validity_and_english_revision.md) — the v0.3.0 validity review and English rewrite (read this first)
- [`docs/scope_revision.md`](docs/scope_revision.md) — the earlier v2.0.0 scope pivot to Brazil-only
- [`docs/data_card.md`](docs/data_card.md)
- [`docs/model_card.md`](docs/model_card.md)
- [`latam_education_data/02_national/brasil/taxas_rendimento/README.md`](latam_education_data/02_national/brasil/taxas_rendimento/README.md)
