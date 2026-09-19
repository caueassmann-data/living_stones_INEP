# Changelog

## 3.1.0 — 2026-09-18

Applies the Foundation's Milestone 1 decision (Notion tasks 6-7): the product
rule is now capacity-constrained **Top-N prioritization within one education
network**, not a threshold on the dropout rate. No model was retrained;
`MODEL_VERSION` stays `0.3.0`.

- **New decision rule** (`src/prioritize.py`, `PRIORITIZATION_VERSION`): filter
  to a scope, rank by `pred_dropout_rate`, take the top N (default 50, set by
  the field team). Scope is a hierarchy — state network, municipality, municipal
  network, or a set of municipalities pooled as one list.
- **`high_risk` is unchanged and no longer user-facing.** `assign_risk_bands`
  in the ETL is untouched; the label stays as a historical/evaluation label
  behind the ranking metrics, and `tests/test_risk_bands.py` still passes
  unmodified.
- **Scope granularity is now explicit, because it has to be.** The median
  municipal network holds 3 schools (Fundamental) and 1 (Medio), so "top 50
  within a municipal network" returns the whole network for ~98% of
  municipalities — 70,491 Fundamental schools nationwide. The default scope is
  therefore the state network (median 355 schools), and the app states plainly
  when N exceeds the selected scope (`coverage_message`).
- **New warning for networks with nothing to prioritize** (`low_signal_message`):
  Salvador's municipal Fundamental network averages 0.05% observed dropout and
  its top 50 has *lower* observed dropout than the network average — the
  ranking is sorting noise, which a fixed-length list cannot reveal on its own.
- **Ranking-quality fix:** the app scored only `filtered.head(20000)` before
  ranking, so a large selection's "top 50" was the top 50 of whatever came
  first in mart order. Now the whole year is scored once, cached on
  `(level, year)` rather than on a DataFrame.
- **Deterministic tie-break.** About 1.9% of predictions are exact duplicates;
  the list is now cut with a stable sort and `cumcount()`, so repeated queries
  return the same schools and a request for 50 never returns 53.
- **New metric:** `evaluate_ranking_within_group` / `test_ranking_within_network`
  in `models/{level}/metrics.json` — precision@N inside each network, which is
  how the tool is used. Held-out top-50 inside a state network: lift 1.45
  (Fundamental) and 1.31 (Medio), against 2.18 for one national ranking.
  Networks no larger than N are excluded, since their precision@N is their base
  rate by construction.
- New `scripts/analyze_network_prioritization.py` (network sizes, Top-N quality,
  rollout list sizes, year-over-year churn) and
  `scripts/backfill_within_network_metrics.py` (adds the new metric to existing
  artifacts without an 18-minute retrain). `scripts/analyze_high_risk_criteria.py`
  and its frozen output are deliberately left alone.
- App: the prioritized list is now the first tab; school-name search moved out
  of the filters, since searching before ranking silently redefined "top 50".

### Metrics for non-technical readers

- The "Model insights" tabs and the Model Results Lab showed raw JSON, raw
  column names (`cv_mae_mean`, `priority_rank`), matplotlib charts in white
  boxes on the dark theme, and MAE/RMSE/R2 jargon. New `app/metric_views.py`
  replaces them with cards (model vs baseline), Altair charts that follow the
  light/dark theme, one-sentence readings, and formatted tables with plain
  column names. The JSON is still available in a collapsed "Technical details"
  expander. Covers all six Lab tabs, the "Compare both" view, and the
  prioritized list, whose on-screen columns are now readable (the downloaded
  CSV keeps the full machine-readable columns).
- Ranking quality is now compared against picking schools **at random** (the
  label's base rate), not against the baseline model. The baseline predicts a
  constant, so its "ranking" only reflected row order in the file (lift 1.42),
  which read as a real effect. The stored baseline ranking numbers are
  unchanged in `metrics.json`; they are simply no longer presented.

### Deployability

The app could not be hosted at all: `.gitignore` excluded both the mart
parquets and `models/*/pipeline.joblib`, so a fresh clone failed at
`load_artifacts()`, and the full marts needed ~1.25 GB of RAM.

- New `scripts/build_deploy_artifacts.py` writes a two-year slice of each mart
  to `latam_education_data/marts_app/`: 10.6 MB on disk and ~154 MB in memory,
  against 49.7 MB / ~620 MB. App RSS with both levels loaded and scored drops
  from 1.25 GB to **680 MB**. It also refuses to build when a retrain selected
  a model whose library is not in the deployment requirements.
- `src/utils.py::resolve_app_mart_path` — the app reads the full mart locally
  and the slim one when deployed; training keeps using `mart_path` so it can
  never fit on the two-year window by accident.
- The model binaries and slim marts are now tracked (~51 MB total, no LFS
  needed). The full marts stay ignored.
- `requirements.txt` is now the app runtime only; ETL, training, test, and
  notebook dependencies moved to `requirements-dev.txt`, so a hosted build
  does not install DuckDB, Jupyter, or XGBoost.

## 3.0.0 — 2026-08-04

- **Validity fix:** both models previously did not beat a trivial
  "predict-the-mean" baseline on the metric used to select them. Added a
  permanent baseline check (`beats_baseline_cv`, `beats_baseline_test` in
  `models/{level}/metrics.json`), surfaced in every app.
- **Split fix:** replaced the random row-level train/test split with a
  school-grouped split (`GroupShuffleSplit`/`GroupKFold` by `school_id`) plus
  a strict out-of-time holdout (train on years <= 2023, test on 2024-2025).
- **New history features:** each school's own lagged dropout/approval/failure
  rates, 3-year trailing average, trend, and municipality/state spatial-lag
  dropout rates — previously absent entirely.
- **New ranking metrics:** precision@k, recall@k, lift, and average precision
  against the mart's `high_risk` label, since the product is a triage tool.
- **New fairness metrics:** hold-out MAE/RMSE/signed-mean-error broken out by
  rural/urban, public/private, and school size (`subgroup_metrics` in
  `models/{level}/metrics.json`, "Equity" tab in the model-results app).
  Rural and public schools show roughly double the MAE of urban/private ones
  in both levels — a gap the previous version had no way to detect.
- **Feature expansion:** state (UF) and administrative-network codes are now
  one-hot encoded rather than scaled as raw integers; added over-age enrollment
  share, EJA enrollment share, average class size, internet-for-students, and
  a `is_covid_year` flag for 2020-2021.
- Removed the undocumented 80,000-row training cap (previously discarded
  ~90% of the Fundamental mart by default).
- **Full English rewrite:** every data column, Python identifier, comment,
  and doc translated from Portuguese abbreviations/mixed-language to English
  (e.g. `tp_dependencia` -> `admin_dependency_type`, `uf` -> `state_code`,
  `taxa_abandono` -> `dropout_rate_*`). Category label values (e.g.
  "Urbana"/"Rural", "Estadual"/"Municipal") are also translated at staging
  time. See `docs/validity_and_english_revision.md` for the full rename map
  and reasoning.
- Product: unified Streamlit entrypoint (`app/main.py`), caching, labeled
  (non-raw-code) single-school inputs, uncertainty display, school search with
  top-N export, and a predicted-vs-actual state (UF) chart (previously only
  predicted was plotted, even though actual was already computed).
- Bounded RandomForest tree size (`max_depth=18`, `min_samples_leaf=25`):
  the previous default-hyperparameter forest scored ~4-5% lower CV-MAE than
  XGBoost but produced a 183MB pipeline; the bounded version keeps essentially
  the same accuracy (R2 within 0.01-0.02) at 29MB (Fundamental) / 12MB (Medio).
- Engineering: git repository initialized; pinned `requirements.txt`; removed
  the unused `shap` and `pyyaml` dependencies; added `pyproject.toml` +
  `ruff` config; added real unit tests with synthetic fixtures that do not
  require the multi-GB marts to be built.

## 2.0.0 — 2026-07-29

- **Scope pivot:** Brazil-only school-level early warning for Ensino Fundamental and Ensino Médio.
- Integrated official INEP Taxas de Rendimento (`taxa de abandono`) with Censo Escolar features.
- Dual marts: `school_risk_br_fundamental`, `school_risk_br_medio`.
- Dual models + dual Streamlit apps; retired Kaggle HE prototype and multi-country LATAM product core.
- Docs rewritten (`spec.md` v2, data/model cards, `docs/scope_revision.md`).

## 0.1.0 — 2026-07-26

- Initial dual-track prototype (LATAM warehouse + Kaggle HE student model). Superseded by 2.0.0.
