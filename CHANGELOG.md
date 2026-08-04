# Changelog

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
