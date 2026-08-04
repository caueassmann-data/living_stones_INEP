# Validity review and English rewrite — v3.0.0

**Project:** Brazil School Dropout Risk (Fundamental & Medio)
**Publisher:** Living Stone Foundation — Applied Data Lab
**Date:** 2026-08-04
**Supersedes for modeling purposes:** `docs/scope_revision.md` (2026-07-29),
which fixed the project's *scope* (Brazil-only, school-level, official
target). This document fixes the project's *validity* (whether the trained
models actually work) and completes a full English rewrite of the codebase.

---

## 1. Why this review happened

A general evaluation of the v2.0.0 prototype was requested. The headline
finding: **both trained models lost to a trivial baseline on the metric used
to select them.**

Because the target is a percentage that cannot go below 0, the mean absolute
error (MAE) of a model that always predicts the training-set mean is exactly
equal to that mean. Comparing v2.0.0's reported metrics against that trivial
number:

| | Model MAE (v2.0.0) | "Always predict the mean" MAE | Model R2 |
|---|---|---|---|
| Fundamental | 1.504 | **1.142** | 0.064 |
| Medio | 3.343 | **3.314** | 0.122 |

On Fundamental, the trained model was **32% worse** than doing nothing. On
Medio, it was statistically indistinguishable from doing nothing. The
positive R2 values show *some* signal exists (the model beats the mean on
squared error), but the leaderboard that selected "XGBoost" and "Random
Forest" as winners never once compared them to a baseline, on the exact
metric — MAE — it used to declare a winner.

## 2. Root causes

### 2.1 No baseline was ever checked
`src/train.py` compared three real candidates (Ridge, RandomForest, XGBoost)
against *each other* and declared whichever had the lowest cross-validated
MAE the winner. Nothing in that leaderboard could ever reveal that all three
were losing to a model that ignores every input feature.

**Fix:** `dummy_mean` and `dummy_median` (`sklearn.dummy.DummyRegressor`) are
now permanent, non-selectable entries in the leaderboard. `beats_baseline_cv`
and `beats_baseline_test` are computed and written to every
`models/{level}/metrics.json`, and every app surfaces a visible warning
banner when a model fails either check (see `app/model_results/main.py`,
`app/ui_common.py`).

### 2.2 The train/test split let schools leak across both sides
`train_test_split` split individual **rows** at random. With roughly 6-8
years of data per school (2018-2025), the same school's other years could
appear in both the training and test sets. A model can partially "memorize"
a school it is nominally being tested on, inflating every reported metric.
The same problem existed in the cross-validation `KFold`.

**Fix:** `GroupShuffleSplit` and `GroupKFold`, grouped by `school_id`
(`src/train.py`). A school's rows are now entirely on one side of any split.
A second, independent, and stricter test was added: an **out-of-time
holdout** — train on years <= 2023, evaluate on 2024-2025 only. This is the
actual scenario the product promises ("predict a dropout rate for a year you
haven't seen yet"), and unlike the group split, no amount of clever modeling
of *past* schools can substitute for genuinely generalizing to the *future*.

### 2.3 The single strongest predictor was completely absent
Every feature in v2.0.0 came from the same-year School Census snapshot. The
model was never given a school's own dropout-rate history — the single
strongest predictor available for a slow-moving, autocorrelated
administrative statistic like this one.

This is not the same issue as 2.2. It is worth stating precisely why
same-year Census features are *not* leakage, because the natural instinct is
to assume they are: INEP defines the dropout-rate outcome window as starting
*after* the Census reference date (see `docs/data_card.md`). The Census
snapshot genuinely precedes the outcome it predicts. The actual gap was
simpler and more basic: nobody had added the school's *own prior-year*
attainment numbers as inputs at all.

**Fix:** `src/etl/build_school_risk_marts.py::_add_history_features` now
computes, from strictly earlier years only:

- `dropout_rate_lag1` / `dropout_rate_lag2` / `dropout_rate_3yr_avg` / `dropout_rate_trend`
- `approval_rate_lag1` / `failure_rate_lag1` (recovered from source columns
  the staging script previously discarded entirely — see section 4)
- `municipal_dropout_rate_lag1` / `state_dropout_rate_lag1` (spatial lags:
  the local neighborhood's recent average, which captures shocks a single
  school's own history cannot)

On the rebuilt Fundamental mart, `dropout_rate_lag1` alone correlates 0.31
with the target — by far the strongest single feature relationship in the
dataset (the next-best prior feature was near zero). Coverage: roughly 86% of
rows in both marts have at least one prior year of history; the remainder are
schools' first appearance in the 2018-2025 window (flagged via `has_history`,
not silently imputed as "zero dropout").

### 2.4 No ranking-quality metric, despite the product being a ranking tool
The product's actual job is triage: "which schools should we look at first."
A model can have a mediocre point-error and still provide a genuinely useful
ranking — MAE/RMSE do not measure that at all.

**Fix:** `src/evaluate.py::evaluate_ranking` reports precision@k, recall@k,
lift, and average precision against the mart's `high_risk` label. `lift`
above 1.0 means the ranking beats selecting schools at random.

### 2.5 No subgroup / fairness breakdown
`docs/model_card.md` (v2.0.0) already warned against punitive use and noted
public/private and rural/urban schools "differ sharply," but nothing checked
whether the model's *error* differed sharply across those groups too — an
unenforced warning is not a safeguard.

**Fix:** `src/evaluate.py::evaluate_by_subgroup` reports MAE, RMSE, and
**signed mean error** (not just absolute error) split by rural/urban,
public/private, and school-size bucket, written to `subgroup_metrics` in
`models/{level}/metrics.json`. Signed error matters because a model can have
an unremarkable MAE for a group while still being consistently biased in one
direction for it — that is the actual fairness failure mode a punitive user
would exploit, and MAE alone cannot detect it.

## 3. Results after the fix

Retrained on the full mart (no sampling cap), with a `RandomForestRegressor`
winning both levels by grouped cross-validated MAE. Both models now clearly
beat the trivial baseline on **every** evaluation view — the grouped-school
holdout, the out-of-time holdout, and cross-validation:

| | Fundamental | Medio |
|---|---|---|
| Selected model | `random_forest` | `random_forest` |
| Test MAE, grouped-by-school holdout | **1.139** | **2.510** |
| Baseline MAE, same holdout | 1.652 | 3.933 |
| Test R2, grouped-by-school holdout | 0.293 (was 0.064) | 0.382 (was 0.122) |
| Test MAE, 2024-2025 out-of-time holdout | **0.640** | **2.255** |
| Baseline MAE, out-of-time holdout | 1.384 | 3.365 |
| Top-decile precision (of the schools flagged highest-risk, share that truly are) | 76% | 83% |
| Top-decile lift over random selection | 2.18x | 2.44x |
| Trained pipeline size | 29 MB | 12 MB |

The out-of-time holdout — training only on 2018-2023 and scoring 2024-2025,
years the model never saw in any form — is the strictest and most realistic
test for an early-warning tool, and both models pass it convincingly. The
single largest driver in both levels' feature-importance rankings is
`dropout_rate_3yr_avg` (24% of importance for Fundamental, 33% for Medio),
confirming the root-cause analysis in section 2.3: a school's own dropout
history was the missing predictor. `overage_enrollment_share` (also new in
this revision) is the second-largest driver for both levels.

RandomForest's default hyperparameters produced a strong but impractically
large model (183 MB for Fundamental) for a ~4-5% CV-MAE edge over XGBoost —
too large to comfortably ship in a Streamlit app reloaded on every
interaction. Constraining `max_depth=18` and raising `min_samples_leaf` to 25
(`src/train.py::_candidates`) cut both models to a fraction of that size
(29 MB / 12 MB) for a difference of 0.01-0.02 in R2 — a reasonable trade.

**Equity finding** (section 2.5's fix in action): the per-subgroup breakdown
shows rural schools have roughly **double** the MAE of urban schools in both
levels (Fundamental: 1.71 vs 0.88; Medio: 4.24 vs 2.32), and public schools
have meaningfully higher MAE than private ones (most pronounced in Medio:
3.23 vs 0.62 — private Medio schools have consistently low, easy-to-predict
dropout, while public Medio schools are far more variable). The *signed*
mean error is close to zero for every group in both levels, meaning this is
higher **noise/variance** for rural and public schools, not a one-directional
bias — but it still means triage flags for rural/public schools should carry
a wider uncertainty band than the pooled MAE alone suggests. This is exactly
the kind of gap `docs/model_card.md` warns readers to check for before using
predictions punitively, and it could not have been seen before this revision.

Full numbers, including the 20%/33% triage tiers and per-subgroup fairness
breakdown: `models/{level}/metrics.json`, or run
`python -m streamlit run app/model_results/main.py` and open a level's
"1. Metrics," "5. Future-year holdout," and "6. Equity" tabs for the live,
rendered version of this comparison.

## 4. Full English rewrite

This was requested independently of the validity fix (the project is
international and every reviewer should not need Portuguese), but touched
nearly every file, so it is documented here rather than split across many
smaller notes.

### What changed
- **Data columns**: every mart/staging column renamed from Portuguese
  abbreviations (or mixed INEP source codes) to descriptive English, e.g.
  `tp_dependencia` -> `admin_dependency_type`, `uf` -> `state_code`,
  `in_agua`/`in_energia`/`in_esgoto` -> `has_water`/`has_electricity`/`has_sewage`,
  `qt_mat_bas` -> `enrollment_basic_ed`, `taxa_abandono_fund`/`_med` ->
  `dropout_rate_fundamental`/`dropout_rate_medio`.
- **Category label values**, not just column names: INEP's raw Portuguese
  labels are translated at staging time (`Urbana`/`Rural` -> `Urban`/`Rural`,
  `Estadual`/`Municipal`/`Privada` -> `State`/`Municipal`/`Private`) in
  `scripts/stage_inep_rendimento.py`.
- **Python identifiers**: e.g. `_brasil_dados_dir` -> `_brazil_school_census_dir`,
  `summary_by_uf` -> `summary_by_state`, `discover_brasil_years` ->
  `discover_brazil_years`, `mean_abandono` -> `mean_dropout_rate`.
- **Prediction/output columns**: `pred_taxa_abandono` -> `pred_dropout_rate`.
- **Comments and docstrings**: rewritten throughout `src/`, `app/`, `scripts/`
  to explain *why*, in English, for a reader with no prior context on this
  project (see section 5).

### What was deliberately kept as-is, and why
- **"Fundamental" and "Medio"** are kept, not translated, as the names of
  Brazil's own two basic-education stages. They do not map cleanly onto a
  single English schooling term (Fundamental spans what US/UK systems would
  split across elementary and middle school; Medio is closest to "upper
  secondary" but not identical). Every place they appear is glossed with the
  grade range and approximate ages (see `src/utils.py::LEVEL_DISPLAY_NAME`).
- **Brazilian place names** (state and municipality names, e.g. "Sao Paulo")
  are kept in their official spelling, matching standard English-language
  practice of not translating proper place names.
- **On-disk raw/staging folder names** (`latam_education_data/02_national/brasil/...`,
  `taxas_rendimento/`) keep INEP's original directory naming. This is a
  pragmatic call: renaming them would force re-downloading ~3.3GB of already-
  fetched Census extracts for a folder-name-only change. Every *Python
  identifier* referencing these paths uses the English spelling ("Brazil"),
  and the discrepancy is called out in code comments at each point it occurs
  (`src/etl/build_school_risk_marts.py`, `scripts/stage_inep_rendimento.py`)
  so it reads as a documented, deliberate choice rather than an oversight.
- **INEP's raw source column codes** (`3_CAT_FUN`, `CO_ENTIDADE`, etc.) are
  kept as literal references *when documenting where a value came from* —
  translating an external agency's own identifier would make it harder, not
  easier, to cross-reference this repo against INEP's published documentation.

## 5. Dependency and engineering cleanup done alongside this revision

- Initialized a git repository (there was none) with a `.gitignore` that
  excludes the multi-GB raw/staged INEP extracts and the trained pipeline
  binaries — both are reproducible via `scripts/` and `src/train.py` and do
  not belong in version control.
- Removed `shap` and `pyyaml` from `requirements.txt`: `shap` was listed but
  never imported anywhere in the codebase despite the spec mentioning SHAP as
  planned XAI; native tree/coefficient/permutation importances
  (`src/evaluate.py::export_global_importance`) already serve the
  explainability requirement. Real SHAP support (e.g. `TreeExplainer` on the
  winning model, with a summary plot in the app) is a reasonable follow-up
  but was not implemented in this pass — recorded here rather than left as a
  silent scope cut.
- Pinned all remaining dependencies to exact, verified-working versions.
- Added `pyproject.toml` with a `ruff` configuration.
- Added real unit tests (`tests/test_history_features.py`,
  `tests/test_risk_bands.py`, `tests/test_evaluate.py`,
  `tests/test_preprocessing.py`, `tests/test_data_loader.py`,
  `tests/test_inference.py`) that use small synthetic fixtures and therefore
  run without the marts being built — the previous test suite silently
  skipped everything until 3+ GB of data existed locally.
  - Writing these tests directly caught two real bugs in the new risk-band
    logic before it shipped: (1) a large tied block at the dropout-rate floor
    (commonly 0%) could average-rank above the "moderate" cutoff once it
    exceeded roughly two-thirds of the population — the real Fundamental mart
    sits at 64.6% zero-rate, uncomfortably close to that line; and (2) the
    `high_risk` flag and the `risk_band == "high"` label used two different
    thresholds (`max(q66, 5.0)` vs. a flat `5.0`) that silently disagreed
    whenever the 66th-percentile rate itself exceeded 5%. Both are fixed in
    `src/etl/build_school_risk_marts.py::assign_risk_bands` and covered by
    `tests/test_risk_bands.py`.
- Cached model/pipeline loading (`functools.lru_cache` in
  `src/inference.py`, `st.cache_resource`/`st.cache_data` in the Streamlit
  apps) — the 51MB Medio pipeline was previously reloaded from disk on every
  UI interaction, including twice per single-school prediction.

## 6. What was not done in this pass (recorded, not silently dropped)

- **A dedicated hurdle/classification model** (classify dropout > 0, then
  regress magnitude) was considered as an alternative to the ranking metrics
  in section 2.4, but was not built: the ranking metrics computed directly
  from the regression's output already serve the "which schools first"
  triage use case, and a second full model track would roughly double the
  modeling effort in this pass for a use case already covered.
- **Real SHAP explainability** — see section 5.
- **Night-shift / EJA-detail enrollment and grade-level attainment
  breakdowns** (e.g. `dropout_rate_medio_grade1..4`, already staged but not
  yet used as model features) are available in the staged data for a future
  iteration but were not added as predictors in this pass.
