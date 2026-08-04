# Model Card — Brazil School Dropout Rate (Fundamental & Medio)

## Intended use

Help education analysts / network managers **triage schools** with elevated
predicted dropout rates and inspect the school features contributing to that
prediction, so limited follow-up resources can be pointed at the schools that
need them most.

## Out of scope

- Individual student risk scores (no student-level labeled data exists in the
  open INEP extracts used here — see `docs/data_card.md`).
- Causal claims ("X causes dropout"). Every reported driver is a model
  *importance*, not an estimated causal effect.
- Cross-country LATAM rankings from these models (Brazil-only, by design —
  see `docs/scope_revision.md`).
- Punitive use (ranking schools to withhold resources or assign blame). See
  "Equity notes" below.

## Training data

Level-specific marts built from INEP Census features plus each school's own
lagged official attainment history (2018-2025). See `docs/data_card.md` for
the full feature list and `docs/validity_and_english_revision.md` for why the
history features were added.

## Validation methodology

Two held-out evaluations are reported for every trained model, because they
test different things:

1. **Held-out schools** (`test_metrics` in `models/{level}/metrics.json`) —
   schools are split into train/test by `school_id` (`GroupShuffleSplit`), so
   no school's rows appear on both sides. Candidate models are compared by
   5-fold cross-validated MAE, also grouped by school (`GroupKFold`).
2. **Held-out future years** (`temporal_holdout` in the same file) — trained
   on years <= 2023, evaluated on 2024-2025 only. This is the scenario the
   product actually promises (predicting a year the model has never seen)
   and is the stricter of the two tests.

Both evaluations are reported **against a trivial baseline** (a model that
ignores every feature and always predicts the training-set mean). A model
that cannot beat this baseline is flagged (`beats_baseline_cv`,
`beats_baseline_test` in `metrics.json`) rather than silently reported as a
"winner" — see `docs/validity_and_english_revision.md` for why this check
was added and what it found in the prior version of this pipeline.

## Candidates evaluated

Regression candidates, selected by grouped cross-validated MAE:

- `dummy_mean` / `dummy_median` (trivial baselines, never eligible to be
  selected as the "winner" — they exist only as a floor to beat)
- Ridge
- RandomForestRegressor
- XGBRegressor

The winner is written to `models/{level}/metrics.json` (`selected_model`).

## Metrics

- Regression: **MAE / RMSE / R2** on `target_dropout_rate` (percentage
  points), on both held-out views above.
- Ranking (triage quality): precision@k, recall@k, lift, and average
  precision against the mart's `high_risk` label (`test_ranking_metrics` in
  `metrics.json`) — see `src/evaluate.py::evaluate_ranking`. This answers the
  question the product actually needs: "if we act on the top decile of
  predicted risk, how many of the truly highest-risk schools do we catch."

## Explainability

Global feature importance exported to
`models/{level}/figures/global_importance_top.csv` (tree importances,
absolute linear coefficients, or permutation importance, depending on the
winning model type — see `src/evaluate.py::export_global_importance`).
One-hot-encoded categorical features (e.g. `state_code`) are reported per
category (e.g. `state_code_SP`).

## Equity notes

- Predictions can reinforce existing inequalities if used punitively (e.g. to
  justify withdrawing resources from a school flagged as high-risk, rather
  than directing support to it).
- Rural, public, and small schools are structurally different from urban,
  private, and large ones; a model with good average error can still be
  systematically worse for one of these groups. Subgroup error breakdowns are
  reported where available — see `docs/validity_and_english_revision.md`.
- Prefer supportive triage (resource targeting) over ranking schools for
  punishment or public comparison.
- Always display the limitation statement (`src/utils.py::LIMITATION_STATEMENT`)
  in app footers — every app in this repo already does this.

## Version

See `MODEL_VERSION` in `src/utils.py` and per-level
`models/*/model_card_meta.json`. Full history: `CHANGELOG.md`.
