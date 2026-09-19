# Model Card — Brazil School Dropout Rate (Fundamental & Medio)

## Intended use

Help education analysts / network managers **triage schools** with elevated
predicted dropout rates and inspect the school features contributing to that
prediction, so limited follow-up resources can be pointed at the schools that
need them most.

### Deployed decision rule

The rule the product applies is **capacity-constrained Top-N within one
education network** (`src/prioritize.py`), decided by the Foundation in
Milestone 1: pick a network, rank its schools by predicted dropout rate, take
the top N, where N is how many schools the field team can follow up on
(default 50). It is deliberately **not** a threshold on the dropout rate: a
fixed cut such as ">= 5%" is strict in Ensino Fundamental (mean 0.5%) and
common in Ensino Medio, so it cannot be one product rule.

The mart's `high_risk` label is **not** the decision rule. It is a historical /
evaluation label only (`assign_risk_bands` in
`src/etl/build_school_risk_marts.py`), used for the ranking metrics below and
not shown to end users. In 2025 it flags 26% of Fundamental schools, which is
not an actionable list — see `docs/milestone1_high_risk_criteria.md`.

## Out of scope

- Individual student risk scores (no student-level labeled data exists in the
  open INEP extracts used here — see `docs/data_card.md`).
- Causal claims ("X causes dropout"). Every reported driver is a model
  *importance*, not an estimated causal effect.
- Cross-country LATAM rankings from these models (Brazil-only, by design —
  see `docs/scope_revision.md`).
- Punitive use (ranking schools to withhold resources or assign blame). See
  "Equity notes" below.
- League tables. A prioritized list is an ordering of where to look first
  inside one network; it is not a comparison between networks, and positions
  are not comparable across scopes or across N.

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
  `metrics.json`) — see `src/evaluate.py::evaluate_ranking`. One national
  ranking: "if we act on the top decile of predicted risk nationwide, how many
  of the truly highest-risk schools do we catch."
- Ranking **within a network** (`test_ranking_within_network` in
  `metrics.json`) — see `src/evaluate.py::evaluate_ranking_within_group`. This
  is the metric that matches the deployed rule, and it is the one to quote
  operationally. On held-out schools, top 50 inside a state network:

  | Level | Networks | Precision@50 | Base rate | Lift | Observed dropout in list |
  |---|---:|---:|---:|---:|---:|
  | Fundamental | 523 | 0.50 | 0.35 | 1.45 | 2.50% |
  | Medio | 223 | 0.44 | 0.33 | 1.31 | 4.99% |

  **Read these with their qualifier.** They cover only networks with more than
  N schools — 94% of Brazilian municipalities have fewer than 50 schools, and
  in those the "top 50" is the whole network, so no lift figure describes them.
  Within-network lift is also structurally lower than the national figure
  (2.18 at the top decile) because schools inside one network are more alike;
  that is expected, not a regression.

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
- **Top-N interacts with network size.** In a small network a school can enter
  the list simply because few schools compete with it, not because its risk is
  high; in a near-zero-dropout network the ordering is largely noise (Salvador's
  municipal Fundamental network averages 0.05% observed dropout, and its top 50
  has *lower* observed dropout than the network average). The app warns in both
  cases — see `coverage_message` and `low_signal_message` in
  `src/prioritize.py` — but a list always returns N rows, so the warnings are
  the only thing standing between a small network and a misread.
- Always display the limitation statement (`src/utils.py::LIMITATION_STATEMENT`)
  in app footers — every app in this repo already does this.

## Version

See `MODEL_VERSION` in `src/utils.py` and per-level
`models/*/model_card_meta.json`. Full history: `CHANGELOG.md`.
