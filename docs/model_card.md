# Model Card — Brazil School Abandonment (Fundamental & Médio)

## Intended use

Help education analysts / network managers **triage schools** with elevated predicted abandonment rates and inspect contributing school features.

## Out of scope

- Individual student risk scores
- Causal claims (“X causes dropout”)
- Cross-country LATAM rankings from these models

## Training data

Level-specific marts built from INEP Censo features + official Taxas de Rendimento abandonment rates (**2018–2025**).

## Candidates evaluated

Regression candidates selected by 5-fold CV MAE:

- Ridge
- RandomForestRegressor
- XGBRegressor

Winner is written to `models/{level}/metrics.json` (`selected_model`).

## Metrics

Hold-out **MAE / RMSE / R²** on `target_dropout_rate` (%).

## Explainability

Global feature importance exported to `models/{level}/figures/global_importance_top.csv` (tree importances, |coefficients|, or permutation importance).

## Ethical notes

- Predictions can reinforce existing inequalities if used punitively.
- Prefer supportive triage (resource targeting) over ranking schools for punishment.
- Always display the limitation statement in app footers.

## Version

See `MODEL_VERSION` in `src/utils.py` and per-level `models/*/model_card_meta.json`.
