# WC Match Predictions

A World Cup 2026 team-only match prediction app using a calibrated specialist ensemble.

## What it predicts

The app predicts:

```text
Team A win probability
Draw probability
Team B win probability
Expected goals
Specialist model breakdown
```

## Main models

```text
Elo strength model
Recent-form model
Poisson scoreline model
Context XGBoost model
Full XGBoost model
Draw-risk model
Optimized calibrated blend
```

---

# Match Edge AI V5: Calibrated Specialist Ensemble

World Cup 2026 team-only match predictor.

## What V5 improves

V5 improves V4 by replacing the final meta-scorer with a more controlled probability system:

```text
specialist model probabilities
        ↓
optimized weighted blend
        ↓
temperature calibration
        ↓
optional fixture-context adjustment
        ↓
final Team A win / Draw / Team B win
```

## Specialist models

| Scenario | Model |
|---|---|
| Overall team strength | Logistic Regression on Elo |
| Recent form | Random Forest |
| Expected score / scoreline | Two Poisson regressors |
| Match context | XGBoost |
| Full nonlinear 1X2 | XGBoost |
| Draw risk | Binary XGBoost classifier |
| Current FIFA rank/points | Heuristic specialist for explanation |
| Final scorer | Optimized weighted blend + temperature calibration |

## Optimized blend weights

See:

```text
reports/optimized_blend_weights.csv
```

Current learned weights:

               specialist  optimized_weight
    strength_elo_logistic      2.922559e-01
recent_form_random_forest      1.003767e-01
  poisson_scoreline_model      3.695481e-01
          context_xgboost      1.023599e-01
             full_xgboost      1.739661e-18
     draw_risk_classifier      1.354593e-01

## Evaluation

Date-aware split:

```text
Base specialist training: 2014-01-01 to 2021-12-31
Blend optimization: 2022-01-01 to 2023-12-31
Final test window: 2024-01-01 to 2026-06-10
```

Metrics:

```text
reports/model_metrics.csv
```

## Run one prediction

```bash
python src/match_edge_v5_predictor.py \
  --team-a "United States" \
  --team-b "Paraguay" \
  --venue-country "United States"
```

With optional context:

```bash
python src/match_edge_v5_predictor.py \
  --team-a "United States" \
  --team-b "Paraguay" \
  --venue-country "United States" \
  --team-a-rest-days 5 \
  --team-b-rest-days 4 \
  --team-a-travel-km 0 \
  --team-b-travel-km 1200 \
  --temperature-c 24 \
  --humidity-pct 55
```

## Run Streamlit app

```bash
streamlit run deployment/app_streamlit.py
```

## Batch prediction

```bash
python src/batch_predict.py \
  --input data/processed/example_matches.csv \
  --output reports/batch_predictions.csv
```

## Group-stage simulation scaffold

```bash
python simulation/simulate_group_stage.py
```

This creates:

```text
reports/group_stage_simulation_results.csv
```

## Important note

The FIFA rank/points specialist is included for current-team explanation, but it is not used as a fully trained historical feature because we do not yet have historical FIFA ranking snapshots for every past training match.

## Next upgrade

The next real accuracy jump is data-driven:

```text
historical FIFA ranking snapshots
official live World Cup match updates
actual team base-camp travel
weather API refresh near kickoff
starting lineups and injuries
```



## Notebook workflow

The project includes notebooks for review and retraining:

```text
notebooks/01_data_overview.ipynb
notebooks/02_eda.ipynb
notebooks/03_feature_engineering_review.ipynb
notebooks/04_train_v5_model.ipynb
notebooks/05_model_evaluation_and_predictions.ipynb
```

The training notebook rebuilds the model and saves:

```text
models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl
reports/model_metrics.csv
reports/optimized_blend_weights.csv
```

Recommended order:

```text
01_data_overview
02_eda
03_feature_engineering_review
04_train_v5_model
05_model_evaluation_and_predictions
```
