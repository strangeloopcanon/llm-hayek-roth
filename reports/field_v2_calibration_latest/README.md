# FieldSim v2 calibration (moment matching)

So what: picks a small set of parameters so the simulated marketplace has
plausible baseline moments (prices, cancels, match rates, congestion) before we
interpret welfare and mechanism comparisons.

Runs: 61 (including iter=0 baseline)
- best loss: 10.2297

Best-fit baseline moments (standard intake + search):
- easy: {'match_rate': 0.41333333333333344, 'cancel_rate': 0.0746390954724288, 'avg_price': 202.75618833990086, 'messages_per_job': 10.906666666666666, 'provider_inbox_per_day': 2.03015873015873, 'consumer_surplus_per_job': 10.057786666666665, 'provider_profit_per_job': 18.595704444444447}
- hard: {'match_rate': 0.28666666666666674, 'cancel_rate': 0.05972222222222223, 'avg_price': 388.5544816798943, 'messages_per_job': 11.83111111111111, 'provider_inbox_per_day': 2.03015873015873, 'consumer_surplus_per_job': 23.528015555555562, 'provider_profit_per_job': 29.08835333333333}

Artifacts:
- `runs.csv`: iteration-by-iteration params + moments + loss
- `fig_loss.svg`: loss curve
- `targets.json`: target moments + tolerances
- `best_params.json`: overrides to pass via `--params-json`
- `eval_settings.json`: evaluation settings used during calibration

Use best params in FieldSim v2:
- `uv run python -m econ_llm_preferences_experiment.field_sim_v2 --params-json reports/field_v2_calibration_latest/best_params.json`

