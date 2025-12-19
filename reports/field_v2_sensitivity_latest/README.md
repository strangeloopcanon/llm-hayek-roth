# FieldSim v2 sensitivity sweep

So what: checks whether the core mechanism interaction (ai×central×hard) is robust
across seeds and a small grid of key knobs.

Runs: 72
- share(triple_welfare>0): 0.61
- share(p<0.05): 0.14
- triple_welfare quantiles: p10=-6.29, p50=2.73, p90=14.13

Artifacts:
- `runs.csv`: run-level DiD + regression triples
- `summary.json`: aggregate quantiles and grid
- `fig_triple_welfare_vs_rec_k.svg`: quick robustness plot

Run:
- `uv run python -m econ_llm_preferences_experiment.field_sim_v2_sensitivity`

