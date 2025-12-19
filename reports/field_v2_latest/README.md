# FieldSim v2 (dynamic + pricing + cancellations + spillovers + scaling)

So what: this is a more realistic simulation of the field experiment design, with
endogenous communication/congestion, pricing/negotiation, cancellations/rematching,
strategic budget shading, imperfect AI specs with user edits, and provider spillovers
across simultaneous cells sharing the same providers.

Quick read (attention_cost=0.25):
- match_rate DiD (easy): +0.017
- match_rate DiD (hard): +0.005
- triple interaction (ai×central×hard): -0.012
  (SE=0.022, p=0.586)
- net_welfare DiD (easy): -0.82
- net_welfare DiD (hard): +3.96
- triple interaction on net_welfare: +4.79
  (SE=2.88, p=0.096)

Key realism features:
- Pricing: provider quotes depend on cost, demand, and inferred budget
- Constraints: license/insurance + schedule overlap (hard feasibility)
- Renegotiation/cancels: under-stated complexity + weirdness revealed post-accept
- Misrep: some customers shade budgets (AI can slightly increase shading)
- Imperfect AI: requirement FP/FN + noisy budget/schedule; edits fix some errors
- Spillovers: easy and hard cells share providers within (city,week)

Artifacts:
- `cells.csv`, `jobs.csv`, `providers.csv`: synthetic datasets
- `arm_summary.md`: arm means by category
- `reg_*.md`: cluster-robust regressions (cluster=cell)
- `spillovers.md`: neighbor-cell spillover diagnostic
- `fig_*`: main plots

Run:
- `uv run python -m econ_llm_preferences_experiment.field_sim_v2`

