# Field-style simulation (cluster randomization)

So what: this run simulates the *actual experimental design* (cluster-level assignment),
then estimates the AI × centralized-mechanism interaction using customer-level outcomes
with cluster-robust SEs.

Quick read (rec_k=5, attention_cost=0.01):
- match_rate DiD (easy): +0.032
- match_rate DiD (hard): +0.041
- triple interaction (ai×central×hard): +0.009
  (SE=0.022, p=0.676)

Design:
- Unit: (city×category×week) cell; assigned to one of 4 arms (2×2).
- Outcomes measured at the customer/job level; inference clustered by cell.
- Describability proxy: P(both accept | recommended) from a thin accept/reject step.

Artifacts:
- `cells.csv`: one row per cell
- `customers.csv`: one row per customer/job
- `reciprocity_by_rank.csv`: acceptance proxy by rec rank
- `arm_summary.md`: arm means (by category) with Monte Carlo SE
- `reg_matched.md`: clustered regression on match indicator
- `reg_matched_heterogeneity.md`: adds a hard-category triple interaction
- `reg_total_value.md`: clustered regression on realized match value (0 if unmatched)
- `reg_total_value_heterogeneity.md`: adds a hard-category triple interaction
- `fig_*`: bar charts + reciprocity curves

Run:
- `uv run python -m econ_llm_preferences_experiment.publishable_field_sim`
