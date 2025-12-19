# Paper bundle (synthetic, but reviewer-proofed)

So what: this folder is the single entrypoint to the repo’s main results
(one table + the key figures).

Notes (quick definitions):
- `attention_cost` (λ): per-action cost charged for communication/attention.
- `λ*`: break-even λ for centralized matching vs search (central wins if λ > λ*).
- FieldSim v2 net welfare subtracts attention costs from total surplus.
- GPT is used for parsing/elicitation only; FieldSim v2 is simulation-only.

Why this matters:
- Search is the backup plan when the platform can’t confidently infer preferences.
- Better AI intake can lower the ROI boundary λ* needed for centralized matching
  to be worthwhile in hard-to-describe jobs.
- Delegated outreach agents can create a congestion externality at high adoption.

Framing (matches reviewer-proof evidence):
- Not “LLMs always improve markets” (effects vary by regime).
- Yes “LLMs can move you across a threshold where central mechanisms become viable,”
  which is captured by the regime map ROI boundary λ*
  and the hard-category interaction.

Key artifacts:
- `key_results.md` / `key_results.csv`: one table spanning all experiments
- `fig_hard_net_welfare.svg`: FieldSim v2 net welfare by arm (hard category)
- `fig_hard_reciprocity_curve.svg`: FieldSim v2 reciprocity proxy vs rank (hard)
- `fig_hard_net_welfare_vs_saturation.svg`: delegated outreach congestion (hard)
- `fig_regime_map_hard_net_welfare_diff.svg`: regime map sweep (hard)
- `fig_regime_map_hard_lambda_star.svg`: regime map ROI boundary (hard)

Source reports (full details + additional plots):
- FieldSim v2: `reports/field_v2_latest`
- Sensitivity: `reports/field_v2_sensitivity_latest`
- LLM ablations: `reports/ablations_latest`
- Congestion: `reports/latest_congestion`
- Minimal 2×2 + regime map: `reports/latest`
- Intake samples (home services): `reports/intakes_latest`

