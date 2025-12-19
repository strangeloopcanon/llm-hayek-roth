# Delegated outreach agents × congestion (saturation design)

So what: delegated outreach agents create an adoption-intensity trade-off. They speed
up matching for treated users, but can impose congestion + communication costs that
lower net welfare at high saturation.

Quick read (attention_cost=0.01):
- easy: welfare-max saturation = 0.0 (net_welfare=0.269)
- hard: welfare-max saturation = 0.0 (net_welfare=0.249)

Context: home services marketplace. Customers request quotes; providers have
limited daily response capacity.
Providers also adapt by raising their accept threshold when their inbox is overloaded.

Treatment: a delegated outreach agent that (i) elicits richer preferences
(higher signal precision) and
(ii) sends more outbound quote requests per day (lower marginal outreach cost).

Design: within each (city×category×week) cell we randomize the fraction
of customers treated (saturation).
We report treated + control outcomes to surface congestion externalities.

Artifacts:
- `congestion_saturation_easy.csv` / `.md`
- `congestion_saturation_hard.csv` / `.md`
- `fig_*_vs_saturation.svg`
- `congestion_meta_*.json`

