# Key results

So what: one table spanning the repo’s synthetic experiments (main effects + mechanisms).

Notes:
- `attention_cost` (λ): per-action cost (messages, inbox screening, accept decisions).
- `λ*`: threshold λ where central and search tie in net welfare; central wins if λ > λ*.
- FieldSim v2: `net_welfare = total_surplus - attention_cost * actions`.
- GPT usage: parsing/elicitation only; FieldSim v2 has no API calls.

| block | metric | outcome | estimate | se | p | n_obs | n_clusters | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FieldSim_v2 | ai×central×hard | matched | -0.0122 | 0.0225 | 0.5863 | 28800 | 480 | cluster-robust (cluster=cell) |
| FieldSim_v2 | ai×central×hard | canceled | -0.0019 | 0.0058 | 0.7389 | 28800 | 480 | cluster-robust (cluster=cell) |
| FieldSim_v2 | ai×central×hard | consumer_surplus | 4.9778 | 2.8 | 0.0754 | 28800 | 480 | cluster-robust (cluster=cell) |
| FieldSim_v2 | ai×central×hard | provider_profit | -0.1692 | 1.5791 | 0.9147 | 28800 | 480 | cluster-robust (cluster=cell) |
| FieldSim_v2 | ai×central×hard | net_welfare | 4.7934 | 2.8775 | 0.0957 | 28800 | 480 | cluster-robust (cluster=cell) |
| FieldSim_v2_sensitivity | share(triple_welfare>0) | net_welfare | 0.611 |  |  | 72 |  | grid over seeds × knobs |
| FieldSim_v2_sensitivity | share(p<0.05) | net_welfare | 0.139 |  |  | 72 |  | p-value for ai×central×hard (clustered) |
| Regime_map | λ* @ low info | attention_cost threshold | 0.014 |  |  |  |  | k_I=1,k_J=1; central wins if λ > λ* |
| Regime_map | λ* @ high info | attention_cost threshold | 0.0123 |  |  |  |  | k_I=6,k_J=6; central wins if λ > λ* |
| Regime_map | λ* range | attention_cost threshold |  |  |  |  |  | min=0.0120, max=0.0144 |
| Congestion_saturation | Δ net_welfare (0.25 - 0.00) | net_welfare_per_customer (easy) | -0.066 |  |  |  |  | treated match_rate=0.825, control match_rate=0.649 |
| Congestion_saturation | Δ net_welfare (1.00 - 0.00) | net_welfare_per_customer (easy) | -0.892 |  |  |  |  | inbox/provider/day@1.0=10.50, response_rate@1.0=0.021 |
| Congestion_saturation | Δ net_welfare (0.25 - 0.00) | net_welfare_per_customer (hard) | -0.066 |  |  |  |  | treated match_rate=0.825, control match_rate=0.623 |
| Congestion_saturation | Δ net_welfare (1.00 - 0.00) | net_welfare_per_customer (hard) | -0.883 |  |  |  |  | inbox/provider/day@1.0=10.58, response_rate@1.0=0.020 |
| LLM_parsing_quality | form_top3 | mean_l1 (easy,customer) | 0.1879 |  |  |  |  | top1_acc=0.700 |
| LLM_parsing_quality | form_top3 | mean_l1 (easy,provider) | 0.1398 |  |  |  |  | top1_acc=0.600 |
| LLM_parsing_quality | free_text_gpt | mean_l1 (easy,customer) | 0.5543 |  |  |  |  | top1_acc=0.633 |
| LLM_parsing_quality | free_text_gpt | mean_l1 (easy,provider) | 0.5552 |  |  |  |  | top1_acc=0.567 |
| LLM_parsing_quality | chat_gpt | mean_l1 (easy,customer) | 0.154 |  |  |  |  | top1_acc=0.700 |
| LLM_parsing_quality | chat_gpt | mean_l1 (easy,provider) | 0.1451 |  |  |  |  | top1_acc=0.600 |
| LLM_parsing_quality | form_top3 | mean_l1 (hard,customer) | 0.2165 |  |  |  |  | top1_acc=0.733 |
| LLM_parsing_quality | form_top3 | mean_l1 (hard,provider) | 0.1915 |  |  |  |  | top1_acc=0.833 |
| LLM_parsing_quality | free_text_gpt | mean_l1 (hard,customer) | 0.4437 |  |  |  |  | top1_acc=0.633 |
| LLM_parsing_quality | free_text_gpt | mean_l1 (hard,provider) | 0.5427 |  |  |  |  | top1_acc=0.733 |
| LLM_parsing_quality | chat_gpt | mean_l1 (hard,customer) | 0.1311 |  |  |  |  | top1_acc=0.733 |
| LLM_parsing_quality | chat_gpt | mean_l1 (hard,provider) | 0.1546 |  |  |  |  | top1_acc=0.833 |
