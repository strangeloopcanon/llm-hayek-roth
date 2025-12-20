# Paper Bundle (LLM-Only, No Synthetic)

**All results use real GPT-5.2 calls** via `parse_batch_with_gpt()`.

## Summary

| Experiment | Source | Real LLM? |
|------------|--------|-----------|
| Main 2×2 | `reports/latest` | ✅ Yes |
| Ablations | `reports/ablations_latest` | ✅ Yes |
| Congestion | `reports/latest_congestion` | ✅ Yes |
| Heterogeneity | `reports/heterogeneity_llm_latest` | ✅ Yes |

## Key Results

| block | metric | category | estimate | se | notes |
| --- | --- | --- | --- | --- | --- |
| Main_2x2_LLM | d_hat_I improvement (AI vs standard) | easy | -0.001 | 0.0 | real GPT-5.2 parsing |
| Main_2x2_LLM | match_rate DiD | easy | 0.001 | 0.002 | (AI_central - AI_search) - (std_central - std_search) |
| Main_2x2_LLM | net_welfare DiD | easy | 0.001 | 0.001 | welfare impact of AI+Central interaction |
| Main_2x2_LLM | λ* (ROI threshold) | easy | 0.0128 |  | central beats search if attention_cost > λ* |
| Main_2x2_LLM | d_hat_I improvement (AI vs standard) | hard | 0.006 | 0.0 | real GPT-5.2 parsing |
| Main_2x2_LLM | match_rate DiD | hard | 0.003 | 0.002 | (AI_central - AI_search) - (std_central - std_search) |
| Main_2x2_LLM | net_welfare DiD | hard | 0.002 | 0.002 | welfare impact of AI+Central interaction |
| Main_2x2_LLM | λ* (ROI threshold) | hard | 0.0126 |  | central beats search if attention_cost > λ* |
| LLM_ablations | form_top3 parsing quality | easy_customer | 0.1879 |  | top1_acc=0.700 |
| LLM_ablations | form_top3 parsing quality | easy_provider | 0.1398 |  | top1_acc=0.600 |
| LLM_ablations | free_text_gpt parsing quality | easy_customer | 0.5543 |  | top1_acc=0.633 |
| LLM_ablations | free_text_gpt parsing quality | easy_provider | 0.5552 |  | top1_acc=0.567 |
| LLM_ablations | chat_gpt parsing quality | easy_customer | 0.154 |  | top1_acc=0.700 |
| LLM_ablations | chat_gpt parsing quality | easy_provider | 0.1451 |  | top1_acc=0.600 |
| LLM_ablations | form_top3 parsing quality | hard_customer | 0.2165 |  | top1_acc=0.733 |
| LLM_ablations | form_top3 parsing quality | hard_provider | 0.1915 |  | top1_acc=0.833 |
| LLM_ablations | free_text_gpt parsing quality | hard_customer | 0.4437 |  | top1_acc=0.633 |
| LLM_ablations | free_text_gpt parsing quality | hard_provider | 0.5427 |  | top1_acc=0.733 |
| LLM_ablations | chat_gpt parsing quality | hard_customer | 0.1311 |  | top1_acc=0.733 |
| LLM_ablations | chat_gpt parsing quality | hard_provider | 0.1546 |  | top1_acc=0.833 |
| Congestion_LLM | Δ net_welfare (100% - 0%) | easy | -0.892 |  | inbox@100%=10.5/day |
| Congestion_LLM | Δ net_welfare (100% - 0%) | hard | -0.883 |  | inbox@100%=10.6/day |
| Heterogeneity_LLM | utility_noise effect on DiD | utility_noise=0.04 | -0.0065 | 0.0025 | real GPT-5.2 parsing |
| Heterogeneity_LLM | utility_noise effect on DiD | utility_noise=0.08 | -0.0061 | 0.003 | real GPT-5.2 parsing |
| Heterogeneity_LLM | utility_noise effect on DiD | utility_noise=0.15 | -0.0069 | 0.0052 | real GPT-5.2 parsing |

## How to regenerate

```bash
make experiment        # Main 2×2 with GPT
make ablations         # Parsing quality with GPT
make congestion        # Saturation effects with GPT
make heterogeneity-llm # Utility noise sweep with GPT
make paper-bundle-llm  # This bundle
```
