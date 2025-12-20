# LLM-Backed Heterogeneity Sweep

**Uses real GPT calls** to parse preference descriptions, unlike the
synthetic simulation in `field_sim_v2_heterogeneity.py`.

## Utility Noise Sweep Results

| utility_noise_sd | mean_did | se |
|------------------|----------|-----|
| 0.04 | -0.0065 | 0.0 |
| 0.08 | -0.0061 | 0.0 |
| 0.15 | -0.0069 | 0.0 |

## Run

```bash
make heterogeneity-llm
```
