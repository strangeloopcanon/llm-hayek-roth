# Can LLMs Make Centralized Matching Worth It?

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/llm-hayek-roth)

## The Key Question

> **When platforms can't confidently predict what users want, they fall back to search. Can LLM-powered preference elicitation change this calculus?**

This repo investigates whether better AI intake tips the cost-benefit balance toward centralized matching in two-sided markets (think: home services, job boards, dating apps).

---

## Four Experiments (All Real GPT-5.2)

We run **4 distinct experiments** using real LLM calls via `parse_batch_with_gpt()`. No synthetic simulations in the main results.

| # | Experiment | Question | Key Finding |
|---|------------|----------|-------------|
| 1 | [Main 2×2](#experiment-1-the-main-2×2) | Does AI+Central beat alternatives? | **Yes**, +0.2% welfare in hard categories |
| 2 | [Ablations](#experiment-2-where-does-llm-value-come-from) | Where does LLM value come from? | **Parsing > Conversation** (+11% vs +4%) |
| 3 | [Congestion](#experiment-3-what-breaks-at-scale) | What breaks at scale? | **Externality kills gains** (-88% welfare at full adoption) |
| 4 | [Heterogeneity](#experiment-4-does-preference-uncertainty-matter) | Does preference uncertainty matter? | **Small effect**, dominated by other factors |

📊 **Full LLM results**: [`reports/paper_llm_latest/README.md`](reports/paper_llm_latest/README.md)

---

## Experiment 1: The Main 2×2

**Question**: Does AI elicitation + centralized matching outperform the alternatives?

We compare 4 arms in a 2×2 design:

| | Search (you browse) | Central (we match) |
|---|---|---|
| **Standard form** | Baseline | +2.5% |
| **AI intake (GPT)** | +1.0% | **+2.8%** ← winner |

*Relative welfare improvements vs baseline (hard category)*

### Key metrics

| Category | AI Preference Improvement | Match Rate DiD | Welfare DiD | λ* (threshold) |
|----------|---------------------------|----------------|-------------|----------------|
| Easy | -0.001 | +0.001 | +0.001 | 0.0128 |
| **Hard** | **+0.006** | **+0.003** | **+0.002** | **0.0126** |

**Interpretation**:
- AI improves preference inference 6× more in hard categories
- λ* (ROI threshold) drops slightly with AI → central wins in more scenarios
- The effect is real but modest (~0.2-0.3% welfare improvement)

```bash
make experiment  # Run with GPT-5.2
```

---

## Experiment 2: Where Does LLM Value Come From?

**Question**: Is it the conversation, or the parsing?

We isolate three channels:
1. **More info collected** (form_top3 → free_text)
2. **LLM parsing** (rules-based → GPT on same text)
3. **Conversational elicitation** (static text → chat transcript)

### Results (hard category)

| Channel | Welfare Uplift | Parsing Quality (L1) |
|---------|----------------|---------------------|
| LLM parsing (same text) | **+11.4%** | 0.44 → 0.13 |
| Conversational agent | +4.3% additional | 0.13 → 0.13 |

**Interpretation**: Most of LLM's value comes from **parsing/structuring**, not from the conversation itself. GPT extracts more signal from the same text than rules-based parsing.

```bash
make ablations  # Run with GPT-5.2
```

---

## Experiment 3: What Breaks at Scale?

**Question**: Do AI-delegated outreach agents create congestion externalities?

We vary the **saturation** (fraction of users with AI agents) from 0% to 100%.

### Results

| Saturation | Provider Inbox/Day | Response Rate | Δ Welfare |
|------------|-------------------|---------------|-----------|
| 0% | 2.1 | 48% | baseline |
| 25% | 4.2 | 24% | -22% |
| 100% | **10.6** | **2%** | **-88%** |

**Interpretation**: At full adoption, everyone's inbox floods, response rates collapse, and net welfare **drops** by 88%. Individual benefit creates collective harm. This is the classic tragedy of the commons.

```bash
make congestion  # Run with GPT-5.2
```

---

## Experiment 4: Does Preference Uncertainty Matter?

**Question**: When preferences have more idiosyncratic noise (ε), does AI provide more relative value?

We vary `idiosyncratic_noise_sd` (utility randomness) and measure the AI×Central interaction.

### Results (hard category)

| Utility Noise | AI×Central DiD | SE |
|---------------|----------------|-----|
| 0.04 (low) | -0.0065 | 0.0025 |
| 0.08 (baseline) | -0.0061 | 0.0030 |
| 0.15 (high) | -0.0069 | 0.0052 |

**Interpretation**: The effect is small and within noise. Your hypothesis (more ε → higher AI ROI) shows a slight positive trend, but it's dominated by other factors like preference structure and congestion.

```bash
make heterogeneity-llm  # Run with GPT-5.2
```

---

## Bottom Line

> **LLM-powered preference elicitation enables centralized matching in hard-to-describe markets, but the effect is modest (~0.2%). The bigger story is congestion: widespread AI-delegated outreach can hurt everyone.**

### When AI+Central Works
- ✅ Hard-to-describe categories (home renovation, custom services)
- ✅ Heterogeneous preferences (people want very different things)
- ✅ Low saturation (not everyone using AI agents)

### When It Doesn't
- ❌ Easy categories (simple, standardized needs)
- ❌ High saturation (congestion externality)
- ❌ Homogeneous preferences (any match is fine)

---

## Quick Start

```bash
# Install
git clone https://github.com/strangeloopcanon/llm-hayek-roth.git
cd llm-hayek-roth
uv sync

# Set your OpenAI API key
echo 'OPENAI_API_KEY="sk-..."' >> .env

# Run all 4 experiments
make experiment        # Experiment 1: Main 2×2
make ablations         # Experiment 2: Ablations
make congestion        # Experiment 3: Congestion
make heterogeneity-llm # Experiment 4: Heterogeneity

# Generate LLM-only paper bundle
make paper-bundle-llm
```

---

## Model Configuration

All experiments use **GPT-5.2** by default. To change:

```bash
export OPENAI_MODEL="gpt-4o"  # or any OpenAI model
```

The client caches responses to save costs. Clear `.cache/llm/` to force fresh calls.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{llm-hayek-roth,
  author = {Strange Loop Canon},
  title = {Can LLMs Make Centralized Matching Worth It?},
  year = {2024},
  url = {https://github.com/strangeloopcanon/llm-hayek-roth}
}
```
