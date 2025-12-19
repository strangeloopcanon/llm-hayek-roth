# LLM ablations: bits vs parsing vs agent

So what: this report separates three channels by construction:
- more info collected (form_top3 vs free_text_rules)
- LLM parsing/structuring (free_text_gpt vs free_text_rules, same text)
- conversational elicitation (chat_gpt vs free_text_gpt, same parser)

Quick read (net_welfare per customer):
- hard: LLM parsing uplift on centralized = +0.114
- hard: agent uplift on centralized = +0.043
- easy: LLM parsing uplift on centralized = +0.106

Arms (per category):
- form_top3: structured form, deterministic parse
- free_text_rules: richer free text, rules-based parse (non-LLM baseline)
- free_text_gpt: same free text, parsed by GPT-5.2
- chat_gpt: conversational transcript, parsed by GPT-5.2

Artifacts:
- `parsing_quality_*.md`: L1 weight error and top-1 accuracy vs truth
- `ablations_summary_*.md`: match/welfare by (mode × mechanism)
- `fig_*`: net welfare bar charts

Run:
- `uv run python -m econ_llm_preferences_experiment.llm_ablations`

