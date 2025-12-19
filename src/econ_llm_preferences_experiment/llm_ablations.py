from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text_topk,
    standard_long_form_text,
)
from econ_llm_preferences_experiment.env import load_dotenv
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
)
from econ_llm_preferences_experiment.models import (
    DIMENSIONS,
    AgentInferred,
    AgentTruth,
    Category,
    MatchOutcome,
)
from econ_llm_preferences_experiment.openai_client import OpenAIClient, OpenAIUsage
from econ_llm_preferences_experiment.plotting import Bar, write_bar_chart_svg
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
    preference_density_proxy,
)

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int

ElicitationMode: TypeAlias = Literal["form_top3", "free_text_rules", "free_text_gpt", "chat_gpt"]
Mechanism: TypeAlias = Literal["search", "central"]

MECHANISMS: tuple[Mechanism, ...] = ("search", "central")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _level_to_weight(level: str) -> float:
    lvl = level.strip().lower()
    if lvl.startswith("high"):
        return 0.35
    if lvl.startswith("medium"):
        return 0.20
    if lvl.startswith("low"):
        return 0.10
    if lvl.startswith("none"):
        return 0.00
    return 0.0


def _renormalize(weights: list[float]) -> tuple[float, ...]:
    vals = [max(0.0, float(x)) for x in weights[: len(DIMENSIONS)]]
    total = sum(vals)
    if total <= 0.0:
        return tuple(1.0 / len(vals) for _ in vals)
    return tuple(v / total for v in vals)


def _parse_form_topk(text: str) -> tuple[float, ...]:
    weights: list[float] = [0.0 for _ in DIMENSIONS]
    for idx, dim in enumerate(DIMENSIONS):
        for line in text.splitlines():
            if dim not in line:
                continue
            if "high" in line.lower():
                weights[idx] = 0.35
            elif "medium" in line.lower():
                weights[idx] = 0.20
            elif "low" in line.lower():
                weights[idx] = 0.10
            elif "none" in line.lower():
                weights[idx] = 0.00
    return _renormalize(weights)


_SYNONYMS: dict[str, tuple[str, ...]] = {
    "price_sensitivity": ("price", "cost", "budget"),
    "quality_focus": ("quality", "craftsmanship"),
    "speed_urgency": ("speed", "turnaround", "asap", "urgent"),
    "communication_fit": ("communication", "updates", "responsiveness"),
    "weirdness_tolerance": ("weird", "unusual", "edge", "surprises"),
    "schedule_flexibility": ("schedule", "timing", "availability"),
}


def _parse_free_text_rules(text: str) -> tuple[float, ...]:
    """
    A simple, non-LLM baseline parser: look for (synonym -> level) cues in free text.
    """
    t = text.lower()
    weights: list[float] = [0.0 for _ in DIMENSIONS]

    for idx, dim in enumerate(DIMENSIONS):
        cues = list(_SYNONYMS.get(dim, (dim,))) + [dim]
        best = 0.0
        for cue in cues:
            for m in re.finditer(re.escape(cue), t):
                window = t[max(0, m.start() - 40) : min(len(t), m.end() + 40)]
                for lvl in ("high", "medium", "low", "none"):
                    if lvl in window:
                        best = max(best, _level_to_weight(lvl))
        weights[idx] = best
    return _renormalize(weights)


def _infer_from_texts(
    *,
    mode: ElicitationMode,
    agents: tuple[AgentTruth, ...],
    texts_by_agent_id: dict[str, str],
    client: OpenAIClient | None,
    skip_llm: bool,
) -> tuple[dict[str, AgentInferred], dict[str, str], OpenAIUsage]:
    truth_by_id = {a.agent_id: a for a in agents}
    if mode == "form_top3":
        inferred = {
            a.agent_id: AgentInferred(
                agent_id=a.agent_id,
                side=a.side,
                category=a.category,
                weights=_parse_form_topk(texts_by_agent_id[a.agent_id]),
                tags=(),
            )
            for a in agents
        }
        return inferred, texts_by_agent_id, OpenAIUsage()

    if mode in {"free_text_rules", "free_text_gpt"}:
        if mode == "free_text_rules" or skip_llm:
            inferred = {
                a.agent_id: AgentInferred(
                    agent_id=a.agent_id,
                    side=a.side,
                    category=a.category,
                    weights=_parse_free_text_rules(texts_by_agent_id[a.agent_id]),
                    tags=(),
                )
                for a in agents
            }
            return inferred, texts_by_agent_id, OpenAIUsage()

        if client is None:
            raise RuntimeError("client is required for free_text_gpt when skip_llm=False")
        parsed = parse_batch_with_gpt(
            client=client, texts_by_agent_id=texts_by_agent_id, truth_by_agent_id=truth_by_id
        )
        return {a.agent_id: a for a in parsed.inferred}, texts_by_agent_id, parsed.usage

    if mode == "chat_gpt":
        if skip_llm:
            inferred = {
                a.agent_id: AgentInferred(
                    agent_id=a.agent_id,
                    side=a.side,
                    category=a.category,
                    weights=_parse_free_text_rules(texts_by_agent_id[a.agent_id]),
                    tags=(),
                )
                for a in agents
            }
            return inferred, texts_by_agent_id, OpenAIUsage()

        if client is None:
            raise RuntimeError("client is required for chat_gpt when skip_llm=False")
        parsed = parse_batch_with_gpt(
            client=client, texts_by_agent_id=texts_by_agent_id, truth_by_agent_id=truth_by_id
        )
        return {a.agent_id: a for a in parsed.inferred}, texts_by_agent_id, parsed.usage

    raise ValueError(f"Unknown mode: {mode}")


@dataclass(frozen=True)
class ParseQuality:
    mode: ElicitationMode
    side: str
    category: Category
    mean_l1: float
    mean_top1_correct: float


def _parse_quality(
    *,
    mode: ElicitationMode,
    agents: tuple[AgentTruth, ...],
    inferred_by_id: dict[str, AgentInferred],
) -> list[ParseQuality]:
    out: list[ParseQuality] = []
    for side in ("customer", "provider"):
        subset = [a for a in agents if a.side == side]
        l1s = []
        top1 = []
        for a in subset:
            inf = inferred_by_id[a.agent_id]
            l1 = sum(abs(t - h) for t, h in zip(a.weights, inf.weights, strict=True))
            l1s.append(l1)
            top_true = max(range(len(a.weights)), key=lambda idx: a.weights[idx])
            top_hat = max(range(len(inf.weights)), key=lambda idx: inf.weights[idx])
            top1.append(1.0 if top_true == top_hat else 0.0)
        out.append(
            ParseQuality(
                mode=mode,
                side=side,
                category=subset[0].category if subset else "easy",
                mean_l1=_mean(l1s),
                mean_top1_correct=_mean(top1),
            )
        )
    return out


def _arm_label(mode: ElicitationMode, mechanism: Mechanism) -> str:
    return f"{mode}_{mechanism}"


def run(
    *,
    out_dir: Path,
    seed: int,
    category: Category,
    replications: int,
    attention_cost: float,
    skip_llm: bool,
) -> dict[str, float]:
    rng = random.Random(seed)  # nosec B311
    market_params = MarketParams()

    customers, providers = generate_population(
        rng=rng,
        category=category,
        n_customers=market_params.n_customers,
        n_providers=market_params.n_providers,
    )
    agents = customers + providers

    client = None
    if not skip_llm:
        load_dotenv()
        client = OpenAIClient(max_calls=9)

    modes: list[ElicitationMode] = ["form_top3", "free_text_rules", "free_text_gpt", "chat_gpt"]

    texts_form = {a.agent_id: standard_form_text_topk(a, top_k=3) for a in agents}
    free_rng = random.Random(seed + 77)  # nosec B311
    texts_free = {a.agent_id: standard_long_form_text(a, rng=free_rng) for a in agents}
    chat_rng = random.Random(seed + 99)  # nosec B311
    texts_chat = {a.agent_id: ai_conversation_transcript(a, rng=chat_rng) for a in agents}

    input_texts_by_mode: dict[ElicitationMode, dict[str, str]] = {
        "form_top3": texts_form,
        "free_text_rules": texts_free,
        "free_text_gpt": texts_free,
        "chat_gpt": texts_chat,
    }

    inferred_by_mode: dict[ElicitationMode, dict[str, AgentInferred]] = {}
    usage_by_mode: dict[ElicitationMode, OpenAIUsage] = {}

    for mode in modes:
        inf, _texts, usage = _infer_from_texts(
            mode=mode,
            agents=agents,
            texts_by_agent_id=input_texts_by_mode[mode],
            client=client,
            skip_llm=skip_llm,
        )
        inferred_by_mode[mode] = inf
        usage_by_mode[mode] = usage

    # Parse quality diagnostics (relative to truth weights).
    pq_rows: list[dict[str, RowValue]] = []
    for mode in modes:
        for row in _parse_quality(mode=mode, agents=agents, inferred_by_id=inferred_by_mode[mode]):
            pq_rows.append(
                {
                    "category": row.category,
                    "mode": row.mode,
                    "side": row.side,
                    "mean_l1": round(row.mean_l1, 4),
                    "top1_accuracy": round(row.mean_top1_correct, 3),
                }
            )
    _write_csv(pq_rows, out_dir / f"parsing_quality_{category}.csv")
    _write_md_table(pq_rows, out_dir / f"parsing_quality_{category}.md")

    # Main outcomes.
    arm_rows: list[dict[str, RowValue]] = []
    for mode in modes:
        w_customer = tuple(inferred_by_mode[mode][a.agent_id].weights for a in customers)
        w_provider = tuple(inferred_by_mode[mode][a.agent_id].weights for a in providers)

        d_i: list[float] = []
        d_j: list[float] = []
        match_rate: dict[Mechanism, list[float]] = {m: [] for m in MECHANISMS}
        net_welfare: dict[Mechanism, list[float]] = {m: [] for m in MECHANISMS}
        total_value: dict[Mechanism, list[float]] = {m: [] for m in MECHANISMS}
        attention: dict[Mechanism, list[float]] = {m: [] for m in MECHANISMS}

        for r in range(replications):
            rep_rng = random.Random(seed * 10_000 + r)  # nosec B311
            market = generate_market_instance(
                rng=rep_rng,
                customers=customers,
                providers=providers,
                idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
            )

            vhat_c = inferred_value_matrix(
                weights_by_agent=w_customer, partner_attributes=market.provider_attributes
            )
            vhat_p = inferred_value_matrix(
                weights_by_agent=w_provider, partner_attributes=market.customer_attributes
            )
            d_i.append(
                preference_density_proxy(
                    v_true=market.v_customer, v_hat=vhat_c, epsilon=market_params.epsilon
                )
            )
            d_j.append(
                preference_density_proxy(
                    v_true=market.v_provider, v_hat=vhat_p, epsilon=market_params.epsilon
                )
            )

            out_search = decentralized_search(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c,
                accept_threshold=market_params.accept_threshold,
                params=SearchParams(max_rounds=30),
            )
            out_central = centralized_recommendations(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c,
                v_provider_hat=vhat_p,
                accept_threshold=market_params.accept_threshold,
                params=CentralizedParams(rec_k=5),
            )

            outcomes: dict[Mechanism, MatchOutcome] = {
                "search": out_search,
                "central": out_central,
            }
            for mech, outcome in outcomes.items():
                m = len(outcome.matches)
                denom = min(len(customers), len(providers))
                mr = m / denom if denom else 0.0
                match_rate[mech].append(mr)
                tv_sum = sum(
                    market.v_customer[i][j] + market.v_provider[j][i] for i, j in outcome.matches
                )
                actions = float(outcome.proposals + outcome.accept_decisions)
                total_value[mech].append(tv_sum / max(1, len(customers)))
                attention[mech].append(actions / max(1, len(customers)))
                net_welfare[mech].append(
                    (tv_sum / max(1, len(customers)))
                    - attention_cost * (actions / max(1, len(customers)))
                )

        for mech in MECHANISMS:
            arm_rows.append(
                {
                    "category": category,
                    "arm": _arm_label(mode, mech),
                    "mode": mode,
                    "mechanism": mech,
                    "match_rate": round(_mean(match_rate[mech]), 3),
                    "match_rate_se": round(_se(match_rate[mech]), 3),
                    "d_hat_i": round(_mean(d_i), 3),
                    "d_hat_j": round(_mean(d_j), 3),
                    "total_value_per_customer": round(_mean(total_value[mech]), 3),
                    "attention_per_customer": round(_mean(attention[mech]), 3),
                    "net_welfare_per_customer": round(_mean(net_welfare[mech]), 3),
                    "llm_input_tokens": int(usage_by_mode[mode].input_tokens or 0),
                    "llm_output_tokens": int(usage_by_mode[mode].output_tokens or 0),
                }
            )

    _write_csv(arm_rows, out_dir / f"ablations_summary_{category}.csv")
    _write_md_table(arm_rows, out_dir / f"ablations_summary_{category}.md")

    # Plots: net welfare by arm, split by mechanism.
    for mech in MECHANISMS:
        bars = [
            Bar(
                label=_arm_label(mode, mech),
                value=_as_float(
                    next(
                        r["net_welfare_per_customer"]
                        for r in arm_rows
                        if r["mode"] == mode and r["mechanism"] == mech
                    )
                ),
            )
            for mode in modes
        ]
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_{mech}_net_welfare.svg",
            title=f"Net welfare per customer ({category}, mech={mech})",
            bars=bars,
            y_label="net_welfare_per_customer",
        )

    # Write a small sample of texts for inspection.
    sample_ids = [agents[i].agent_id for i in range(min(3, len(agents)))]
    sample: dict[str, dict[str, str]] = {}
    for mode in modes:
        sample[mode] = {agent_id: input_texts_by_mode[mode][agent_id] for agent_id in sample_ids}
    (out_dir / f"sample_texts_{category}.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def stat(mode: ElicitationMode, mech: Mechanism, key: str) -> float:
        for r in arm_rows:
            if r["mode"] == mode and r["mechanism"] == mech:
                return _as_float(r[key])
        raise KeyError(f"Missing {category} {mode} {mech} {key}")

    return {
        "bits_uplift_central": stat("free_text_rules", "central", "net_welfare_per_customer")
        - stat("form_top3", "central", "net_welfare_per_customer"),
        "llm_parse_uplift_central": stat("free_text_gpt", "central", "net_welfare_per_customer")
        - stat("free_text_rules", "central", "net_welfare_per_customer"),
        "agent_uplift_central": stat("chat_gpt", "central", "net_welfare_per_customer")
        - stat("free_text_gpt", "central", "net_welfare_per_customer"),
        "llm_parse_uplift_search": stat("free_text_gpt", "search", "net_welfare_per_customer")
        - stat("free_text_rules", "search", "net_welfare_per_customer"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/ablations_latest")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--attention-cost", type=float, default=0.25)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": args.seed,
        "replications": args.replications,
        "attention_cost": args.attention_cost,
        "market_params": MarketParams().__dict__,
        "dimensions": list(DIMENSIONS),
        "skip_llm": bool(args.skip_llm),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for category in ("easy", "hard"):
        log(logger, 20, "ablations_start", category=category, skip_llm=bool(args.skip_llm))
        stats = run(
            out_dir=out_dir,
            seed=args.seed + (0 if category == "easy" else 100),
            category=category,
            replications=args.replications,
            attention_cost=args.attention_cost,
            skip_llm=bool(args.skip_llm),
        )
        (out_dir / f"quick_stats_{category}.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log(logger, 20, "ablations_done", category=category)

    hard_stats = json.loads((out_dir / "quick_stats_hard.json").read_text(encoding="utf-8"))
    easy_stats = json.loads((out_dir / "quick_stats_easy.json").read_text(encoding="utf-8"))

    readme = "\n".join(
        [
            "# LLM ablations: bits vs parsing vs agent",
            "",
            "So what: this report separates three channels by construction:",
            "- more info collected (form_top3 vs free_text_rules)",
            "- LLM parsing/structuring (free_text_gpt vs free_text_rules, same text)",
            "- conversational elicitation (chat_gpt vs free_text_gpt, same parser)",
            "",
            "Quick read (net_welfare per customer):",
            f"- hard: LLM parsing uplift on centralized = "
            f"{hard_stats['llm_parse_uplift_central']:+.3f}",
            f"- hard: agent uplift on centralized = {hard_stats['agent_uplift_central']:+.3f}",
            f"- easy: LLM parsing uplift on centralized = "
            f"{easy_stats['llm_parse_uplift_central']:+.3f}",
            "",
            "Arms (per category):",
            "- form_top3: structured form, deterministic parse",
            "- free_text_rules: richer free text, rules-based parse (non-LLM baseline)",
            "- free_text_gpt: same free text, parsed by GPT-5.2",
            "- chat_gpt: conversational transcript, parsed by GPT-5.2",
            "",
            "Artifacts:",
            "- `parsing_quality_*.md`: L1 weight error and top-1 accuracy vs truth",
            "- `ablations_summary_*.md`: match/welfare by (mode × mechanism)",
            "- `fig_*`: net welfare bar charts",
            "",
            "Run:",
            "- `uv run python -m econ_llm_preferences_experiment.llm_ablations`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")

    log(logger, 20, "ablations_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
