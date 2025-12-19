from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.models import AgentTruth, Category
from econ_llm_preferences_experiment.openai_client import OpenAIClient
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
)

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


@dataclass(frozen=True)
class CongestionParams:
    horizon_days: int = 7
    k_manual_per_day: int = 2
    k_agent_per_day: int = 10
    provider_daily_response_cap: int = 5
    provider_weekly_capacity: int = 4
    customer_accept_threshold: float = 0.25
    provider_accept_threshold: float = 0.25
    provider_threshold_uplift_per_extra_cap: float = 0.10
    provider_threshold_cap: float = 0.55
    manual_ranking_noise_sd: float = 0.18
    agent_ranking_noise_sd: float = 0.06
    provider_eval_noise_sd: float = 0.04
    message_quality_bonus: float = 0.03
    attention_cost: float = 0.01


@dataclass(frozen=True)
class CellOutcomes:
    saturation: float
    match_rate_all: float
    match_rate_treated: float
    match_rate_control: float
    mean_days_to_match_all: float
    mean_days_to_match_treated: float
    mean_days_to_match_control: float
    messages_sent_per_customer: float
    responses_sent_per_provider: float
    inbox_per_provider_per_day: float
    provider_response_rate: float
    overload_rate_provider_days: float
    net_welfare_per_customer: float


def _simulate_cell(
    *,
    rng: random.Random,
    saturation: float,
    congestion_params: CongestionParams,
    customers: tuple[AgentTruth, ...],
    providers: tuple[AgentTruth, ...],
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    vhat_customer_standard: tuple[tuple[float, ...], ...],
    vhat_customer_ai: tuple[tuple[float, ...], ...],
) -> CellOutcomes:
    n_c = len(customers)
    n_p = len(providers)
    n_treated = int(round(saturation * n_c))
    treated = set(rng.sample(list(range(n_c)), k=n_treated)) if n_treated > 0 else set()

    matched_provider: list[int | None] = [None for _ in range(n_c)]
    match_day: list[int | None] = [None for _ in range(n_c)]
    contacted: list[set[int]] = [set() for _ in range(n_c)]

    provider_capacity = [congestion_params.provider_weekly_capacity for _ in range(n_p)]

    messages_sent = 0
    responses_sent = 0
    provider_inbox_total = 0
    provider_overload_days = 0

    for day in range(1, congestion_params.horizon_days + 1):
        inbox: list[list[int]] = [[] for _ in range(n_p)]
        for i in range(n_c):
            if matched_provider[i] is not None:
                continue
            k = (
                congestion_params.k_agent_per_day
                if i in treated
                else congestion_params.k_manual_per_day
            )
            if k <= 0:
                continue

            base_scores = vhat_customer_ai[i] if i in treated else vhat_customer_standard[i]
            noise_sd = (
                congestion_params.agent_ranking_noise_sd
                if i in treated
                else congestion_params.manual_ranking_noise_sd
            )
            scored = []
            for j in range(n_p):
                if provider_capacity[j] <= 0 or j in contacted[i]:
                    continue
                scored.append((base_scores[j] + rng.gauss(0.0, noise_sd), j))
            scored.sort(reverse=True)
            for _score, j in scored[:k]:
                inbox[j].append(i)
                contacted[i].add(j)
                messages_sent += 1

        responses_by_customer: list[list[int]] = [[] for _ in range(n_c)]
        for j in range(n_p):
            incoming = inbox[j]
            if not incoming:
                continue
            provider_inbox_total += len(incoming)
            if len(incoming) > congestion_params.provider_daily_response_cap:
                provider_overload_days += 1
            if provider_capacity[j] <= 0:
                continue

            load_ratio = len(incoming) / max(1, congestion_params.provider_daily_response_cap)
            threshold = (
                congestion_params.provider_accept_threshold
                + congestion_params.provider_threshold_uplift_per_extra_cap
                * max(0.0, load_ratio - 1.0)
            )
            threshold = min(congestion_params.provider_threshold_cap, threshold)

            candidates = []
            for i in incoming:
                observed = v_provider_true[j][i]
                if i in treated:
                    observed += congestion_params.message_quality_bonus
                observed += rng.gauss(0.0, congestion_params.provider_eval_noise_sd)
                observed = 0.0 if observed < 0.0 else 1.0 if observed > 1.0 else observed
                if observed < threshold:
                    continue
                candidates.append((observed, i))
            candidates.sort(reverse=True)
            for _obs, i in candidates[: congestion_params.provider_daily_response_cap]:
                responses_by_customer[i].append(j)
                responses_sent += 1

        accepts_by_provider: list[list[int]] = [[] for _ in range(n_p)]
        for i in range(n_c):
            if matched_provider[i] is not None:
                continue
            offers = responses_by_customer[i]
            if not offers:
                continue
            best_j = max(offers, key=lambda j: v_customer_true[i][j])
            if v_customer_true[i][best_j] >= congestion_params.customer_accept_threshold:
                accepts_by_provider[best_j].append(i)

        for j in range(n_p):
            if provider_capacity[j] <= 0:
                continue
            accepters = accepts_by_provider[j]
            if not accepters:
                continue
            accepters.sort(key=lambda i: v_provider_true[j][i], reverse=True)
            for i in accepters:
                if provider_capacity[j] <= 0 or matched_provider[i] is not None:
                    continue
                provider_capacity[j] -= 1
                matched_provider[i] = j
                match_day[i] = day

    matched_indices = [i for i in range(n_c) if matched_provider[i] is not None]
    match_rate_all = len(matched_indices) / n_c if n_c else 0.0

    treated_indices = [i for i in range(n_c) if i in treated]
    control_indices = [i for i in range(n_c) if i not in treated]

    def rate(indices: list[int]) -> float:
        if not indices:
            return 0.0
        return sum(1 for i in indices if matched_provider[i] is not None) / len(indices)

    def mean_days(indices: list[int]) -> float:
        days: list[int] = []
        for i in indices:
            d = match_day[i]
            if d is not None:
                days.append(d)
        if not days:
            return float(congestion_params.horizon_days)
        return sum(days) / len(days)

    match_rate_treated = rate(treated_indices)
    match_rate_control = rate(control_indices)
    mean_days_all = mean_days(list(range(n_c)))
    mean_days_treated = mean_days(treated_indices)
    mean_days_control = mean_days(control_indices)

    # Welfare: realized match value minus attention costs on both sides.
    total_value = 0.0
    for i in matched_indices:
        provider_idx = matched_provider[i]
        if provider_idx is None:
            raise RuntimeError("matched_indices invariant violated: provider_idx is None")
        total_value += v_customer_true[i][provider_idx] + v_provider_true[provider_idx][i]

    # Each sent message creates screening burden; responses are additional actions.
    total_actions = messages_sent + provider_inbox_total + responses_sent
    net_welfare_per_customer = (
        (total_value / n_c) - congestion_params.attention_cost * (total_actions / n_c)
        if n_c
        else 0.0
    )

    inbox_per_provider_per_day = (
        provider_inbox_total / (n_p * congestion_params.horizon_days) if n_p else 0.0
    )
    responses_sent_per_provider = responses_sent / n_p if n_p else 0.0
    provider_response_rate = responses_sent / provider_inbox_total if provider_inbox_total else 0.0
    overload_rate = provider_overload_days / (n_p * congestion_params.horizon_days) if n_p else 0.0
    messages_sent_per_customer = messages_sent / n_c if n_c else 0.0

    return CellOutcomes(
        saturation=saturation,
        match_rate_all=match_rate_all,
        match_rate_treated=match_rate_treated,
        match_rate_control=match_rate_control,
        mean_days_to_match_all=mean_days_all,
        mean_days_to_match_treated=mean_days_treated,
        mean_days_to_match_control=mean_days_control,
        messages_sent_per_customer=messages_sent_per_customer,
        responses_sent_per_provider=responses_sent_per_provider,
        inbox_per_provider_per_day=inbox_per_provider_per_day,
        provider_response_rate=provider_response_rate,
        overload_rate_provider_days=overload_rate,
        net_welfare_per_customer=net_welfare_per_customer,
    )


def run_saturation_experiment(
    *,
    category: Category,
    out_dir: Path,
    seed: int,
    n_cells_per_saturation: int,
    saturations: list[float],
    market_params: MarketParams,
    congestion_params: CongestionParams,
    client: OpenAIClient,
    n_customers: int,
    n_providers: int,
) -> list[dict[str, RowValue]]:
    rng = random.Random(seed)  # nosec B311
    customers, providers = generate_population(
        rng=rng, category=category, n_customers=n_customers, n_providers=n_providers
    )
    truth_by_id = {a.agent_id: a for a in customers + providers}
    std_texts = {a.agent_id: standard_form_text(a) for a in customers + providers}
    ai_texts = {a.agent_id: ai_conversation_transcript(a, rng=rng) for a in customers + providers}

    log(logger, 20, "gpt_parse_start", category=category, agents=len(truth_by_id))
    parsed_std = parse_batch_with_gpt(
        client=client, texts_by_agent_id=std_texts, truth_by_agent_id=truth_by_id
    )
    parsed_ai = parse_batch_with_gpt(
        client=client, texts_by_agent_id=ai_texts, truth_by_agent_id=truth_by_id
    )
    log(logger, 20, "gpt_parse_done", category=category)

    std_by_id = {a.agent_id: a for a in parsed_std.inferred}
    ai_by_id = {a.agent_id: a for a in parsed_ai.inferred}
    w_customer_std = tuple(std_by_id[a.agent_id].weights for a in customers)
    w_customer_ai = tuple(ai_by_id[a.agent_id].weights for a in customers)

    rows: list[dict[str, RowValue]] = []
    summary: dict[float, list[CellOutcomes]] = {s: [] for s in saturations}

    for s in saturations:
        for c in range(n_cells_per_saturation):
            cell_seed = seed * 1_000_000 + int(s * 10_000) * 10_000 + c
            cell_rng = random.Random(cell_seed)  # nosec B311
            market = generate_market_instance(
                rng=cell_rng,
                customers=customers,
                providers=providers,
                idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
            )
            vhat_std = inferred_value_matrix(
                weights_by_agent=w_customer_std, partner_attributes=market.provider_attributes
            )
            vhat_ai = inferred_value_matrix(
                weights_by_agent=w_customer_ai, partner_attributes=market.provider_attributes
            )
            out = _simulate_cell(
                rng=cell_rng,
                saturation=s,
                congestion_params=congestion_params,
                customers=customers,
                providers=providers,
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                vhat_customer_standard=vhat_std,
                vhat_customer_ai=vhat_ai,
            )
            summary[s].append(out)

    for s in saturations:
        cells = summary[s]
        rows.append(
            {
                "category": category,
                "saturation": s,
                "match_rate_all": round(_mean([c.match_rate_all for c in cells]), 3),
                "match_rate_all_se": round(_se([c.match_rate_all for c in cells]), 3),
                "match_rate_treated": round(_mean([c.match_rate_treated for c in cells]), 3),
                "match_rate_treated_se": round(_se([c.match_rate_treated for c in cells]), 3),
                "match_rate_control": round(_mean([c.match_rate_control for c in cells]), 3),
                "match_rate_control_se": round(_se([c.match_rate_control for c in cells]), 3),
                "mean_days_to_match_all": round(
                    _mean([c.mean_days_to_match_all for c in cells]), 3
                ),
                "messages_sent_per_customer": round(
                    _mean([c.messages_sent_per_customer for c in cells]), 3
                ),
                "inbox_per_provider_per_day": round(
                    _mean([c.inbox_per_provider_per_day for c in cells]), 3
                ),
                "provider_response_rate": round(
                    _mean([c.provider_response_rate for c in cells]), 3
                ),
                "overload_rate_provider_days": round(
                    _mean([c.overload_rate_provider_days for c in cells]), 3
                ),
                "net_welfare_per_customer": round(
                    _mean([c.net_welfare_per_customer for c in cells]), 3
                ),
                "net_welfare_per_customer_se": round(
                    _se([c.net_welfare_per_customer for c in cells]), 3
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"congestion_saturation_{category}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / f"congestion_saturation_{category}.md"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    x = saturations

    def pts(key: str) -> list[tuple[float, float]]:
        return [(s, float(next(r[key] for r in rows if r["saturation"] == s))) for s in x]

    write_line_chart_svg(
        out_path=out_dir / f"fig_{category}_match_rate_vs_saturation.svg",
        title=f"Match rate vs saturation (category={category})",
        x_label="saturation (share of customers with delegated outreach agent)",
        y_label="match_rate",
        series=[
            LineSeries(label="all", points=pts("match_rate_all")),
            LineSeries(label="treated", points=pts("match_rate_treated")),
            LineSeries(label="control", points=pts("match_rate_control")),
        ],
    )
    write_line_chart_svg(
        out_path=out_dir / f"fig_{category}_provider_inbox_vs_saturation.svg",
        title=f"Provider inbox load vs saturation (category={category})",
        x_label="saturation",
        y_label="messages received per provider per day",
        series=[LineSeries(label="inbox", points=pts("inbox_per_provider_per_day"))],
    )
    write_line_chart_svg(
        out_path=out_dir / f"fig_{category}_net_welfare_vs_saturation.svg",
        title=f"Net welfare vs saturation (category={category})",
        x_label="saturation",
        y_label="net_welfare_per_customer",
        series=[LineSeries(label="net welfare", points=pts("net_welfare_per_customer"))],
    )

    meta = {
        "category": category,
        "seed": seed,
        "n_cells_per_saturation": n_cells_per_saturation,
        "saturations": saturations,
        "market_params": {
            "epsilon": market_params.epsilon,
            "accept_threshold": market_params.accept_threshold,
            "idiosyncratic_noise_sd": market_params.idiosyncratic_noise_sd,
        },
        "congestion_params": congestion_params.__dict__,
        "note": (
            "Home services context: customers send quote requests; providers have limited daily "
            "response capacity and become more selective under overload. A delegated outreach "
            "agent lowers per-contact cost (higher k/day) and improves targeting via richer "
            "elicitation."
        ),
    }
    (out_dir / f"congestion_meta_{category}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/latest_congestion")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cells", type=int, default=60)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    market_params = MarketParams(accept_threshold=0.25)
    congestion_params = CongestionParams(attention_cost=args.attention_cost)
    client = OpenAIClient(max_calls=9)

    rows_by_category: dict[str, list[dict[str, RowValue]]] = {}
    for category in ("easy", "hard"):
        rows_by_category[category] = run_saturation_experiment(
            category=category,
            out_dir=out_dir,
            seed=args.seed,
            n_cells_per_saturation=args.cells,
            saturations=saturations,
            market_params=market_params,
            congestion_params=congestion_params,
            client=client,
            n_customers=80,
            n_providers=40,
        )

    def best_net_welfare(rows: list[dict[str, RowValue]]) -> tuple[float, float]:
        best_row = max(rows, key=lambda r: float(r["net_welfare_per_customer"]))
        return float(best_row["saturation"]), float(best_row["net_welfare_per_customer"])

    easy_best_s, easy_best_w = best_net_welfare(rows_by_category["easy"])
    hard_best_s, hard_best_w = best_net_welfare(rows_by_category["hard"])

    readme = "\n".join(
        [
            "# Delegated outreach agents × congestion (saturation design)",
            "",
            "So what: delegated outreach agents create an adoption-intensity trade-off. They speed",
            "up matching for treated users, but can impose congestion + communication costs that",
            "lower net welfare at high saturation.",
            "",
            f"Quick read (attention_cost={args.attention_cost}):",
            f"- easy: welfare-max saturation = {easy_best_s} (net_welfare={easy_best_w:.3f})",
            f"- hard: welfare-max saturation = {hard_best_s} (net_welfare={hard_best_w:.3f})",
            "",
            "Context: home services marketplace. Customers request quotes; providers have",
            "limited daily response capacity.",
            "Providers also adapt by raising their accept threshold when their inbox is "
            "overloaded.",
            "",
            "Treatment: a delegated outreach agent that (i) elicits richer preferences",
            "(higher signal precision) and",
            "(ii) sends more outbound quote requests per day (lower marginal outreach cost).",
            "",
            "Design: within each (city×category×week) cell we randomize the fraction",
            "of customers treated (saturation).",
            "We report treated + control outcomes to surface congestion externalities.",
            "",
            "Artifacts:",
            "- `congestion_saturation_easy.csv` / `.md`",
            "- `congestion_saturation_hard.csv` / `.md`",
            "- `fig_*_vs_saturation.svg`",
            "- `congestion_meta_*.json`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    log(logger, 20, "congestion_report_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
