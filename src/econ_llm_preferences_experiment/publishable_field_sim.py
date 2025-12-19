from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from econ_llm_preferences_experiment.econometrics import OLSResult, ols_cluster_robust
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.mechanisms import SearchParams, _hopcroft_karp
from econ_llm_preferences_experiment.models import DIMENSIONS, Category
from econ_llm_preferences_experiment.plotting import (
    Bar,
    LineSeries,
    write_bar_chart_svg,
    write_line_chart_svg,
)
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
)

logger = get_logger(__name__)

Elicitation = Literal["standard", "ai"]
Mechanism = Literal["search", "central"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _level_from_weight(w: float) -> str:
    if w >= 0.30:
        return "high"
    if w >= 0.18:
        return "medium"
    if w >= 0.08:
        return "low"
    return "none"


def _level_to_numeric(level: str) -> float:
    if level == "high":
        return 0.35
    if level == "medium":
        return 0.20
    if level == "low":
        return 0.10
    return 0.0


def _renormalize(weights: list[float]) -> tuple[float, ...]:
    vals = [max(0.0, float(w)) for w in weights]
    total = sum(vals)
    if total <= 0.0:
        return tuple(1.0 / len(vals) for _ in vals)
    return tuple(v / total for v in vals)


def infer_weights_from_truth(
    weights: tuple[float, ...], *, elicitation: Elicitation
) -> tuple[float, ...]:
    k = len(weights)
    if k != len(DIMENSIONS):
        raise ValueError("weights dimension mismatch")
    if elicitation == "standard":
        top = max(range(k), key=lambda idx: weights[idx])
        one_hot = [0.0 for _ in range(k)]
        one_hot[top] = 1.0
        return tuple(one_hot)
    mapped = [_level_to_numeric(_level_from_weight(w)) for w in weights]
    return _renormalize(mapped)


@dataclass(frozen=True)
class MeasurementStats:
    d_hat_i_obs: float
    d_hat_j_obs: float
    reciprocity_obs: float
    accept_decisions: int
    accepted_adj: list[list[int]]
    reciprocity_by_rank: dict[int, float]


def measure_recommendation_acceptance(
    *,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    v_customer_hat: tuple[tuple[float, ...], ...],
    v_provider_hat: tuple[tuple[float, ...], ...],
    accept_threshold: float,
    rec_k: int,
) -> MeasurementStats:
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)
    accepted_adj: list[list[int]] = [[] for _ in range(n_c)]

    accepts_i = 0
    accepts_j = 0
    accepts_both = 0
    total_pairs = 0
    reciprocity_by_rank_counts: dict[int, list[int]] = {
        rank: [0, 0] for rank in range(1, rec_k + 1)
    }

    for i in range(n_c):
        scores = [(v_customer_hat[i][j] + v_provider_hat[j][i], j) for j in range(n_p)]
        scores.sort(reverse=True)
        recs = [j for _s, j in scores[:rec_k]]
        for rank, j in enumerate(recs, start=1):
            total_pairs += 1
            accept_i = v_customer_true[i][j] >= accept_threshold
            accept_j = v_provider_true[j][i] >= accept_threshold
            accepts_i += 1 if accept_i else 0
            accepts_j += 1 if accept_j else 0
            both = accept_i and accept_j
            accepts_both += 1 if both else 0
            if both:
                accepted_adj[i].append(j)
                reciprocity_by_rank_counts[rank][0] += 1
            reciprocity_by_rank_counts[rank][1] += 1

    d_i = accepts_i / total_pairs if total_pairs else 0.0
    d_j = accepts_j / total_pairs if total_pairs else 0.0
    r = accepts_both / total_pairs if total_pairs else 0.0
    by_rank = {
        rank: (num / denom if denom else 0.0)
        for rank, (num, denom) in reciprocity_by_rank_counts.items()
    }
    return MeasurementStats(
        d_hat_i_obs=d_i,
        d_hat_j_obs=d_j,
        reciprocity_obs=r,
        accept_decisions=2 * total_pairs,
        accepted_adj=accepted_adj,
        reciprocity_by_rank=by_rank,
    )


@dataclass(frozen=True)
class SearchOutcomeDetailed:
    matches: tuple[tuple[int, int], ...]
    proposals_total: int
    proposals_by_customer: tuple[int, ...]
    rounds: int


def decentralized_search_detailed(
    *,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    v_customer_hat: tuple[tuple[float, ...], ...],
    accept_threshold: float,
    params: SearchParams,
) -> SearchOutcomeDetailed:
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)
    customer_order = [
        sorted(range(n_p), key=lambda j: v_customer_hat[i][j], reverse=True) for i in range(n_c)
    ]
    next_choice = [0 for _ in range(n_c)]
    proposals_by_customer = [0 for _ in range(n_c)]

    held_by_provider: list[int | None] = [None for _ in range(n_p)]
    proposals = 0
    rounds = 0

    unmatched_customers = set(range(n_c))

    for r in range(1, params.max_rounds + 1):
        rounds = r
        proposals_this_round: list[list[int]] = [[] for _ in range(n_p)]
        for i in list(unmatched_customers):
            if next_choice[i] >= n_p:
                unmatched_customers.discard(i)
                continue
            j = customer_order[i][next_choice[i]]
            next_choice[i] += 1
            proposals_by_customer[i] += 1
            proposals_this_round[j].append(i)
            proposals += 1

        any_activity = any(proposals_this_round)
        if not any_activity:
            break

        for j in range(n_p):
            candidates = proposals_this_round[j]
            held = held_by_provider[j]
            if held is not None:
                candidates.append(held)
            best_i: int | None = None
            best_val = -1.0
            for i in candidates:
                if v_customer_true[i][j] < accept_threshold:
                    continue
                if v_provider_true[j][i] < accept_threshold:
                    continue
                val = v_provider_true[j][i]
                if val > best_val:
                    best_val = val
                    best_i = i
            held_by_provider[j] = best_i

        unmatched_customers = set(range(n_c))
        for held_customer in held_by_provider:
            if held_customer is not None:
                unmatched_customers.discard(held_customer)

    matches = tuple((i, j) for j, i in enumerate(held_by_provider) if i is not None)
    return SearchOutcomeDetailed(
        matches=matches,
        proposals_total=proposals,
        proposals_by_customer=tuple(proposals_by_customer),
        rounds=rounds,
    )


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    category: Category
    elicitation: Elicitation
    mechanism: Mechanism
    match_rate: float
    total_value_per_customer: float
    attention_per_customer: float
    net_welfare_per_customer: float
    d_hat_i_obs: float
    d_hat_j_obs: float
    reciprocity_obs: float


def _arm_label(elicitation: Elicitation, mechanism: Mechanism) -> str:
    return f"{elicitation}_{mechanism}"


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def simulate_cell(
    *,
    rng: random.Random,
    cell_id: str,
    category: Category,
    elicitation: Elicitation,
    mechanism: Mechanism,
    market_params: MarketParams,
    rec_k: int,
    search_params: SearchParams,
    attention_cost: float,
) -> tuple[CellOutcome, list[dict[str, object]], list[dict[str, object]]]:
    customers, providers = generate_population(
        rng=rng,
        category=category,
        n_customers=market_params.n_customers,
        n_providers=market_params.n_providers,
    )
    market = generate_market_instance(
        rng=rng,
        customers=customers,
        providers=providers,
        idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
    )

    w_c = tuple(infer_weights_from_truth(a.weights, elicitation=elicitation) for a in customers)
    w_p = tuple(infer_weights_from_truth(a.weights, elicitation=elicitation) for a in providers)
    vhat_c = inferred_value_matrix(
        weights_by_agent=w_c, partner_attributes=market.provider_attributes
    )
    vhat_p = inferred_value_matrix(
        weights_by_agent=w_p, partner_attributes=market.customer_attributes
    )

    measurement = measure_recommendation_acceptance(
        v_customer_true=market.v_customer,
        v_provider_true=market.v_provider,
        v_customer_hat=vhat_c,
        v_provider_hat=vhat_p,
        accept_threshold=market_params.accept_threshold,
        rec_k=rec_k,
    )

    n_c = len(customers)
    n_p = len(providers)
    denom = min(n_c, n_p)

    matches: tuple[tuple[int, int], ...]
    proposals_total = 0
    accept_decisions = 0
    proposals_by_customer = [0 for _ in range(n_c)]

    if mechanism == "central":
        match_r = _hopcroft_karp(measurement.accepted_adj, n_left=n_c, n_right=n_p)
        pairs: list[tuple[int, int]] = []
        for j, i in enumerate(match_r):
            if i != -1:
                pairs.append((i, j))
        matches = tuple(pairs)
        proposals_total = 0
        accept_decisions = measurement.accept_decisions
    else:
        out = decentralized_search_detailed(
            v_customer_true=market.v_customer,
            v_provider_true=market.v_provider,
            v_customer_hat=vhat_c,
            accept_threshold=market_params.accept_threshold,
            params=search_params,
        )
        matches = out.matches
        proposals_total = out.proposals_total
        accept_decisions = 0
        proposals_by_customer = list(out.proposals_by_customer)

    matched_provider_by_customer: list[int | None] = [None for _ in range(n_c)]
    for i, j in matches:
        matched_provider_by_customer[i] = j

    total_value = 0.0
    customer_rows: list[dict[str, object]] = []
    for i in range(n_c):
        provider_idx = matched_provider_by_customer[i]
        matched = 1 if provider_idx is not None else 0
        if provider_idx is None:
            value = 0.0
        else:
            value = market.v_customer[i][provider_idx] + market.v_provider[provider_idx][i]
        total_value += value
        customer_rows.append(
            {
                "cell_id": cell_id,
                "category": category,
                "elicitation": elicitation,
                "mechanism": mechanism,
                "ai": 1 if elicitation == "ai" else 0,
                "central": 1 if mechanism == "central" else 0,
                "matched": matched,
                "total_value": round(value, 6),
                "proposals_sent": proposals_by_customer[i],
            }
        )

    pair_rows: list[dict[str, object]] = []
    for rank, r_val in measurement.reciprocity_by_rank.items():
        pair_rows.append(
            {
                "cell_id": cell_id,
                "category": category,
                "elicitation": elicitation,
                "mechanism": mechanism,
                "ai": 1 if elicitation == "ai" else 0,
                "central": 1 if mechanism == "central" else 0,
                "rank": rank,
                "reciprocity_rate": round(r_val, 6),
            }
        )

    match_rate = len(matches) / denom if denom else 0.0
    total_value_per_customer = total_value / n_c if n_c else 0.0
    attention_per_customer = (proposals_total + accept_decisions) / n_c if n_c else 0.0
    net_welfare = total_value_per_customer - attention_cost * attention_per_customer

    cell_out = CellOutcome(
        cell_id=cell_id,
        category=category,
        elicitation=elicitation,
        mechanism=mechanism,
        match_rate=match_rate,
        total_value_per_customer=total_value_per_customer,
        attention_per_customer=attention_per_customer,
        net_welfare_per_customer=net_welfare,
        d_hat_i_obs=measurement.d_hat_i_obs,
        d_hat_j_obs=measurement.d_hat_j_obs,
        reciprocity_obs=measurement.reciprocity_obs,
    )
    return cell_out, customer_rows, pair_rows


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_ols_table(*, result: OLSResult, names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "term": name,
                "coef": round(result.coef[idx], 4),
                "se(cluster)": round(result.se[idx], 4),
                "t": round(result.t[idx], 3),
                "p(normal)": round(result.p[idx], 4),
            }
        )
    rows.append(
        {
            "term": "n_obs / n_clusters",
            "coef": f"{result.n_obs} / {result.n_clusters}",
            "se(cluster)": "",
            "t": "",
            "p(normal)": "",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/field_latest")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cells-per-category", type=int, default=200)
    parser.add_argument("--n-customers", type=int, default=40)
    parser.add_argument("--n-providers", type=int, default=40)
    parser.add_argument("--rec-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_params = MarketParams(n_customers=args.n_customers, n_providers=args.n_providers)
    search_params = SearchParams(max_rounds=args.max_rounds)

    rng = random.Random(args.seed)  # nosec B311
    arms: list[tuple[Elicitation, Mechanism]] = [
        ("standard", "search"),
        ("standard", "central"),
        ("ai", "search"),
        ("ai", "central"),
    ]

    cells: list[CellOutcome] = []
    customer_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for category in ("easy", "hard"):
        cell_ids = [f"{category}_{i:04d}" for i in range(args.cells_per_category)]
        assignments = (arms * ((len(cell_ids) + len(arms) - 1) // len(arms)))[: len(cell_ids)]
        rng.shuffle(assignments)
        for cell_id, (elicitation, mechanism) in zip(cell_ids, assignments, strict=True):
            cell_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
            out, cust, ranks = simulate_cell(
                rng=cell_rng,
                cell_id=cell_id,
                category=category,
                elicitation=elicitation,
                mechanism=mechanism,
                market_params=market_params,
                rec_k=args.rec_k,
                search_params=search_params,
                attention_cost=args.attention_cost,
            )
            cells.append(out)
            customer_rows.extend(cust)
            rank_rows.extend(ranks)

    cell_rows = [asdict(c) for c in cells]
    _write_csv(cell_rows, out_dir / "cells.csv")
    _write_csv(customer_rows, out_dir / "customers.csv")
    _write_csv(rank_rows, out_dir / "reciprocity_by_rank.csv")

    def arm_summary(category: Category) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for elicitation, mechanism in arms:
            arm = _arm_label(elicitation, mechanism)
            subset = [
                c
                for c in cells
                if c.category == category and _arm_label(c.elicitation, c.mechanism) == arm
            ]
            rows.append(
                {
                    "category": category,
                    "arm": arm,
                    "match_rate": round(_mean([c.match_rate for c in subset]), 3),
                    "match_rate_se": round(_se([c.match_rate for c in subset]), 3),
                    "net_welfare": round(_mean([c.net_welfare_per_customer for c in subset]), 3),
                    "net_welfare_se": round(_se([c.net_welfare_per_customer for c in subset]), 3),
                    "reciprocity_obs": round(_mean([c.reciprocity_obs for c in subset]), 3),
                    "reciprocity_obs_se": round(_se([c.reciprocity_obs for c in subset]), 3),
                }
            )
        return rows

    summary_rows = arm_summary("easy") + arm_summary("hard")
    _write_md_table(summary_rows, out_dir / "arm_summary.md")
    _write_csv(summary_rows, out_dir / "arm_summary.csv")

    def run_regression(outcome: str) -> tuple[list[dict[str, object]], OLSResult]:
        y = [_as_float(r[outcome]) for r in customer_rows]
        x = []
        clusters = []
        for r in customer_rows:
            ai = _as_float(r["ai"])
            central = _as_float(r["central"])
            hard = 1.0 if r["category"] == "hard" else 0.0
            x.append([1.0, ai, central, ai * central, hard])
            clusters.append(str(r["cell_id"]))
        res = ols_cluster_robust(y=y, x=x, clusters=clusters)
        names = ["intercept", "ai", "central", "ai_x_central", "hard"]
        return _format_ols_table(result=res, names=names), res

    def run_regression_heterogeneity(outcome: str) -> tuple[list[dict[str, object]], OLSResult]:
        y = [_as_float(r[outcome]) for r in customer_rows]
        x = []
        clusters = []
        for r in customer_rows:
            ai = _as_float(r["ai"])
            central = _as_float(r["central"])
            hard = 1.0 if r["category"] == "hard" else 0.0
            x.append(
                [
                    1.0,
                    ai,
                    central,
                    ai * central,
                    hard,
                    ai * hard,
                    central * hard,
                    ai * central * hard,
                ]
            )
            clusters.append(str(r["cell_id"]))
        res = ols_cluster_robust(y=y, x=x, clusters=clusters)
        names = [
            "intercept",
            "ai",
            "central",
            "ai_x_central",
            "hard",
            "ai_x_hard",
            "central_x_hard",
            "ai_x_central_x_hard",
        ]
        return _format_ols_table(result=res, names=names), res

    reg_rows_match, _res_m = run_regression("matched")
    _write_md_table(reg_rows_match, out_dir / "reg_matched.md")
    _write_csv(reg_rows_match, out_dir / "reg_matched.csv")

    reg_rows_match_h, _res_mh = run_regression_heterogeneity("matched")
    _write_md_table(reg_rows_match_h, out_dir / "reg_matched_heterogeneity.md")
    _write_csv(reg_rows_match_h, out_dir / "reg_matched_heterogeneity.csv")

    reg_rows_value, _res_v = run_regression("total_value")
    _write_md_table(reg_rows_value, out_dir / "reg_total_value.md")
    _write_csv(reg_rows_value, out_dir / "reg_total_value.csv")

    reg_rows_value_h, _res_vh = run_regression_heterogeneity("total_value")
    _write_md_table(reg_rows_value_h, out_dir / "reg_total_value_heterogeneity.md")
    _write_csv(reg_rows_value_h, out_dir / "reg_total_value_heterogeneity.csv")

    def arm_stat(category: Category, arm: str, key: str) -> float:
        for row in summary_rows:
            if row["category"] == category and row["arm"] == arm:
                return _as_float(row[key])
        raise KeyError(f"Missing {category} {arm} {key}")

    did_easy_match = (
        arm_stat("easy", "ai_central", "match_rate")
        - arm_stat("easy", "standard_central", "match_rate")
        - (
            arm_stat("easy", "ai_search", "match_rate")
            - arm_stat("easy", "standard_search", "match_rate")
        )
    )
    did_hard_match = (
        arm_stat("hard", "ai_central", "match_rate")
        - arm_stat("hard", "standard_central", "match_rate")
        - (
            arm_stat("hard", "ai_search", "match_rate")
            - arm_stat("hard", "standard_search", "match_rate")
        )
    )

    triple_idx = 7
    triple_coef = _res_mh.coef[triple_idx]
    triple_se = _res_mh.se[triple_idx]
    triple_p = _res_mh.p[triple_idx]

    categories: tuple[Category, ...] = ("easy", "hard")
    for category in categories:
        bars_match = []
        for elicitation, mechanism in arms:
            arm = _arm_label(elicitation, mechanism)
            subset = [
                c
                for c in cells
                if c.category == category and _arm_label(c.elicitation, c.mechanism) == arm
            ]
            bars_match.append(Bar(label=arm, value=_mean([c.match_rate for c in subset])))
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_match_rate_by_arm.svg",
            title=f"Match rate by arm (category={category})",
            bars=bars_match,
            y_label="match_rate",
        )

        bars_rec = []
        for elicitation, mechanism in arms:
            arm = _arm_label(elicitation, mechanism)
            subset = [
                c
                for c in cells
                if c.category == category and _arm_label(c.elicitation, c.mechanism) == arm
            ]
            bars_rec.append(Bar(label=arm, value=_mean([c.reciprocity_obs for c in subset])))
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_reciprocity_by_arm.svg",
            title=f"Reciprocity proxy by arm (category={category})",
            bars=bars_rec,
            y_label="P(both accept | recommended)",
        )

        def series_for(arm: tuple[Elicitation, Mechanism], *, cat: Category) -> LineSeries:
            elicitation, mechanism = arm
            pts = []
            for rank in range(1, args.rec_k + 1):
                vals = [
                    _as_float(r["reciprocity_rate"])
                    for r in rank_rows
                    if r["category"] == cat
                    and r["elicitation"] == elicitation
                    and r["mechanism"] == mechanism
                    and int(_as_float(r["rank"])) == rank
                ]
                pts.append((float(rank), _mean(vals)))
            return LineSeries(label=_arm_label(elicitation, mechanism), points=pts)

        write_line_chart_svg(
            out_path=out_dir / f"fig_{category}_reciprocity_curve.svg",
            title=f"Reciprocity vs recommendation rank (category={category})",
            x_label="rank (1=best)",
            y_label="P(both accept)",
            series=[series_for(a, cat=category) for a in arms],
        )

    meta = {
        "seed": args.seed,
        "cells_per_category": args.cells_per_category,
        "market_params": asdict(market_params),
        "rec_k": args.rec_k,
        "search_params": asdict(search_params),
        "attention_cost": args.attention_cost,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readme_lines = [
        "# Field-style simulation (cluster randomization)",
        "",
        "So what: this run simulates the *actual experimental design* (cluster-level assignment),",
        "then estimates the AI × centralized-mechanism interaction using customer-level outcomes",
        "with cluster-robust SEs.",
        "",
        f"Quick read (rec_k={args.rec_k}, attention_cost={args.attention_cost}):",
        f"- match_rate DiD (easy): {did_easy_match:+.3f}",
        f"- match_rate DiD (hard): {did_hard_match:+.3f}",
        f"- triple interaction (ai×central×hard): {triple_coef:+.3f}",
        f"  (SE={triple_se:.3f}, p={triple_p:.3f})",
        "",
        "Design:",
        "- Unit: (city×category×week) cell; assigned to one of 4 arms (2×2).",
        "- Outcomes measured at the customer/job level; inference clustered by cell.",
        "- Describability proxy: P(both accept | recommended) from a thin accept/reject step.",
        "",
        "Artifacts:",
        "- `cells.csv`: one row per cell",
        "- `customers.csv`: one row per customer/job",
        "- `reciprocity_by_rank.csv`: acceptance proxy by rec rank",
        "- `arm_summary.md`: arm means (by category) with Monte Carlo SE",
        "- `reg_matched.md`: clustered regression on match indicator",
        "- `reg_matched_heterogeneity.md`: adds a hard-category triple interaction",
        "- `reg_total_value.md`: clustered regression on realized match value (0 if unmatched)",
        "- `reg_total_value_heterogeneity.md`: adds a hard-category triple interaction",
        "- `fig_*`: bar charts + reciprocity curves",
        "",
        "Run:",
        "- `uv run python -m econ_llm_preferences_experiment.publishable_field_sim`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    log(logger, 20, "field_sim_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
