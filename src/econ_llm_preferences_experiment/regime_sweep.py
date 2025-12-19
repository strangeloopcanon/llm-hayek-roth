from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
)
from econ_llm_preferences_experiment.models import DIMENSIONS, Category
from econ_llm_preferences_experiment.plotting import write_heatmap_svg
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
    preference_density_proxy,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _truncate_weights(weights: tuple[float, ...], k: int) -> tuple[float, ...]:
    if k <= 0:
        return tuple(1.0 / len(weights) for _ in weights)
    top = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:k]
    kept = [weights[i] if i in top else 0.0 for i in range(len(weights))]
    total = sum(kept)
    if total <= 0:
        return tuple(1.0 / len(weights) for _ in weights)
    return tuple(v / total for v in kept)


def run_regime_sweep(
    *,
    category: Category,
    market_params: MarketParams,
    replications: int,
    seed: int,
    attention_cost: float,
    max_k: int,
) -> tuple[list[dict[str, object]], list[list[float]], list[list[float]]]:
    rng = random.Random(seed)  # nosec B311
    customers, providers = generate_population(
        rng=rng,
        category=category,
        n_customers=market_params.n_customers,
        n_providers=market_params.n_providers,
    )

    ks = list(range(1, max_k + 1))
    weights_customer = {k: tuple(_truncate_weights(a.weights, k) for a in customers) for k in ks}
    weights_provider = {k: tuple(_truncate_weights(a.weights, k) for a in providers) for k in ks}

    n_customers = len(customers)
    d_hat_i_by_k: dict[int, list[float]] = {k: [] for k in ks}
    d_hat_j_by_k: dict[int, list[float]] = {k: [] for k in ks}

    search_value_by_k: dict[int, list[float]] = {k: [] for k in ks}
    search_attention_by_k: dict[int, list[float]] = {k: [] for k in ks}
    search_welfare_by_k: dict[int, list[float]] = {k: [] for k in ks}
    search_match_rate_by_k: dict[int, list[float]] = {k: [] for k in ks}

    central_value_by_cell: dict[tuple[int, int], list[float]] = {
        (k_i, k_j): [] for k_i in ks for k_j in ks
    }
    central_attention_by_cell: dict[tuple[int, int], list[float]] = {
        (k_i, k_j): [] for k_i in ks for k_j in ks
    }
    central_welfare_by_cell: dict[tuple[int, int], list[float]] = {
        (k_i, k_j): [] for k_i in ks for k_j in ks
    }
    central_match_rate_by_cell: dict[tuple[int, int], list[float]] = {
        (k_i, k_j): [] for k_i in ks for k_j in ks
    }

    for r in range(replications):
        rep_rng = random.Random(seed * 10_000 + r)  # nosec B311
        market = generate_market_instance(
            rng=rep_rng,
            customers=customers,
            providers=providers,
            idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
        )

        vhat_c_by_k = {
            k: inferred_value_matrix(
                weights_by_agent=weights_customer[k], partner_attributes=market.provider_attributes
            )
            for k in ks
        }
        vhat_p_by_k = {
            k: inferred_value_matrix(
                weights_by_agent=weights_provider[k], partner_attributes=market.customer_attributes
            )
            for k in ks
        }

        for k in ks:
            d_hat_i_by_k[k].append(
                preference_density_proxy(
                    v_true=market.v_customer, v_hat=vhat_c_by_k[k], epsilon=market_params.epsilon
                )
            )
            d_hat_j_by_k[k].append(
                preference_density_proxy(
                    v_true=market.v_provider, v_hat=vhat_p_by_k[k], epsilon=market_params.epsilon
                )
            )

            outcome_search = decentralized_search(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c_by_k[k],
                accept_threshold=market_params.accept_threshold,
                params=SearchParams(max_rounds=30),
            )
            tv_sum = sum(
                market.v_customer[i][j] + market.v_provider[j][i] for i, j in outcome_search.matches
            )
            attn_total = outcome_search.proposals + outcome_search.accept_decisions
            search_value_by_k[k].append(tv_sum / n_customers)
            search_attention_by_k[k].append(attn_total / n_customers)
            search_welfare_by_k[k].append(
                (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
            )
            search_match_rate_by_k[k].append(
                len(outcome_search.matches) / min(len(customers), len(providers))
            )

        for k_i in ks:
            for k_j in ks:
                outcome_central = centralized_recommendations(
                    v_customer_true=market.v_customer,
                    v_provider_true=market.v_provider,
                    v_customer_hat=vhat_c_by_k[k_i],
                    v_provider_hat=vhat_p_by_k[k_j],
                    accept_threshold=market_params.accept_threshold,
                    params=CentralizedParams(rec_k=3),
                )
                tv_sum = sum(
                    market.v_customer[i][j] + market.v_provider[j][i]
                    for i, j in outcome_central.matches
                )
                attn_total = outcome_central.proposals + outcome_central.accept_decisions
                central_value_by_cell[(k_i, k_j)].append(tv_sum / n_customers)
                central_attention_by_cell[(k_i, k_j)].append(attn_total / n_customers)
                central_welfare_by_cell[(k_i, k_j)].append(
                    (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
                )
                central_match_rate_by_cell[(k_i, k_j)].append(
                    len(outcome_central.matches) / min(len(customers), len(providers))
                )

    rows: list[dict[str, object]] = []
    welfare_diff_matrix: list[list[float]] = []
    lambda_star_matrix: list[list[float]] = []
    for k_j in ks:
        welfare_vals: list[float] = []
        lambda_vals: list[float] = []
        for k_i in ks:
            w_c = central_welfare_by_cell[(k_i, k_j)]
            w_s = search_welfare_by_k[k_i]
            diff_series = [w_c[t] - w_s[t] for t in range(replications)]
            diff = _mean(diff_series)
            welfare_vals.append(diff)

            v_s = search_value_by_k[k_i]
            a_s = search_attention_by_k[k_i]
            v_c = central_value_by_cell[(k_i, k_j)]
            a_c = central_attention_by_cell[(k_i, k_j)]

            mean_vs = _mean(v_s)
            mean_vc = _mean(v_c)
            mean_as = _mean(a_s)
            mean_ac = _mean(a_c)
            denom = mean_as - mean_ac
            lambda_star = (mean_vs - mean_vc) / denom if denom > 1e-9 else 0.0
            lambda_vals.append(lambda_star)

            rows.append(
                {
                    "category": category,
                    "k_I": k_i,
                    "k_J": k_j,
                    "d_hat_I": round(_mean(d_hat_i_by_k[k_i]), 3),
                    "d_hat_J": round(_mean(d_hat_j_by_k[k_j]), 3),
                    "search_match_rate": round(_mean(search_match_rate_by_k[k_i]), 3),
                    "search_match_rate_se": round(_se(search_match_rate_by_k[k_i]), 3),
                    "central_match_rate": round(_mean(central_match_rate_by_cell[(k_i, k_j)]), 3),
                    "central_match_rate_se": round(_se(central_match_rate_by_cell[(k_i, k_j)]), 3),
                    "search_total_value_per_customer": round(mean_vs, 3),
                    "search_total_value_per_customer_se": round(_se(v_s), 3),
                    "search_attention_per_customer": round(mean_as, 3),
                    "search_attention_per_customer_se": round(_se(a_s), 3),
                    "search_net_welfare": round(_mean(w_s), 3),
                    "search_net_welfare_se": round(_se(w_s), 3),
                    "central_total_value_per_customer": round(mean_vc, 3),
                    "central_total_value_per_customer_se": round(_se(v_c), 3),
                    "central_attention_per_customer": round(mean_ac, 3),
                    "central_attention_per_customer_se": round(_se(a_c), 3),
                    "central_net_welfare": round(_mean(w_c), 3),
                    "central_net_welfare_se": round(_se(w_c), 3),
                    "net_welfare_diff": round(diff, 3),
                    "net_welfare_diff_se": round(_se(diff_series), 3),
                    "lambda_star": round(lambda_star, 4),
                }
            )
        welfare_diff_matrix.append(welfare_vals)
        lambda_star_matrix.append(lambda_vals)

    return rows, welfare_diff_matrix, lambda_star_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/latest")
    parser.add_argument("--category", choices=["easy", "hard"], default="hard")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    parser.add_argument("--max-k", type=int, default=len(DIMENSIONS))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_params = MarketParams()
    rows, welfare_diff, lambda_star = run_regime_sweep(
        category=args.category,
        market_params=market_params,
        replications=args.replications,
        seed=args.seed,
        attention_cost=args.attention_cost,
        max_k=args.max_k,
    )

    csv_path = out_dir / f"regime_grid_{args.category}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ks = list(range(1, args.max_k + 1))
    x_labels = [str(k) for k in ks]
    y_labels = [str(k) for k in ks]
    write_heatmap_svg(
        out_path=out_dir / f"regime_map_{args.category}_net_welfare_diff.svg",
        title=f"Central - Search net welfare (category={args.category})",
        x_labels=x_labels,
        y_labels=y_labels,
        values=welfare_diff,
    )
    write_heatmap_svg(
        out_path=out_dir / f"regime_map_{args.category}_lambda_star.svg",
        title=f"ROI boundary λ*: central beats search if λ > λ* (category={args.category})",
        x_labels=x_labels,
        y_labels=y_labels,
        values=lambda_star,
    )

    meta = {
        "category": args.category,
        "seed": args.seed,
        "replications": args.replications,
        "attention_cost": args.attention_cost,
        "max_k": args.max_k,
        "market_params": asdict(market_params),
    }
    (out_dir / f"regime_meta_{args.category}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
