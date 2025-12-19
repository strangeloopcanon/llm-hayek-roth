from __future__ import annotations

from dataclasses import dataclass

from econ_llm_preferences_experiment.models import MatchOutcome


@dataclass(frozen=True)
class Metrics:
    match_rate: float
    mean_customer_value: float
    mean_provider_value: float
    mean_total_value: float
    proposals_per_match: float
    accept_decisions_per_match: float
    mean_rounds: float


def compute_metrics(
    *,
    outcome: MatchOutcome,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
) -> Metrics:
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)
    denom = min(n_c, n_p)
    m = len(outcome.matches)
    match_rate = m / denom if denom else 0.0

    if m == 0:
        return Metrics(
            match_rate=0.0,
            mean_customer_value=0.0,
            mean_provider_value=0.0,
            mean_total_value=0.0,
            proposals_per_match=0.0,
            accept_decisions_per_match=0.0,
            mean_rounds=float(outcome.rounds),
        )

    c_vals = [v_customer_true[i][j] for i, j in outcome.matches]
    p_vals = [v_provider_true[j][i] for i, j in outcome.matches]
    mean_c = sum(c_vals) / m
    mean_p = sum(p_vals) / m
    total = (sum(c_vals) + sum(p_vals)) / m

    return Metrics(
        match_rate=match_rate,
        mean_customer_value=mean_c,
        mean_provider_value=mean_p,
        mean_total_value=total,
        proposals_per_match=outcome.proposals / m if m else 0.0,
        accept_decisions_per_match=outcome.accept_decisions / m if m else 0.0,
        mean_rounds=float(outcome.rounds),
    )
