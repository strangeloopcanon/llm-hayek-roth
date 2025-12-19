from __future__ import annotations

import random

from econ_llm_preferences_experiment.experiment2_congestion import CongestionParams, _simulate_cell
from econ_llm_preferences_experiment.models import AgentTruth


def test_simulate_cell_saturation_changes_message_volume() -> None:
    customers = tuple(
        AgentTruth(
            agent_id=f"c{i:03d}",
            side="customer",
            category="hard",
            weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6),
        )
        for i in range(6)
    )
    providers = tuple(
        AgentTruth(
            agent_id=f"p{j:03d}",
            side="provider",
            category="hard",
            weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6),
        )
        for j in range(4)
    )

    # Everyone likes everyone; ensures matching possible.
    v_c = tuple(tuple(0.9 for _ in providers) for _ in customers)
    v_p = tuple(tuple(0.9 for _ in customers) for _ in providers)
    vhat = tuple(tuple(0.5 for _ in providers) for _ in customers)

    params = CongestionParams(
        horizon_days=2,
        k_manual_per_day=1,
        k_agent_per_day=3,
        provider_daily_response_cap=10,
        provider_weekly_capacity=10,
        customer_accept_threshold=0.1,
        provider_accept_threshold=0.1,
        attention_cost=0.0,
    )
    out0 = _simulate_cell(
        rng=random.Random(0),
        saturation=0.0,
        congestion_params=params,
        customers=customers,
        providers=providers,
        v_customer_true=v_c,
        v_provider_true=v_p,
        vhat_customer_standard=vhat,
        vhat_customer_ai=vhat,
    )
    out1 = _simulate_cell(
        rng=random.Random(0),
        saturation=1.0,
        congestion_params=params,
        customers=customers,
        providers=providers,
        v_customer_true=v_c,
        v_provider_true=v_p,
        vhat_customer_standard=vhat,
        vhat_customer_ai=vhat,
    )

    assert out1.messages_sent_per_customer > out0.messages_sent_per_customer
    assert 0.0 <= out0.match_rate_all <= 1.0
    assert 0.0 <= out1.match_rate_all <= 1.0
