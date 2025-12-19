from __future__ import annotations

from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
)


def test_matching_outputs_are_feasible() -> None:
    v_c_true = (
        (0.9, 0.1),
        (0.2, 0.8),
    )
    v_p_true = (
        (0.9, 0.2),
        (0.1, 0.8),
    )
    v_c_hat = v_c_true
    v_p_hat = v_p_true

    out_s = decentralized_search(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        accept_threshold=0.5,
        params=SearchParams(max_rounds=3),
    )
    assert len(out_s.matches) <= 2

    out_c = centralized_recommendations(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.5,
        params=CentralizedParams(rec_k=2),
    )
    assert len(out_c.matches) <= 2
